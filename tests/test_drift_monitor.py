"""The graded, model-free change signal (scilink/live/drift.py).

It reads the data only, never the recipe: review of PR 656 showed the same real
series flagging most frames under one locked recipe and none under another.
"""

import numpy as np
import pytest

from scilink.live.drift import DriftMonitor

X = np.linspace(0.0, 20.0, 800)


def curve(peaks, noise=0.02, seed=0, base=0.3, slope=0.01):
    y = base + slope * X + sum(h * np.exp(-0.5 * ((X - c) / w) ** 2) for c, h, w in peaks)
    return X, y + np.random.default_rng(seed).normal(0, noise, X.size)


TWO = [(6.0, 3.0, 0.5), (12.0, 1.4, 0.9)]


def stream(mon, frames):
    out = []
    for x, y in frames:
        v = mon.judge(x, y)
        if not v["suspected"]:
            mon.learn(x, y, v)
        out.append(v)
    return out


def armed(n=6):
    mon = DriftMonitor()
    mon.seed([curve(TWO, seed=1)])
    stream(mon, [curve(TWO, seed=10 + i) for i in range(n)])
    return mon


def test_the_same_physics_is_nothing_new_whatever_its_strength():
    mon = armed()
    strong = [(c, 3.0 * h, w) for c, h, w in TWO]
    # a few frames have to SHOW that strength varies before it counts as normal
    stream(mon, [curve([(c, k * h, w) for c, h, w in TWO], seed=30 + i)
                 for i, k in enumerate((1.3, 0.7, 1.8, 2.4))])
    v = mon.judge(*curve(strong, seed=40))
    assert not v["suspected"] and v["fraction"] < 0.05


def test_a_new_feature_is_graded_not_just_flagged():
    mon = armed()
    small = mon.judge(*curve(TWO + [(16.5, 0.6, 0.7)], seed=50))
    large = mon.judge(*curve(TWO + [(16.5, 4.0, 0.7)], seed=51))
    assert small["suspected"] and large["suspected"]
    assert 0.1 < small["fraction"] < large["fraction"] <= 1.0            # a magnitude, not a bit
    assert large["from_reference"] > small["from_reference"]


def test_a_slow_drift_is_followed_and_its_distance_reported():
    mon = armed()
    frames = [curve([(6.0 + 0.01 * i, 3.0, 0.5), (12.0, 1.4, 0.9)], seed=60 + i) for i in range(60)]
    verdicts = stream(mon, frames)
    assert not any(v["suspected"] for v in verdicts)
    assert verdicts[-1]["from_reference"] > 3 * verdicts[0]["from_reference"]   # it did move


def test_a_weak_change_that_persists_is_caught_together():
    # Material (a fifth of the structure) but, frame by frame, within the noise.
    def run(noise, height, n):
        mon = armed(10)
        stream(mon, [curve(TWO, noise=noise, seed=200 + i) for i in range(8)])
        return [mon.judge(*curve(TWO + [(16.5, height, 1.2)], noise=noise, seed=70 + i)) for i in range(n)]
    soon = run(0.2, 0.5, 4)
    assert not soon[0]["suspected"] and any(v["suspected"] for v in soon)       # within a few frames
    later = run(0.3, 0.6, 7)
    assert not any(v["suspected"] for v in later[:3]) and any(v["suspected"] for v in later)
    quiet = armed(10)
    stream(quiet, [curve(TWO, noise=0.3, seed=200 + i) for i in range(8)])
    assert not any(quiet.judge(*curve(TWO, noise=0.3, seed=300 + i))["suspected"] for i in range(10))


def test_a_glitch_is_one_frame_not_three():
    mon = armed()
    spike = curve(TWO, seed=80)
    spike[1][400] += 40.0
    assert mon.judge(*spike)["suspected"]
    assert not any(mon.judge(*curve(TWO, seed=81 + i))["suspected"] for i in range(3))


def test_held_frames_become_normal_only_when_adopted():
    mon = armed()
    new = [curve(TWO + [(16.5, 3.0, 0.7)], seed=90 + i) for i in range(4)]
    assert all(mon.judge(*f)["suspected"] for f in new[:3])
    assert mon.n_held == 3 and mon.held_agree()
    assert mon.adopt() == 3 and mon.n_held == 0
    assert not mon.judge(*new[3])["suspected"]
    moving = armed()
    moving.judge(*curve(TWO + [(16.5, 3.0, 0.7)], seed=95))
    moving.judge(*curve(TWO + [(3.0, 5.0, 0.4)], seed=96))
    assert not moving.held_agree()                                      # still changing: not a state


def test_a_shorter_window_is_compared_where_it_overlaps():
    mon = armed()
    x, y = curve(TWO, seed=100)
    same = mon.judge(x[:520], y[:520])                                  # the scan ended early
    assert same["available"] and same["window_share"] < 0.7 and not same["suspected"]
    xs, ys = curve([(6.0, 3.0, 0.5), (9.0, 4.0, 0.4)], seed=101)
    assert mon.judge(xs[:520], ys[:520])["suspected"]
    assert mon.judge(x[:100], y[:100]) == {"available": False}          # too little in common


def test_nothing_is_judged_before_a_few_frames_are_known():
    mon = DriftMonitor(warmup=4)
    mon.seed([curve(TWO, seed=1)])
    v = mon.judge(*curve(TWO + [(16.5, 4.0, 0.7)], seed=2))
    assert v["learning"] and not v["suspected"] and v["fraction"] > 0.3   # measured, not judged


def test_deterministic_and_survives_a_restart():
    mon = armed()
    frame = curve(TWO + [(16.5, 1.0, 0.7)], seed=110)
    again = DriftMonitor()
    again.load_state(mon.to_state())
    a, b = mon.judge(*frame), again.judge(*frame)
    assert (a["fraction"], a["suspected"]) == (b["fraction"], b["suspected"])


def test_the_knobs_do_something():
    frame = curve(TWO + [(16.5, 0.6, 0.7)], seed=120)
    strict, lax = armed(), armed()
    lax.fraction_bar = 0.6
    assert strict.judge(*frame)["suspected"] and not lax.judge(*frame)["suspected"]


def test_a_state_accepted_once_is_never_asked_about_again():
    # Review-driven, seen live on a real line scan: crossing the same kind of
    # region again, after the rolling window had forgotten it, asked again.
    mon = DriftMonitor(window=8)
    mon.seed([curve(TWO, seed=1)])
    stream(mon, [curve(TWO, seed=10 + i) for i in range(6)])
    other = lambda s: curve(TWO + [(16.5, 3.0, 0.7)], seed=s)                   # noqa: E731
    assert all(mon.judge(*other(20 + i))["suspected"] for i in range(3))
    mon.adopt()
    stream(mon, [other(30 + i) for i in range(12)])                   # the window now holds only the new state
    back = mon.judge(*curve(TWO, seed=50))                            # ...and the first one comes back
    assert not back["suspected"] and "known_state" in back
    again = DriftMonitor()
    again.load_state(mon.to_state())
    assert not again.judge(*curve(TWO, seed=51))["suspected"]


def test_a_new_recipe_keeps_what_the_stream_has_shown():
    mon = armed()
    new = [curve(TWO + [(16.5, 3.0, 0.7)], seed=60 + i) for i in range(4)]
    for f in new[:3]:
        mon.judge(*f)                                                 # held while a recipe is rebuilt
    mon.seed([new[0]], keep=True)
    assert not mon.judge(*new[3])["suspected"]                        # the held frames were kept
    assert not mon.judge(*curve(TWO, seed=70))["suspected"]           # and so was the earlier state
    fresh = armed()
    fresh.seed([new[0]])                                              # a different stream: start over
    assert fresh.n_learned == 1
