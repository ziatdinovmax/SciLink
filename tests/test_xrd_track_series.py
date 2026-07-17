"""Offline tests for track_phase_series (in-situ series phase tracking).

Deterministic, no LLM, no network: a tiny library built from local CIFs, a
synthetic crossfade series (Si -> NaCl over 11 frames at known fractions),
endmembers passed in each of the three accepted reference forms. Machinery
under test: per-frame joint MILP tracking, onset/coexistence detection,
residual alerts for a transient intermediate, stride."""

from __future__ import annotations

import numpy as np
import pytest

pymatgen = pytest.importorskip("pymatgen.core")
pytest.importorskip("pulp")
from pymatgen.core import Lattice, Structure  # noqa: E402

from scilink.skills.structure_matching.xrd.fingerprint import (  # noqa: E402
    build_fingerprint_library)
from scilink.skills.structure_matching.xrd.track_phase_series import (  # noqa: E402
    track_phase_series, TOOL_SPEC as TS_SPEC)

LAM = 1.5406


def _write(structure, path):
    path.write_text(structure.to(fmt="cif"))


@pytest.fixture()
def tiny_library(tmp_path):
    cifs = tmp_path / "cifs"
    cifs.mkdir()
    _write(Structure.from_spacegroup("Fd-3m", Lattice.cubic(5.43088), ["Si"], [[0, 0, 0]]),
           cifs / "si.cif")
    _write(Structure.from_spacegroup("Fm-3m", Lattice.cubic(5.6402), ["Na", "Cl"],
                                     [[0, 0, 0], [0.5, 0.5, 0.5]]), cifs / "nacl.cif")
    _write(Structure.from_spacegroup("P4_2/mnm", Lattice.tetragonal(4.594, 2.959),
                                     ["Ti", "O"], [[0, 0, 0], [0.305, 0.305, 0]]),
           cifs / "rutile.cif")
    out = tmp_path / "lib.parquet"
    assert build_fingerprint_library(str(cifs), str(out))["n_indexed"] == 3
    return str(out)


def _entry_lines(library_path, formula):
    import pandas as pd
    row = pd.read_parquet(library_path).set_index("formula").loc[formula]
    d = np.asarray(row["ds"], dtype=float)
    tt = 2.0 * np.degrees(np.arcsin(np.clip(LAM / (2.0 * d), -1, 1)))
    return tt.tolist(), np.asarray(row["intensities"], dtype=float).tolist(), \
        str(row["source_id"]) if "source_id" in row else None


def _source_id(library_path, formula):
    import pandas as pd
    df = pd.read_parquet(library_path)
    return str(df[df["formula"] == formula]["source_id"].iloc[0])


def _crossfade_series(library_path, n=11, intruder_frames=()):
    """Si -> NaCl linear crossfade; optionally inject rutile lines mid-series."""
    si_tt, si_ii, _ = _entry_lines(library_path, "Si")
    na_tt, na_ii, _ = _entry_lines(library_path, "NaCl")
    ru_tt, ru_ii, _ = _entry_lines(library_path, "TiO2")
    frames = []
    for k in range(n):
        f = k / (n - 1)
        tt = list(si_tt) + list(na_tt)
        ii = [(1 - f) * v for v in si_ii] + [f * v for v in na_ii]
        if k in intruder_frames:
            tt += list(ru_tt)
            ii += [1.0 * v for v in ru_ii]
        order = np.argsort(tt)
        tt_s = np.asarray(tt)[order]
        ii_s = np.asarray(ii)[order]
        # detection floor: a real extract_peaks never reports zero-intensity
        # peaks, and keeping them would make the two endpoint frames
        # positionally identical (degenerate references)
        m = ii_s > 1.0
        frames.append({"positions": tt_s[m].tolist(),
                       "intensities": ii_s[m].tolist(),
                       "label": 300 + 10 * k})
    return frames


def test_tool_registered_with_knobs():
    from scilink.skills._shared._registry import get_tools_for
    names = {t.name for t in get_tools_for("structure_matching", active_skills=["xrd"])}
    assert "track_phase_series" in names
    for knob in ("tol_deg", "onset_threshold", "residual_alert_frac",
                 "frame_stride", "fit_lattice_scale"):
        assert knob in TS_SPEC.parameters, f"knob {knob} undocumented"


def test_crossfade_tracked_with_empirical_endmembers(tiny_library):
    frames = _crossfade_series(tiny_library)
    # empirical form: the pure endpoint frames ARE the references
    endmembers = [dict(frames[0], label="A"), dict(frames[-1], label="B")]
    r = track_phase_series(frames, endmembers)
    a = [sh["A"] for sh in r["shares"]]
    b = [sh["B"] for sh in r["shares"]]
    assert a[0] > 0.9 and b[0] < 0.05          # pure A at start
    assert b[-1] > 0.9 and a[-1] < 0.05        # pure B at end
    # monotone crossfade (tolerate small MILP wiggle)
    assert all(a[i] >= a[i + 1] - 0.05 for i in range(len(a) - 1))
    assert all(b[i] <= b[i + 1] + 0.05 for i in range(len(b) - 1))
    assert r["coexistence_frames"], "no coexistence window detected"
    assert r["phase_events"]["B"]["onset_frame"] >= 1
    assert max(r["residual_frac"]) < 0.15      # endmembers explain everything


def test_library_and_simulated_reference_forms(tiny_library):
    frames = _crossfade_series(tiny_library, n=5)
    si_tt, si_ii, _ = _entry_lines(tiny_library, "Si")
    endmembers = [
        {"source_id": _source_id(tiny_library, "NaCl")},          # library form
        {"sim_two_theta": si_tt, "sim_intensity": si_ii,          # simulated form
         "label": "Si-sim"},
    ]
    r = track_phase_series(frames, endmembers, library_path=tiny_library)
    assert r["endmembers"][0] == "NaCl"        # label resolved from the library
    assert r["shares"][0]["Si-sim"] > 0.9
    assert r["shares"][-1]["NaCl"] > 0.9


def test_transient_intermediate_alert(tiny_library):
    frames = _crossfade_series(tiny_library, intruder_frames=(5, 6))
    endmembers = [dict(frames[0], label="A"), dict(frames[-1], label="B")]
    r = track_phase_series(frames, endmembers)
    assert set(r["residual_alert_frames"]) == {5, 6}
    clean = [rf for k, rf in enumerate(r["residual_frac"]) if k not in (5, 6)]
    assert max(clean) < 0.25


def test_alert_loop_converges_after_adding_intermediate(tiny_library):
    # The documented two-pass workflow: alerts flag the transient intermediate,
    # the discovered phase is ADDED to the endmember set, and re-tracking
    # clears the alerts while giving the intermediate its own share trace.
    frames = _crossfade_series(tiny_library, intruder_frames=(5, 6))
    ru_tt, ru_ii, _ = _entry_lines(tiny_library, "TiO2")
    endmembers = [dict(frames[0], label="A"), dict(frames[-1], label="B"),
                  {"sim_two_theta": ru_tt, "sim_intensity": ru_ii, "label": "I"}]
    r = track_phase_series(frames, endmembers)
    assert r["residual_alert_frames"] == []
    shares_i = [sh["I"] for sh in r["shares"]]
    assert shares_i[5] > 0.1 and shares_i[6] > 0.1     # intermediate traced
    assert max(shares_i[:5] + shares_i[7:]) < 0.05     # and only where present
    assert r["phase_events"]["I"]["onset_frame"] == 5
    assert r["phase_events"]["I"]["final_frame"] == 6


def test_stride_and_events(tiny_library):
    frames = _crossfade_series(tiny_library)
    endmembers = [dict(frames[0], label="A"), dict(frames[-1], label="B")]
    r = track_phase_series(frames, endmembers, frame_stride=2)
    assert r["n_processed"] == 6
    assert r["frame_indices"] == [0, 2, 4, 6, 8, 10]
    # events are ORIGINAL series indices, not processed positions
    assert r["phase_events"]["A"]["onset_frame"] == 0
    assert r["phase_events"]["B"]["final_frame"] == 10


def test_input_guards(tiny_library):
    frames = _crossfade_series(tiny_library, n=3)
    with pytest.raises(ValueError):
        track_phase_series([], [dict(frames[0], label="A")])
    with pytest.raises(ValueError):
        track_phase_series(frames, [])
    with pytest.raises(ValueError):
        track_phase_series(frames, [{"label": "no-pattern-keys"}])
    with pytest.raises(ValueError):
        track_phase_series(frames, [{"source_id": "does-not-exist"}],
                           library_path=tiny_library)
