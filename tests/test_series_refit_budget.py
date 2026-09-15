"""max_series_refits: cap on independent per-unit re-analyses in a series.

Worst locked-model fits are re-analyzed first; the rest keep their locked
result and are listed under refit_skipped_by_budget. None = unlimited
(unchanged behaviour); 0 = no refits.
"""
import logging
from types import SimpleNamespace

from scilink.agents.exp_agents.controllers.curve_fitting_controllers import (
    AdaptiveRefitController,
)


class _Ctl(AdaptiveRefitController):
    """Records which units would be re-analyzed instead of running LLM loops."""

    def __init__(self):
        self.logger = logging.getLogger("t")
        self.enable_human_feedback = False
        self.refitted = []
        self._fitting_helper = SimpleNamespace(_detect_outliers=lambda results: [])

    def _load_spectrum(self, idx, *a, **k):
        self.refitted.append(idx)
        return None          # "could not load" -> loop continues, no LLM work


def _state(max_refits, n=5):
    flagged = [{"index": i, "name": f"u{i}", "reason": "below_threshold",
                "r_squared": r2} for i, r2 in enumerate([0.93, 0.80, 0.91, None, 0.70])]
    return {"num_spectra": n, "is_single_spectrum": False,
            "flagged_spectra": flagged, "series_results": [{}] * n,
            "spectrum_paths": [f"p{i}" for i in range(n)],
            "max_series_refits": max_refits, "locked_fitting_config": {}}


def test_unlimited_by_default():
    c = _Ctl(); st = c.execute(_state(None))
    assert sorted(c.refitted) == [0, 1, 2, 3, 4]
    assert st["refit_skipped_by_budget"] == []


def test_cap_takes_worst_first_and_lists_skipped():
    c = _Ctl(); st = c.execute(_state(2))
    # worst R² first: 0.70 (idx 4) then 0.80 (idx 1); a None R² sorts last
    assert c.refitted == [4, 1]
    assert [s["index"] for s in st["refit_skipped_by_budget"]] == [2, 0, 3]
    assert st["refit_skipped_by_budget"][0]["r_squared"] == 0.91


def test_zero_means_no_refits():
    c = _Ctl(); st = c.execute(_state(0))
    assert c.refitted == []
    assert len(st["refit_skipped_by_budget"]) == 5
