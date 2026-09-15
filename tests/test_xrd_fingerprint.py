"""Offline tests for the fingerprint search-match route and zero calibration.

The fingerprint library + search is DETERMINISTIC (no LLM, no network): build a
tiny library from local CIFs, query with synthetic measured peaks, assert the
right phase ranks first. Grading in the real benchmark showed matches carry
measured occupancies/hydrates (e.g. dolomite as Ca3.006Mg2.994C6O18), so tests
compare by element set where exact strings would be brittle.
"""

from __future__ import annotations

import numpy as np
import pytest

pymatgen = pytest.importorskip("pymatgen.core")
from pymatgen.core import Lattice, Structure  # noqa: E402

from scilink.skills.structure_matching.xrd.fingerprint import (  # noqa: E402
    build_fingerprint_library, search_match_pattern, TOOL_SPEC as FP_SPEC)
from scilink.skills.structure_matching.xrd.calibrate_zero import (  # noqa: E402
    calibrate_zero, _reference_two_theta, TOOL_SPEC as CZ_SPEC)


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
    # NOTE: diamond would be skipped here (only 2 lines below 90 deg 2-theta —
    # the min_lines curation working as designed), so use rutile as the third.
    _write(Structure.from_spacegroup("P4_2/mnm", Lattice.tetragonal(4.594, 2.959),
                                     ["Ti", "O"], [[0, 0, 0], [0.305, 0.305, 0]]),
           cifs / "rutile.cif")
    out = tmp_path / "lib.parquet"
    summary = build_fingerprint_library(str(cifs), str(out))
    assert summary["n_indexed"] == 3
    return str(out)


def test_tools_registered():
    from scilink.skills._shared._registry import get_tools_for
    names = {t.name for t in get_tools_for("structure_matching", active_skills=["xrd"])}
    assert "search_match_pattern" in names
    assert "calibrate_zero" in names
    # knob docs present
    assert "tol_deg" in FP_SPEC.parameters and "n_key_lines" in FP_SPEC.parameters
    assert "fit_displacement" in CZ_SPEC.parameters


def test_search_match_identifies_si(tiny_library):
    # Si Cu-Ka peak list (positions + realistic relative intensities)
    tt = [28.442, 47.303, 56.121, 69.130, 76.377, 88.032]
    ii = [100.0, 55.0, 30.0, 6.0, 11.0, 12.0]
    r = search_match_pattern(tt, ii, wavelength="CuKa", library_path=tiny_library)
    assert r["matches"], "no matches returned"
    assert r["matches"][0]["formula"] == "Si"
    assert r["matches"][0]["figure_of_merit"] > 0.8
    # identification -> structure bridge: a locally built library resolves the
    # top hits' structure_path from its own CIFs (no network)
    import os
    sp = r["matches"][0]["structure_path"]
    assert sp and os.path.exists(sp)
    # the wrong-lattice phases must NOT outrank Si
    forms = [m["formula"] for m in r["matches"]]
    assert forms.index("Si") == 0


def test_materialize_dangling_path_falls_back_gracefully(tiny_library, tmp_path,
                                                         monkeypatch):
    # Distributed-artifact case: cif_path points at the (deleted) build machine
    # paths. Non-COD ids + no network must yield structure_path=None, not an
    # error; matching itself is unaffected.
    import pandas as pd
    from scilink.skills.structure_matching.xrd import fingerprint as fp
    df = pd.read_parquet(tiny_library)
    df["cif_path"] = "/nonexistent/build/machine/path.cif"   # dangling
    lib2 = tmp_path / "dangling.parquet"
    df.to_parquet(lib2, index=False)
    monkeypatch.setenv("SCILINK_COD_ALLOW_WEB", "0")          # simulate offline
    tt = [28.442, 47.303, 56.121, 69.130, 76.377, 88.032]
    ii = [100.0, 55.0, 30.0, 6.0, 11.0, 12.0]
    r = search_match_pattern(tt, ii, wavelength="CuKa", library_path=str(lib2),
                             materialize_dir=str(tmp_path / "m"))
    assert r["matches"][0]["formula"] == "Si"                 # matching unaffected
    assert r["matches"][0]["structure_path"] is None          # honest None, no crash


def test_search_match_rejects_when_absent(tiny_library):
    # quartz peaks against a library with no quartz: no high-confidence match
    tt = [20.86, 26.64, 36.54, 39.47, 50.14, 59.96]
    ii = [22.0, 100.0, 8.0, 8.0, 14.0, 9.0]
    r = search_match_pattern(tt, ii, wavelength="CuKa", library_path=tiny_library)
    top_fom = r["matches"][0]["figure_of_merit"] if r["matches"] else 0.0
    assert top_fom < 0.7   # nothing convincing — falls to the indexing route


def test_absent_lines_evidence_and_penalty(tiny_library, tmp_path):
    # Absent-strong-lines negative evidence: build an IMPOSTOR entry = Si's
    # lines PLUS extra strong lines that the measurement lacks (the
    # superlattice-impostor signature). Default (penalty=0): evidence reported,
    # ranking by FOM unchanged. With penalty>0: impostor demoted below Si.
    import pandas as pd
    from scilink.skills.structure_matching.xrd import fingerprint as fp
    df = pd.read_parquet(tiny_library)
    si = df[df["formula"] == "Si"].iloc[0]
    imp = dict(si)
    imp["source_id"] = "impostor"
    imp["formula"] = "XxSi"
    # Si's own lines + two strong fake superlattice lines at 24 and 40 deg
    lam = 1.5406
    import numpy as np
    # inside the measured window (28.4-88.0) and >tol from any Si line
    fake_d = [lam / (2 * np.sin(np.radians(t / 2))) for t in (33.0, 40.0)]
    imp["ds"] = list(si["ds"]) + fake_d
    imp["intensities"] = list(si["intensities"]) + [80.0, 60.0]
    lib2 = tmp_path / "with_impostor.parquet"
    pd.concat([df, pd.DataFrame([imp])], ignore_index=True).to_parquet(lib2)

    tt = [28.442, 47.303, 56.121, 69.130, 76.377, 88.032]
    ii = [100.0, 55.0, 30.0, 6.0, 11.0, 12.0]

    r0 = fp.search_match_pattern(tt, ii, wavelength="CuKa", library_path=str(lib2),
                                 materialize_top=0)
    by_id = {m["source_id"]: m for m in r0["matches"]}
    assert "impostor" in by_id
    assert by_id["impostor"]["frac_strong_lines_absent"] > 0.2   # evidence reported
    assert 33.0 in by_id["impostor"]["absent_strong_lines"]
    sim = [m for m in r0["matches"] if m["formula"] == "Si"][0]
    assert sim["frac_strong_lines_absent"] == 0.0                # truth: none absent
    # default penalty=0: adjusted == fom (ranking semantics unchanged)
    assert by_id["impostor"]["adjusted_score"] == pytest.approx(
        by_id["impostor"]["figure_of_merit"], abs=1e-4)  # tool rounds to 4 dp

    r1 = fp.search_match_pattern(tt, ii, wavelength="CuKa", library_path=str(lib2),
                                 materialize_top=0, absent_lines_penalty=0.5)
    forms = [m["formula"] for m in r1["matches"]]
    assert forms.index("Si") < forms.index("XxSi")               # impostor demoted


def test_search_match_needs_peaks(tiny_library):
    with pytest.raises(ValueError):
        search_match_pattern([28.4], [100.0], library_path=tiny_library)


def test_fetch_and_default_store_loading(tiny_library, tmp_path, monkeypatch):
    # fetch: file:// URL -> dest, checksum verified, atomic install; loader:
    # with no explicit path and no env var, the per-user store is found.
    import hashlib
    from scilink.skills.structure_matching.xrd import fingerprint as fp

    data = open(tiny_library, "rb").read()
    good_sha = hashlib.sha256(data).hexdigest()
    url = "file://" + tiny_library
    dest = tmp_path / "store" / "cod_fingerprints.parquet"

    info = fp.fetch_fingerprint_library(url=url, dest=str(dest), sha256=good_sha)
    assert info["n_entries"] == 3 and info["sha256"] == good_sha
    assert dest.exists()

    # refuses to clobber without overwrite=True
    with pytest.raises(FileExistsError):
        fp.fetch_fingerprint_library(url=url, dest=str(dest))
    fp.fetch_fingerprint_library(url=url, dest=str(dest), sha256=good_sha,
                                 overwrite=True)

    # checksum mismatch -> rejected, nothing installed
    bad_dest = tmp_path / "store2" / "lib.parquet"
    with pytest.raises(RuntimeError):
        fp.fetch_fingerprint_library(url=url, dest=str(bad_dest), sha256="0" * 64)
    assert not bad_dest.exists()

    # loader resolution: default per-user store found when nothing else is set
    monkeypatch.delenv(fp._ENV_DB, raising=False)
    monkeypatch.setattr(fp, "_default_store_path", lambda: str(dest))
    df = fp._load_library(None)
    assert len(df) == 3

    # nothing anywhere -> actionable error naming all three routes
    monkeypatch.setattr(fp, "_default_store_path",
                        lambda: str(tmp_path / "nowhere.parquet"))
    fp._LIB_CACHE.clear()
    with pytest.raises(RuntimeError) as exc:
        fp._load_library(None)
    msg = str(exc.value)
    assert "fetch-xrd-library" in msg and fp._ENV_DB in msg and "build" in msg


def test_calibrate_zero_reference_lines():
    # Si @ CuKa: textbook positions
    ref = _reference_two_theta("Si", 1.5406, 95.0)
    for expect in (28.44, 47.30, 56.12, 69.13, 76.38, 88.03):
        assert np.min(np.abs(ref - expect)) < 0.02


def test_calibrate_zero_recovers_correction():
    # distort standard + sample lines by zero+displacement; corrected sample
    # peaks must land back on truth (the SUM of the terms is what matters —
    # the zero/disp split is documented as ill-conditioned).
    ref = _reference_two_theta("Si", 1.5406, 90.0)
    sample = [24.1, 33.9, 41.2, 54.8, 62.3]
    def distort(t):
        return t + 0.08 - 0.05 * np.cos(np.radians(t / 2.0))
    peaks = sorted(distort(t) for t in list(sample) + [float(x) for x in ref])
    r = calibrate_zero(peaks, standard="Si", wavelength="CuKa")
    assert r["n_lines_matched"] >= 5
    assert r["residual_rms_deg"] < 0.02
    corr = sorted(r["corrected_peaks"])
    err = max(abs(c - t) for c, t in zip(corr, sorted(sample)))
    assert err < 0.02


def test_calibrate_zero_guards():
    with pytest.raises(ValueError):
        calibrate_zero([28.4, 47.3], standard="Si")           # too few peaks
    with pytest.raises(ValueError):
        calibrate_zero([10.0, 20.0, 30.0, 40.0], standard="Si")  # no lines match
    with pytest.raises(ValueError):
        calibrate_zero([28.4, 47.3, 56.1], standard="quartzite")  # unknown standard