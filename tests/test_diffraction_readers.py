"""scilink.utils.readers.diffraction: X-ray diffraction instrument files and their wiring.

Every file is built here with known contents (the layouts were checked against
real Panalytical, Bruker and Rigaku files), so each check is against values the
test chose: the pattern, the instrument record, what resolve_wavelength /
resolve_geometry make of it, the merge with what the person gave, and the
curve agent's use of it.
"""

import logging
import struct
import types
import zipfile

import numpy as np
import pytest

from scilink.utils.readers import (
    can_read, file_metadata, merge_file_metadata, read_dataset, read_datasets,
    records_calibration,
)
from scilink.utils.readers._model import UnsupportedContent
from scilink.skills.structure_matching.xrd.wavelength import resolve_geometry, resolve_wavelength

X = np.linspace(10.0, 20.0, 11)
Y = np.array([100, 120, 150, 900, 3000, 800, 160, 130, 110, 105, 100], float)


# ---------------------------------------------------------------- writers --
def xrdml(path, *, version="2.2", ratio="0.5000", mono=None, mode="Reflection", element="counts",
          y=Y, att=None, positions=None, scans=1, scan_axis="Gonio", filt="Beta-filter Nickel"):
    mono_xml = (f'<monochromator id="1" name="{mono}"><crystal>Ge</crystal></monochromator>'
                if mono else "")
    filt_xml = f'<filter id="2" name="{filt}"><material>Ni</material></filter>' if filt else ""
    if positions is None:
        pos = (f'<positions axis="2Theta" unit="deg"><startPosition>{X[0]}</startPosition>'
               f'<endPosition>{X[-1]}</endPosition></positions>'
               f'<positions axis="Omega" unit="deg"><startPosition>{X[0] / 2}</startPosition>'
               f'<endPosition>{X[-1] / 2}</endPosition></positions>')
    else:
        pos = (f'<positions axis="2Theta" unit="deg"><listPositions>{" ".join(map(str, positions))}'
               '</listPositions></positions>')
    att_xml = f'<beamAttenuationFactors>{" ".join(map(str, att))}</beamAttenuationFactors>' if att is not None else ""
    scan = (f'<scan appendNumber="{{n}}" mode="Continuous" scanAxis="{scan_axis}" status="Completed"><header>'
            '<startTimeStamp>2024-04-12T22:32:06-07:00</startTimeStamp><source>'
            '<applicationSoftware version="1">Data Collector</applicationSoftware>'
            '<instrumentControlSoftware version="2">EMPYREAN</instrumentControlSoftware></source></header>'
            f'<dataPoints>{pos}<commonCountingTime unit="seconds">10.0</commonCountingTime>{att_xml}'
            f'<{element} unit="counts">{" ".join(str(int(v)) for v in y)}</{element}></dataPoints></scan>')
    path.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<xrdMeasurements xmlns="http://www.xrdml.com/XRDMeasurement/{version}" '
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" status="Completed">'
        '<comment><entry>Goniometer=PW3050/60 (Theta/2Theta); Minimum step size 2Theta:0.001</entry>'
        '<entry>Sample stage=Spinner PW3064</entry></comment>'
        '<sample type="To be analyzed"><id>S-17</id><name></name></sample>'
        f'<xrdMeasurement measurementType="Scan" status="Completed" sampleMode="{mode}">'
        '<usedWavelength intended="K-Alpha 1"><kAlpha1 unit="Angstrom">1.5405980</kAlpha1>'
        '<kAlpha2 unit="Angstrom">1.5444260</kAlpha2><kBeta unit="Angstrom">1.3922500</kBeta>'
        f'<ratioKAlpha2KAlpha1>{ratio}</ratioKAlpha2KAlpha1></usedWavelength>'
        '<incidentBeamPath><radius unit="mm">240.00</radius>'
        '<xRayTube id="3" name="Cu LFF"><tension unit="kV">45</tension><current unit="mA">40</current>'
        f'<anodeMaterial>Cu</anodeMaterial></xRayTube>{mono_xml}{filt_xml}'
        '<sollerSlit id="4" name="Soller 0.04 rad."><opening unit="rad">0.04</opening></sollerSlit>'
        '</incidentBeamPath><sampleMovement xsi:type="spinningSampleMovementType">'
        '<spinnerRevolutionTime unit="seconds">4.0</spinnerRevolutionTime></sampleMovement>'
        '<diffractedBeamPath><radius unit="mm">240.00</radius>'
        '<detector id="5" name="X\'Celerator" xsi:type="rtmsDetectorType"><mode>Scanning</mode></detector>'
        '</diffractedBeamPath>'
        + "".join(scan.format(n=i) for i in range(scans)) +
        '</xrdMeasurement></xrdMeasurements>', encoding="utf-8")
    return path


def raw_v3(path, ranges, anode=b"Cu", lines=(1.54184, 1.5406, 1.54439, 1.39222, 0.5)):
    """RAW1.01: 712-byte header; per range a 304-byte header, a supplementary block, float32 counts."""
    h = bytearray(712)
    h[:7] = b"RAW1.01"
    struct.pack_into("<I", h, 12, len(ranges))
    h[16:24], h[26:34] = b"06/07/16", b"13:36:40"
    h[326:331] = b"KNbO3"
    struct.pack_into("<f", h, 564, 217.5)
    h[608:610] = anode
    struct.pack_into("<5d", h, 616, *lines)
    out = bytes(h)
    for start, step, y, t_k, sup in ranges:
        r = bytearray(304)
        struct.pack_into("<II", r, 0, 304, len(y))
        struct.pack_into("<d", r, 16, start)
        struct.pack_into("<d", r, 176, step)
        struct.pack_into("<f", r, 212, t_k)
        struct.pack_into("<II", r, 224, 40, 40)
        struct.pack_into("<I", r, 256, sup)
        out += bytes(r) + bytes(sup) + np.asarray(y, "<f4").tobytes()
    path.write_bytes(out)
    return path


def raw_v4(path, ranges, anode=b"Mo", lines=(0.71073, 0.7093, 0.71359, 0.63229, 0.5)):
    """RAW4.00: header, tagged segments (10 text, 30 source), per range a 160-byte
    header + its segments (temperature in the last 16 bytes) + the step records."""
    b = bytearray(61)
    b[:7] = b"RAW4.00"
    b[12:22], b[24:32] = b"03/05/2015", b"08:47:53"
    val = b"calcite-01\x00"
    b += struct.pack("<II", 10, 36 + len(val)) + b"\x00" * 4 + b"SampleID".ljust(24, b"\x00") + val
    s30 = bytearray(136)
    struct.pack_into("<II", s30, 0, 30, 136)
    struct.pack_into("<5d", s30, 72, *lines)
    s30[116:116 + len(anode)] = anode
    b += s30
    for start, step, y, datum, t_k in ranges:
        r = bytearray(160)
        struct.pack_into("<I", r, 0, 160)
        r[32:46] = b"Locked Coupled"
        struct.pack_into("<ddI", r, 72, start, step, len(y))
        struct.pack_into("<f", r, 92, 384.0)
        struct.pack_into("<ff", r, 100, 50.0, 50.0)
        struct.pack_into("<II", r, 136, datum, 32)
        seg = bytearray(32)
        struct.pack_into("<II", seg, 0, 110, 32)
        struct.pack_into("<f", seg, 16, t_k)
        if datum == 4:
            data = np.asarray(y, "<f4").tobytes()
        else:
            rec = np.zeros(len(y), dtype=[("y", "<f4"), ("k", "<i4")])
            rec["y"], rec["k"] = y, 1
            data = rec.tobytes()
        b += r + seg + data
    path.write_bytes(bytes(b))
    return path


def ras(path, *, wave_type="Ka", att=None, mono="None"):
    att = np.ones(len(X)) if att is None else att
    lines = ["*RAS_DATA_START", "*RAS_HEADER_START",
             '*FILE_SAMPLE "Si-std"', '*HW_GONIOMETER_NAME "SmartLab|SmartLab"',
             '*HW_SAMPLE_HOLDER_NAME "標準Zステージ|Standard Z stage"', '*HW_ATTACHMENT_NAME "Capillary|Capillary"',
             '*HW_XG_TARGET_NAME "Cu"', '*HW_XG_WAVE_LENGTH_ALPHA1 "1.540593"',
             '*HW_XG_WAVE_LENGTH_ALPHA2 "1.544414"', f'*MEAS_COND_XG_WAVE_TYPE "{wave_type}"',
             '*MEAS_COND_XG_VOLTAGE "40.0"', '*MEAS_COND_XG_CURRENT "44.0"',
             '*MEAS_COND_AXIS_NAME-14 "Incident monochromator"', f'*MEAS_COND_AXIS_POSITION-14 "{mono}"',
             '*MEAS_SCAN_AXIS_X "2-Theta"', '*MEAS_SCAN_MODE "CONTINUOUS"', '*MEAS_SCAN_STEP "1.0"',
             "*RAS_HEADER_END", "*RAS_INT_START"]
    lines += [f"{x:.4f} {y:.4f} {a:.4f}" for x, y, a in zip(X, Y, att)]
    lines += ["*RAS_INT_END", "*RAS_DATA_END"]
    path.write_bytes("\r\n".join(lines).encode("cp932"))
    return path


def rasx(path, wave_type="Ka1"):
    cond = ('<?xml version="1.0" encoding="utf-8"?><MeasurementConditions><GeneralInformation>'
            '<SystemName>SmartLabXE</SystemName></GeneralInformation><XrayGenerator>'
            '<TargetAtomicNumber>29</TargetAtomicNumber><TargetName>Cu</TargetName>'
            f'<WaveType>{wave_type}</WaveType><Current>200</Current><Voltage>45</Voltage>'
            '<WavelengthKalpha1>1.540593</WavelengthKalpha1><WavelengthKalpha2>1.544414</WavelengthKalpha2>'
            '</XrayGenerator><ScanInformation><AxisName>TwoTheta</AxisName><Mode>CONTINUOUS</Mode>'
            '<Start>10.0</Start><Stop>20.0</Stop><Step>1.0</Step><Speed>1.0</Speed><SpeedUnit>deg/min</SpeedUnit>'
            '<IntensityUnit>counts</IntensityUnit></ScanInformation></MeasurementConditions>')
    prof = "\n".join(f"{x:.4f}\t{y}\t1" for x, y in zip(X, Y))
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("root.xml", "<root/>")
        z.writestr("Data0/MesurementConditions0.xml", "﻿" + cond)   # Rigaku's spelling
        z.writestr("Data0/Profile0.txt", "﻿" + prof)
    return path


def brml(path, *, absorption=None, psd=False):
    absorption = np.ones(len(X)) if absorption is None else absorption
    if psd:
        rows = [f"9,1,{X[0]},5.0," + ",".join(str(int(v)) for v in Y)]
        count_view = (f'<RawDataView xsi:type="RecordedRawDataView" Start="4" Length="{len(Y)}">'
                      '<Recording LogicName="Counter1D" Category="Count"/></RawDataView>')
    else:
        rows = [f"1,{a},{x},{x / 2},{int(y)}" for x, y, a in zip(X, Y, absorption)]
        count_view = ('<RawDataView xsi:type="RecordedRawDataView" Start="4" Length="1">'
                      '<Recording LogicName="ScanCounter1" Category="Count"/></RawDataView>')
    xml = ('<?xml version="1.0"?><RawData xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
           '<TimeStampStarted>2023-01-26T15:23:09+01:00</TimeStampStarted><DataRoutes>'
           '<DataRoute RouteFlag="Measured"><ScanInformation ScanName="TwoThetaOmegaScan" VisibleName="2Theta-Omega">'
           '<MeasurementPoints>11</MeasurementPoints><TimePerStep>1</TimePerStep>'
           '<ScanModeVisibleName>Continuous</ScanModeVisibleName><ScanAxes>'
           '<ScanAxisInfo AxisId="TwoTheta"><Start>10</Start><Stop>20</Stop><Increment>1</Increment></ScanAxisInfo>'
           '</ScanAxes></ScanInformation>'
           + "".join(f"<Datum>{r}</Datum>" for r in rows) +
           '<DataViews><RawDataView xsi:type="FixedRawDataView" Start="0" Length="1" LogicName="MeasuredTime"/>'
           '<RawDataView xsi:type="FixedRawDataView" Start="1" Length="1" LogicName="AbsorptionFactor"/>'
           '<RawDataView xsi:type="VaryingRawDataView" Start="2" Length="2"><Varying LogicName="ScanAxes"/></RawDataView>'
           f'{count_view}</DataViews></DataRoute></DataRoutes>'
           '<FixedInformation><Instrument><WaveLengthAlpha1 Value="1.5406"/><WaveLengthAlpha2 Value="1.54439"/>'
           '<WaveLengthRatio Value="0.5"/><TubeMaterial>Cu</TubeMaterial><Voltage Value="40"/>'
           '<Current Value="40"/><Radius Value="300"/><GoniometerType>Theta_2Theta</GoniometerType>'
           '</Instrument></FixedInformation></RawData>')
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("experimentCollection.xml", "<c/>")
        z.writestr("Experiment0/RawData0.xml", xml)
    return path


# ---------------------------------------------------------------- formats --
def test_xrdml_doublet_with_filter(tmp_path):
    ds = read_dataset(xrdml(tmp_path / "a.xrdml"))
    assert ds.kind == "spectrum" and ds.format == "xrdml"
    np.testing.assert_allclose(ds.axes[0].coordinates(), X)
    np.testing.assert_allclose(ds.data, Y)
    m = ds.metadata
    assert m["wavelength_angstrom"] == {"K-alpha1": 1.540598, "K-alpha2": 1.544426, "K-alpha2/K-alpha1": 0.5}
    assert m["anode"] == "Cu" and m["sample_mode"] == "Reflection" and m["tube_kV"] == 45
    assert m["filters"] == ["source side: Beta-filter Nickel"]
    assert m["goniometer"].startswith("PW3050/60") and m["sample_id"] == "S-17"
    assert m["scan"]["points"] == 11 and m["scan"]["time_per_step_s"] == 10.0


def test_xrdml_incident_monochromator_means_kalpha1(tmp_path):
    # Panalytical keeps the tube's ratio in the header whatever the optics
    m = read_dataset(xrdml(tmp_path / "a.xrdml", version="1.3", element="intensities",
                           mono="Inc. Beam 1xGe111", filt=None)).metadata
    assert m["wavelength_angstrom"]["K-alpha2/K-alpha1"] == 0.0
    assert m["incident_monochromator"] == "Inc. Beam 1xGe111"
    assert any("passes K-alpha1 only" in n for n in m["notes"])


def test_xrdml_attenuation_applies_to_raw_counts_only(tmp_path):
    att = np.ones(len(Y)); att[4] = 188.0
    y = Y.copy(); y[4] = 16.0
    ds = read_dataset(xrdml(tmp_path / "c.xrdml", y=y, att=att))
    assert ds.data[4] == pytest.approx(16.0 * 188.0)
    ds = read_dataset(xrdml(tmp_path / "i.xrdml", y=y, att=att, element="intensities"))
    assert ds.data[4] == pytest.approx(16.0)      # <intensities> are already corrected


def test_xrdml_list_positions_and_several_scans(tmp_path):
    xs = np.array([10, 11, 12.5, 14, 15, 16, 17, 18, 19, 19.5, 20])
    found = read_datasets(xrdml(tmp_path / "a.xrdml", positions=xs, scans=3))
    assert len(found) == 3
    np.testing.assert_allclose(found[0].axes[0].coordinates(), xs)
    assert found[0].metadata["scans_in_file"] == 3 and "series" in found[1].metadata["notes"][-1]


def test_bruker_raw_v3_ranges_and_temperatures(tmp_path):
    p = raw_v3(tmp_path / "s.raw", [(10.0, 1.0, Y, 303.15, 40), (10.0, 1.0, 2 * Y, 573.15, 0)])
    found = read_datasets(p)
    assert [d.metadata["temperature_C"] for d in found] == [30.0, 300.0]
    np.testing.assert_allclose(found[1].data, 2 * Y)
    np.testing.assert_allclose(found[0].axes[0].coordinates(), X)
    m = found[0].metadata
    assert m["wavelength_angstrom"] == {"K-alpha1": 1.5406, "K-alpha2": 1.54439, "K-alpha2/K-alpha1": 0.5}
    assert (m["anode"], m["tube_kV"], m["goniometer_radius_mm"], m["sample_id"]) == ("Cu", 40.0, 217.5, "KNbO3")


@pytest.mark.parametrize("datum", [4, 8])
def test_bruker_raw_v4(tmp_path, datum):
    ds = read_dataset(raw_v4(tmp_path / "m.raw", [(10.0, 1.0, Y, datum, 298.15)]))
    np.testing.assert_allclose(ds.data, Y)
    np.testing.assert_allclose(ds.axes[0].coordinates(), X)
    m = ds.metadata
    assert m["anode"] == "Mo" and m["wavelength_angstrom"]["K-alpha1"] == pytest.approx(0.7093)
    assert m["sample_id"] == "calcite-01" and m["temperature_C"] == pytest.approx(25.0)
    assert m["scan"]["time_per_step_s"] == pytest.approx(0.384) and m["tube_kV"] == 50.0


def test_bruker_raw_old_versions_refused_by_name(tmp_path):
    p = tmp_path / "old.raw"
    p.write_bytes(b"RAW2" + bytes(300))
    with pytest.raises(UnsupportedContent, match="version 2"):
        read_dataset(p)


def test_other_raw_and_ras_files_are_not_claimed(tmp_path):
    img = tmp_path / "frame.raw"
    img.write_bytes(np.arange(64, dtype="<u2").tobytes())
    sun = tmp_path / "pic.ras"
    sun.write_bytes(b"\x59\xa6\x6a\x95" + bytes(60))
    for p in (img, sun):
        assert not records_calibration(p) and not can_read(p)


def test_ras_quoted_values_japanese_pairs_and_attenuator(tmp_path):
    att = np.ones(len(X)); att[4] = 2.0
    ds = read_dataset(ras(tmp_path / "a.ras", att=att))
    np.testing.assert_allclose(ds.axes[0].coordinates(), X)
    assert ds.data[4] == pytest.approx(Y[4] * 2.0)
    m = ds.metadata
    assert m["wavelength_angstrom"] == {"K-alpha1": 1.540593, "K-alpha2": 1.544414, "K-alpha2/K-alpha1": 0.5}
    assert m["instrument"] == "Rigaku SmartLab" and m["sample_stage"] == "Capillary, Standard Z stage"
    assert m["scan"]["step"] == 1.0 and m["tube_mA"] == 44.0


def test_ras_named_incident_monochromator(tmp_path):
    m = read_dataset(ras(tmp_path / "a.ras", mono="Ge(220)x2")).metadata
    assert m["incident_monochromator"] == "Ge(220)x2" and m["wavelength_angstrom"]["K-alpha2/K-alpha1"] == 0.0


def test_rasx_wave_type_kalpha1(tmp_path):
    ds = read_dataset(rasx(tmp_path / "a.rasx"))
    np.testing.assert_allclose(ds.data, Y)
    m = ds.metadata
    assert m["wavelength_angstrom"]["K-alpha2/K-alpha1"] == 0.0 and m["instrument"] == "Rigaku SmartLabXE"
    assert m["scan"]["axis"] == "TwoTheta" and m["tube_kV"] == 45


def test_brml_columns_from_data_views_and_absorption(tmp_path):
    absorption = np.ones(len(X)); absorption[4] = 8.3
    ds = read_dataset(brml(tmp_path / "a.brml", absorption=absorption))
    np.testing.assert_allclose(ds.axes[0].coordinates(), X)
    assert ds.data[4] == pytest.approx(Y[4] * 8.3) and ds.data[0] == Y[0]
    m = ds.metadata
    assert m["wavelength_angstrom"]["K-alpha2/K-alpha1"] == 0.5 and m["goniometer"] == "Theta_2Theta"
    assert m["goniometer_radius_mm"] == 300.0


def test_brml_detector_snapshot_refused(tmp_path):
    with pytest.raises(UnsupportedContent, match="stationary-detector"):
        read_dataset(brml(tmp_path / "psd.brml", psd=True))


def test_curve_loader_reads_instrument_files(tmp_path):
    from scilink.skills._shared.curve_fitting_tools import load_curve_data
    arr = load_curve_data(str(xrdml(tmp_path / "a.xrdml")))
    np.testing.assert_allclose(arr, np.column_stack([X, Y]))


# --------------------------------------------- what the XRD tools make of it --
@pytest.mark.parametrize("make, wavelength, geometry", [
    (lambda p: xrdml(p / "a.xrdml"), "CuKa", "bragg-brentano"),
    (lambda p: xrdml(p / "b.xrdml", mono="Inc. Beam 1xGe111", filt=None), "CuKa1", "bragg-brentano"),
    (lambda p: xrdml(p / "c.xrdml", mode="Transmission"), "CuKa", "debye-scherrer"),
    (lambda p: raw_v4(p / "d.raw", [(10.0, 1.0, Y, 4, 298.15)]), "MoKa", None),
    (lambda p: rasx(p / "e.rasx"), "CuKa1", None),
    (lambda p: ras(p / "f.ras"), "CuKa", "debye-scherrer"),      # 'Capillary' attachment
    (lambda p: brml(p / "g.brml"), "CuKa", None),
])
def test_record_resolves(tmp_path, make, wavelength, geometry):
    md = file_metadata(read_dataset(make(tmp_path), header_only=True))
    assert md["experiment_type"] == "Diffraction"
    assert resolve_wavelength(md) == wavelength
    assert resolve_geometry(md) == geometry


def test_recorded_sample_mode_beats_stage_names(tmp_path):
    md = file_metadata(read_dataset(xrdml(tmp_path / "a.xrdml"), header_only=True))
    md["experiment"]["sample_stage"] = "Reflection-transmission spinner"
    assert resolve_geometry(md) == "bragg-brentano"


def test_negated_monochromator_is_the_doublet():
    assert resolve_wavelength({"radiation": "Cu K-alpha", "optics": "no incident-beam monochromator"}) == "CuKa"
    assert resolve_wavelength({"radiation": "Cu K-alpha", "optics": "without a primary monochromator"}) == "CuKa"
    assert resolve_wavelength({"radiation": "Cu K-alpha", "optics": "Ge(111) incident-beam monochromator"}) == "CuKa1"


# ------------------------------------------------------------- precedence --
def test_person_values_stand_and_disagreement_is_reported(tmp_path):
    fmd = file_metadata(read_dataset(xrdml(tmp_path / "a.xrdml"), header_only=True))
    given = {"wavelength_angstrom": 0.49591, "instrument": "synchrotron, Debye-Scherrer capillary"}
    merged, filled, conflicts = merge_file_metadata(given, fmd)
    assert resolve_wavelength(merged) == pytest.approx(0.49591)
    assert resolve_geometry(merged) == "debye-scherrer"
    assert any(c.startswith("wavelength") for c in conflicts)
    assert any(c.startswith("geometry") for c in conflicts)
    assert "experiment" in filled and "wavelength" not in merged["experiment"]


def test_agreeing_person_values_raise_no_conflict(tmp_path):
    fmd = file_metadata(read_dataset(xrdml(tmp_path / "a.xrdml"), header_only=True))
    given = {"radiation": "Cu K-alpha (K-alpha2 present)", "instrument": "Bragg-Brentano reflection"}
    merged, _, conflicts = merge_file_metadata(given, fmd)
    assert conflicts == [] and resolve_wavelength(merged) == "CuKa"


def test_nothing_given_the_file_fills_it(tmp_path):
    fmd = file_metadata(read_dataset(raw_v4(tmp_path / "m.raw", [(10.0, 1.0, Y, 4, 298.15)]), header_only=True))
    merged, filled, conflicts = merge_file_metadata({"sample": "calcite"}, fmd)
    assert resolve_wavelength(merged) == "MoKa" and not conflicts
    assert "experiment" in filled and merged["sample"] == "calcite"


# ------------------------------------------------------------- curve agent --
def _agent():
    from scilink.agents.exp_agents.curve_fitting_agent import CurveFittingAgent
    fake = types.SimpleNamespace(logger=logging.getLogger("t"),
                                 _PER_FILE_KEYS=CurveFittingAgent._PER_FILE_KEYS)
    return lambda info, paths: CurveFittingAgent._merge_instrument_record(fake, info, paths)


def test_agent_fills_system_info_from_the_file(tmp_path):
    merge = _agent()
    info = merge({}, [str(xrdml(tmp_path / "a.xrdml", mono="Inc. Beam 1xGe111", filt=None))])
    assert resolve_wavelength(info) == "CuKa1" and info["metadata_source"].startswith("Read from the xrdml")
    assert merge(info, [str(tmp_path / "a.xrdml")]) is info          # already merged: unchanged
    csv = tmp_path / "p.csv"
    csv.write_text("x,y\n1,2\n2,3\n")
    assert merge({"a": 1}, [str(csv)]) == {"a": 1}                   # not an instrument file


def test_agent_series_leaves_out_per_file_values(tmp_path):
    merge = _agent()
    p = raw_v3(tmp_path / "s.raw", [(10.0, 1.0, Y, 303.15, 0)])
    one = merge({"sample": "KNbO3"}, [str(p)])
    many = merge({"sample": "KNbO3"}, [str(p), str(p)])
    assert one["acquisition"]["temperature_C"] == 30.0
    assert "temperature_C" not in many["acquisition"] and resolve_wavelength(many) == "CuKa"
