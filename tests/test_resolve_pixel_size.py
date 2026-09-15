"""Offline tests for resolve_pixel_size_nm — the FOV/shape calibration helper.

Regression for the benchmark bug where generated code divided field_of_view by a
non-existent metadata count field (n_cols) and left pixel_size None, instead of
dividing by the image width.
"""
from scilink.skills._shared.image_analysis_tools import resolve_pixel_size_nm


def test_fov_over_shape_sm13():
    """sm_13: 10 nm FOV over a 1540-px-wide image -> ~0.006494 nm/px (matches the
    pixel size embedded in the TIFF tags). The agent's script returned None."""
    md = {"experimental_details": {"spatial_info": {
        "field_of_view_x": 10, "field_of_view_y": 10, "field_of_view_units": "nm"}}}
    r = resolve_pixel_size_nm(md, (1540, 1540))
    assert r is not None and r["source"] == "field_of_view"
    assert abs(r["x"] - 10.0 / 1540) < 1e-9
    assert abs(r["y"] - 10.0 / 1540) < 1e-9


def test_units_micron_conversion():
    md = {"experimental_details": {"spatial_info": {
        "field_of_view_x": 2, "field_of_view_y": 2, "field_of_view_units": "um"}}}
    r = resolve_pixel_size_nm(md, (256, 256))  # AFM (easy) Au case, 2 um / 256
    assert abs(r["x"] - 2000.0 / 256) < 1e-6


def test_non_square_uses_correct_axis():
    md = {"spatial_info": {"field_of_view_x": 800, "field_of_view_y": 400,
                           "field_of_view_units": "nm"}}
    r = resolve_pixel_size_nm(md, (400, 800))  # rows=400, cols=800
    assert abs(r["x"] - 800.0 / 800) < 1e-9   # x uses cols
    assert abs(r["y"] - 400.0 / 400) < 1e-9   # y uses rows


def test_system_info_wrapper():
    md = {"system_info": {"experimental_details": {"spatial_info": {
        "field_of_view_x": 1046.34, "field_of_view_units": "nm"}}}}
    r = resolve_pixel_size_nm(md, (1024, 1024))
    assert abs(r["x"] - 1046.34 / 1024) < 1e-6


def test_embedded_tag_fallback():
    md = {"embedded_file_metadata": {"pixel_size": {"x": 0.05, "y": 0.05, "unit": "nm"}}}
    r = resolve_pixel_size_nm(md, (512, 512))
    assert r["source"] == "embedded_tags" and abs(r["x"] - 0.05) < 1e-9


def test_returns_none_when_unresolvable():
    assert resolve_pixel_size_nm({}, (512, 512)) is None
    assert resolve_pixel_size_nm(None, (512, 512)) is None
    assert resolve_pixel_size_nm({"spatial_info": {}}, (512, 512)) is None
    # unitless embedded pixel size (sm_13's embedded tag had unit "none") -> skip
    md = {"embedded_file_metadata": {"pixel_size": {"x": 0.0065, "unit": "none"}}}
    assert resolve_pixel_size_nm(md, (1540, 1540)) is None


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        try:
            fn(); print(f"PASS {fn.__name__}"); passed += 1
        except Exception:
            print(f"FAIL {fn.__name__}"); traceback.print_exc()
    print(f"\n{passed}/{len(fns)} passed")
