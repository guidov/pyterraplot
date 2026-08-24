"""Tests for the Dataset accessor — ds.tp.quiver_html / compare_html."""
import re

import numpy as np
import pytest
import xarray as xr

import pyterraplot  # noqa: F401 — registers the .tp accessors


def shared_scale(html: str) -> tuple[float, float]:
    """(vmin, vmax) from compare_html's shared options line.

    Matched precisely rather than by splitting on "vmin: ", which also occurs
    inside the inlined terraplot bundle.
    """
    m = re.search(r"const opts = \{[^}]*?vmin: (-?[\d.eE+-]+), vmax: (-?[\d.eE+-]+)",
                  html)
    assert m, "shared colour-scale options not found in the emitted page"
    return float(m.group(1)), float(m.group(2))


def make_ds():
    lats = np.linspace(90, -90, 25)
    lons = np.linspace(-180, 175, 48)
    LON, LAT = np.meshgrid(lons, lats)
    u = 20 * np.cos(np.deg2rad(LAT))
    v = 8 * np.sin(np.deg2rad(2 * LON))
    ds = xr.Dataset(
        {"u10": (("lat", "lon"), u),
         "v10": (("lat", "lon"), v),
         "wspd": (("lat", "lon"), np.hypot(u, v)),
         "wspd_b": (("lat", "lon"), np.hypot(u, v) * 1.1)},
        coords={"lat": lats, "lon": lons},
    )
    ds["wspd"].attrs = {"units": "m s-1", "long_name": "10 m wind speed"}
    ds["wspd_b"].attrs = {"units": "m s-1", "long_name": "10 m wind speed (model)"}
    return ds


class TestQuiverHtml:
    def test_basic_structure(self, tmp_path):
        html = make_ds().tp.quiver_html(tmp_path / "q.html",
                                        u="u10", v="v10").read_text()
        assert "new GeoMap('#map'" in html
        assert "map.quiver(uData.lons, uData.lats, uData.field, vData.field" in html
        assert html.count("await unpackField(") == 2      # u and v

    def test_background_field_is_drawn(self, tmp_path):
        html = make_ds().tp.quiver_html(tmp_path / "q.html", u="u10", v="v10",
                                        background="wspd", cmap="magma").read_text()
        assert "map.pcolormesh(bg.lons, bg.lats, bg.field" in html
        assert "cmap: 'magma'" in html
        assert html.count("await unpackField(") == 3      # u, v and background

    def test_background_gets_a_colorbar(self, tmp_path):
        """Regression: the colorbar block used to be built and never inserted,
        so a background field rendered with no colour scale at all."""
        html = make_ds().tp.quiver_html(tmp_path / "q.html", u="u10", v="v10",
                                        background="wspd").read_text()
        assert '<div id="colorbar"></div>' in html
        assert "new Colorbar(host, cbarOpts)" in html
        assert "})(bg.field," in html          # reads the background, not `payload`

    def test_no_dangling_placeholder_or_empty_import(self, tmp_path):
        """The old block imported Colorbar from an empty data: URL, which would
        have thrown in the browser. Colorbar comes from the inlined bundle."""
        html = make_ds().tp.quiver_html(tmp_path / "q.html", u="u10", v="v10",
                                        background="wspd").read_text()
        assert "data:text/javascript;base64,';" not in html
        assert "CBAR_OPTS" not in html

    def test_colorbar_label_from_attrs(self, tmp_path):
        html = make_ds().tp.quiver_html(tmp_path / "q.html", u="u10", v="v10",
                                        background="wspd").read_text()
        assert "10 m wind speed [m s-1]" in html

    def test_cbar_opts_are_honoured(self, tmp_path):
        html = make_ds().tp.quiver_html(
            tmp_path / "q.html", u="u10", v="v10", background="wspd",
            cbar_opts={"orientation": "vertical", "position": "right",
                       "ticks": [0, 10, 20]},
        ).read_text()
        assert '"orientation": "vertical"' in html
        assert '"position": "right"' in html
        assert '"ticks": [0, 10, 20]' in html

    def test_no_background_means_no_colorbar(self, tmp_path):
        html = make_ds().tp.quiver_html(tmp_path / "q.html",
                                        u="u10", v="v10").read_text()
        assert '<div id="colorbar">' not in html
        assert "new Colorbar(" not in html

    def test_vmin_vmax_reach_the_colorbar(self, tmp_path):
        html = make_ds().tp.quiver_html(tmp_path / "q.html", u="u10", v="v10",
                                        background="wspd", vmin=0, vmax=25).read_text()
        assert '"vmin": 0' in html and '"vmax": 25' in html

    def test_label_without_background(self, tmp_path):
        html = make_ds().tp.quiver_html(tmp_path / "q.html",
                                        u="u10", v="v10").read_text()
        assert '<div id="label">u10 / v10</div>' in html

    def test_projection_extent_and_coastlines(self, tmp_path):
        html = make_ds().tp.quiver_html(
            tmp_path / "q.html", u="u10", v="v10", projection="orthographic",
            center=(-95, 60), extent=(-141, -52, 41, 84), coastlines=True,
        ).read_text()
        assert "projection: 'orthographic'" in html
        assert "center: [-95, 60]" in html
        assert "extent: [-141, -52, 41, 84]" in html
        assert "map.addFeature('coastlines')" in html

    def test_coastlines_off(self, tmp_path):
        html = make_ds().tp.quiver_html(tmp_path / "q.html", u="u10", v="v10",
                                        coastlines=False).read_text()
        assert "addFeature('coastlines')" not in html

    def test_quiver_styling_options(self, tmp_path):
        html = make_ds().tp.quiver_html(
            tmp_path / "q.html", u="u10", v="v10",
            quiver_cmap="plasma", quiver_density=3, quiver_scale=30,
        ).read_text()
        assert "density: 3" in html
        assert "scale: 30" in html
        assert "cmap: 'plasma'" in html

    def test_missing_variable_raises(self, tmp_path):
        with pytest.raises(KeyError):
            make_ds().tp.quiver_html(tmp_path / "q.html", u="nope", v="v10")


class TestCompareHtml:
    def test_two_panels_share_a_scale(self, tmp_path):
        html = make_ds().tp.compare_html(tmp_path / "c.html",
                                         a="wspd", b="wspd_b").read_text()
        assert html.count("new GeoMap(") == 1        # one ctor, looped over both
        assert "'#map-a'" in html and "'#map-b'" in html
        assert "new Colorbar('#shared-cbar'" in html

    def test_symmetric_scale_is_centred_on_zero(self, tmp_path):
        html = make_ds().tp.compare_html(tmp_path / "c.html", a="wspd",
                                         b="wspd_b", symmetric=True).read_text()
        vmin, vmax = shared_scale(html)
        assert vmin == pytest.approx(-vmax)

    def test_explicit_range_respected(self, tmp_path):
        html = make_ds().tp.compare_html(tmp_path / "c.html", a="wspd",
                                         b="wspd_b", vmin=-5, vmax=5).read_text()
        assert shared_scale(html) == (-5.0, 5.0)
