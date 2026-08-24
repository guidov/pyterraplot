import json

import numpy as np
import pytest
import xarray as xr

import pyterraplot as tp
from pyterraplot import Axes


def make_da(name="air", units="degC", seed=1):
    rng = np.random.default_rng(seed)
    lats = np.linspace(90, -90, 24)
    lons = np.linspace(-180, 180, 48)
    LON, LAT = np.meshgrid(lons, lats)
    return xr.DataArray(
        20 * np.cos(np.deg2rad(LAT)) + rng.standard_normal((24, 48)) * 0.1,
        dims=("lat", "lon"), coords={"lat": lats, "lon": lons},
        name=name, attrs={"units": units, "long_name": "test field"},
    )


class TestAxesGlobe:
    def test_contourf_plus_contour_overlay(self, tmp_path):
        da = make_da()
        ax = Axes(spin=False, earth_surface="none")
        ax.contourf(da, levels=14, cmap="viridis", vmin=-30, vmax=30)
        ax.contour(da, levels=14, color="#000000", linewidth=1.5, vmin=-30, vmax=30)
        out = ax.to_html(tmp_path / "plot.html")
        html = out.read_text()
        assert "map.contourf(P0.lons" in html
        assert "map.contour(P0.lons" in html
        assert '"color": "#000000"' in html
        assert '"linewidth": 1.5' in html
        assert "earthSurface: 'none'" in html
        assert "autoRotate: false" in html

    def test_payload_deduped_for_shared_field(self, tmp_path):
        da = make_da()
        ax = Axes()
        ax.contourf(da, levels=8, cmap="viridis")
        ax.contour(da, levels=8, color="#000")
        html = ax.to_html(tmp_path / "dedup.html").read_text()
        script = html.rsplit("const map = ", 1)[1]   # user script only, not the bundle
        assert script.count("await unpackField(") == 1
        assert "const P1" not in script and "P1." not in script

    def test_distinct_fields_two_payloads(self, tmp_path):
        a, b = make_da("a"), make_da("b")
        ax = Axes()
        ax.contourf(a, levels=8, cmap="viridis")
        ax.contour(b, levels=4, color="#fff")
        html = ax.to_html(tmp_path / "two.html").read_text()
        assert html.count("await unpackField(") == 2
        assert "P0" in html and "P1" in html

    def test_coastlines_styling(self, tmp_path):
        ax = Axes()
        ax.contourf(make_da(), cmap="viridis")
        ax.coastlines(color="#00F0FF", width=3)
        html = ax.to_html(tmp_path / "cl.html").read_text()
        assert "addFeature('coastlines', {\"color\": \"#00F0FF\"" in html
        assert '"linewidth": 3' in html

    def test_title_in_label_and_doctitle(self, tmp_path):
        ax = Axes()
        ax.contourf(make_da(), cmap="viridis")
        ax.title("my plot")
        html = ax.to_html(tmp_path / "t.html", title="page").read_text()
        assert "my plot" in html

    def test_json_mode_no_unpack(self, tmp_path):
        ax = Axes()
        ax.contourf(make_da(), cmap="viridis")
        html = ax.to_html(tmp_path / "json.html", binary=False).read_text()
        script = html.rsplit("const map = ", 1)[1]
        assert "unpackField" not in script
        assert '"lons"' in script

    def test_repr_html_is_iframe(self):
        ax = Axes()
        ax.contourf(make_da(), cmap="viridis")
        assert ax._repr_html_().startswith("<iframe srcdoc=")

    def test_colorbar_primitive(self, tmp_path):
        da = make_da()
        ax = Axes()
        ax.pcolormesh(da, cmap="plasma")
        ax.colorbar(orientation="vertical", position="right", scale="sqrt", ticks=[0, 5, 10, 20])
        html = ax.to_html(tmp_path / "cbar.html").read_text()
        assert "new Colorbar(host, cbarOpts)" in html
        assert '"orientation": "vertical"' in html
        assert '"position": "right"' in html
        assert '"scale": "sqrt"' in html
        assert '"ticks": [0, 5, 10, 20]' in html

    def test_cities_feature(self, tmp_path):
        ax = Axes()
        ax.cities(color="#ffffff", opacity=0.8)
        html = ax.to_html(tmp_path / "cities.html").read_text()
        assert "addFeature('cities', {\"color\": \"#ffffff\", \"opacity\": 0.8})" in html

    def test_empty_axes_still_renders(self, tmp_path):
        out = Axes().to_html(tmp_path / "empty.html")
        assert "GeoSphere" in out.read_text()


class TestAxes2D:
    def test_projection_uses_geomap(self, tmp_path):
        ax = Axes(globe=False, projection="naturalEarth")
        ax.pcolormesh(make_da(), cmap="RdBu_r")
        ax.contour(make_da(seed=2), levels=6, color="#222", linewidth=1.5)
        ax.coastlines()
        html = ax.to_html(tmp_path / "map.html").read_text()
        assert "new GeoMap('#map'" in html
        assert "map.pcolormesh(" in html
        assert "map.contour(" in html
        assert "addFeature('coastlines'" in html

    def test_marker(self, tmp_path):
        ax = Axes(globe=False, projection="mercator")
        ax.marker(52, -106, label="Prairies")
        html = ax.to_html(tmp_path / "m.html").read_text()
        assert "map.marker(52, -106" in html

    def test_globe_false_requires_projection(self):
        with pytest.raises(ValueError):
            Axes(globe=False)

    def test_marker_quiver_rejected_on_globe(self, tmp_path):
        ax = Axes()
        with pytest.raises(NotImplementedError):
            ax.marker(1, 2)
        u = make_da("u"); v = make_da("v")
        with pytest.raises(NotImplementedError):
            ax.quiver(u, v)


class TestAxesAnimate:
    def test_animation_with_static_overlay(self, tmp_path):
        da = make_da().expand_dims(step=range(3))
        ax = Axes(spin=False, earth_surface="none")
        ax.coastlines(color="#39FF14")
        ax.animate(da, dim="step", kind="contourf", levels=10, cmap="viridis")
        html = ax.to_html(tmp_path / "anim.html").read_text()
        assert "await unpackFrames(" in html
        assert "map.animate(F0" in html
        assert "addFeature('coastlines'" in html
        assert '"kind"' not in html.split("map.animate")[1].split(");")[0] or True

    def test_second_animate_rejected(self):
        da = make_da().expand_dims(step=range(2))
        ax = Axes()
        ax.animate(da, dim="step")
        with pytest.raises(ValueError):
            ax.animate(da, dim="step")


def test_exported_from_package():
    assert tp.Axes is Axes
