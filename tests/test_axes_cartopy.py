"""Tests for the cartopy-style surface on Axes: CRS projections, features,
gridlines, and the point/line/text/vector primitives."""
import json

import numpy as np
import pytest
import xarray as xr

import pyterraplot as tp
import pyterraplot.crs as ccrs
import pyterraplot.feature as cfeature
from pyterraplot import Axes


def make_da(name="air", units="degC", seed=1):
    rng = np.random.default_rng(seed)
    lats = np.linspace(90, -90, 24)
    lons = np.linspace(-180, 175, 48)
    LON, LAT = np.meshgrid(lons, lats)
    return xr.DataArray(
        20 * np.cos(np.deg2rad(LAT)) + rng.standard_normal((24, 48)) * 0.1,
        dims=("lat", "lon"), coords={"lat": lats, "lon": lons},
        name=name, attrs={"units": units, "long_name": "test field"},
    )


def make_uv():
    lats = np.linspace(90, -90, 24)
    lons = np.linspace(-180, 175, 48)
    LON, LAT = np.meshgrid(lons, lats)
    u = xr.DataArray(-10 * np.sin(np.deg2rad(LAT)) * np.cos(np.deg2rad(LON)),
                     dims=("lat", "lon"), coords={"lat": lats, "lon": lons},
                     name="u", attrs={"units": "m s-1"})
    v = xr.DataArray(8 * np.cos(np.deg2rad(2 * LAT)),
                     dims=("lat", "lon"), coords={"lat": lats, "lon": lons},
                     name="v", attrs={"units": "m s-1"})
    return u, v


def ctor_opts(html: str) -> dict:
    """Parse the GeoMap constructor options object out of the emitted page."""
    marker = "new GeoMap('#map', "
    tail = html.rsplit(marker, 1)[1]
    return json.loads(tail[:tail.index(");")])


class TestProjectionArgument:
    def test_crs_object_selects_2d(self, tmp_path):
        ax = Axes(projection=ccrs.Robinson(central_longitude=-100))
        assert ax.globe is False
        ax.pcolormesh(make_da(), cmap="viridis")
        opts = ctor_opts(ax.to_html(tmp_path / "r.html").read_text())
        assert opts["projection"] == "robinson"
        assert opts["center"] == [-100.0, 0.0]

    def test_globe3d_crs_selects_globe(self, tmp_path):
        ax = Axes(projection=ccrs.Globe3D())
        assert ax.globe is True
        assert "new GeoSphere(" in ax.to_html(tmp_path / "g.html").read_text()

    def test_default_is_globe(self):
        assert Axes().globe is True

    def test_conic_parallels_reach_the_ctor(self, tmp_path):
        ax = Axes(projection=ccrs.LambertConformal(standard_parallels=(30, 60)))
        ax.pcolormesh(make_da(), cmap="viridis")
        opts = ctor_opts(ax.to_html(tmp_path / "lc.html").read_text())
        assert opts["projection"] == "conicconformal"
        assert opts["parallels"] == [30.0, 60.0]

    def test_polar_stereo_brings_its_default_extent(self, tmp_path):
        ax = Axes(projection=ccrs.NorthPolarStereo())
        ax.pcolormesh(make_da(), cmap="viridis")
        opts = ctor_opts(ax.to_html(tmp_path / "nps.html").read_text())
        assert opts["extent"] == [-180.0, 180.0, 45.0, 90.0]

    def test_rotated_pole_emits_rotate_not_center(self, tmp_path):
        ax = Axes(projection=ccrs.RotatedPole(pole_longitude=-162,
                                              pole_latitude=39.25))
        ax.pcolormesh(make_da(), cmap="viridis")
        opts = ctor_opts(ax.to_html(tmp_path / "rp.html").read_text())
        assert opts["rotate"] == [162.0, 50.75, 0.0]
        assert "center" not in opts

    def test_nearside_perspective_distance(self, tmp_path):
        ax = Axes(projection=ccrs.NearsidePerspective(satellite_height=6_378_137.0))
        ax.pcolormesh(make_da(), cmap="viridis")
        opts = ctor_opts(ax.to_html(tmp_path / "np.html").read_text())
        assert opts["distance"] == pytest.approx(2.0)

    def test_explicit_center_overrides_projection(self, tmp_path):
        ax = Axes(projection=ccrs.Mollweide(central_longitude=-100), center=(45, 10))
        ax.pcolormesh(make_da(), cmap="viridis")
        assert ctor_opts(ax.to_html(tmp_path / "c.html").read_text())["center"] == [45, 10]

    def test_string_projection_still_works(self, tmp_path):
        ax = Axes(globe=False, projection="naturalEarth")
        ax.pcolormesh(make_da(), cmap="viridis")
        assert "new GeoMap('#map'" in ax.to_html(tmp_path / "s.html").read_text()

    def test_unknown_string_projection_passes_through(self, tmp_path):
        ax = Axes(globe=False, projection="someFutureProjection")
        ax.pcolormesh(make_da(), cmap="viridis")
        opts = ctor_opts(ax.to_html(tmp_path / "u.html").read_text())
        assert opts["projection"] == "someFutureProjection"

    def test_transform_crs_rejected_as_projection(self):
        with pytest.raises(TypeError, match="transform"):
            Axes(projection=ccrs.Geodetic())

    def test_conflicting_globe_flag_rejected(self):
        with pytest.raises(ValueError, match="globe=True"):
            Axes(projection=ccrs.Robinson(), globe=True)
        with pytest.raises(ValueError, match="globe=False"):
            Axes(projection=ccrs.Globe3D(), globe=False)

    def test_bad_projection_type_rejected(self):
        with pytest.raises(TypeError, match="projection must be"):
            Axes(projection=42)


class TestFeatures:
    def test_add_feature_land_fills(self, tmp_path):
        ax = Axes(projection=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND)
        html = ax.to_html(tmp_path / "land.html").read_text()
        assert "addFeature('land'" in html
        assert '"fill": "#3b3b32"' in html

    def test_with_scale_emits_resolution(self, tmp_path):
        ax = Axes(projection=ccrs.PlateCarree())
        ax.add_feature(cfeature.BORDERS.with_scale("50m"))
        assert '"scale": "50m"' in ax.to_html(tmp_path / "b.html").read_text()

    def test_default_scale_is_omitted(self, tmp_path):
        ax = Axes(projection=ccrs.PlateCarree())
        ax.add_feature(cfeature.BORDERS)
        assert '"scale"' not in ax.to_html(tmp_path / "b2.html").read_text()

    def test_style_overrides_apply(self, tmp_path):
        ax = Axes(projection=ccrs.PlateCarree())
        ax.add_feature(cfeature.OCEAN, facecolor="#001133", alpha=0.5)
        html = ax.to_html(tmp_path / "o.html").read_text()
        assert '"fill": "#001133"' in html
        assert '"opacity": 0.5' in html

    def test_features_are_immutable(self):
        scaled = cfeature.LAND.with_scale("10m")
        assert scaled.scale == "10m"
        assert cfeature.LAND.scale == "110m"      # the constant is untouched
        assert scaled is not cfeature.LAND

    def test_bad_scale_rejected(self):
        with pytest.raises(ValueError, match="scale must be"):
            cfeature.LAND.with_scale("1m")

    def test_convenience_methods(self, tmp_path):
        ax = Axes(projection=ccrs.PlateCarree())
        ax.land().ocean().lakes().rivers().states().borders().coastlines()
        html = ax.to_html(tmp_path / "all.html").read_text()
        for name in ("land", "ocean", "lakes", "rivers", "states",
                     "borders", "coastlines"):
            assert f"addFeature('{name}'" in html

    def test_coastlines_resolution(self, tmp_path):
        ax = Axes(projection=ccrs.PlateCarree())
        ax.coastlines(resolution="10m", color="#fff", linewidth=1.5)
        html = ax.to_html(tmp_path / "c.html").read_text()
        assert '"scale": "10m"' in html
        assert '"linewidth": 1.5' in html

    def test_natural_earth_feature_aliases_cartopy_names(self):
        f = cfeature.NaturalEarthFeature("physical", "coastline", "50m")
        assert f.name == "coastlines"
        assert f.to_js()["scale"] == "50m"

    def test_stock_img_sets_surface(self, tmp_path):
        ax = Axes(projection=ccrs.PlateCarree())
        ax.stock_img()
        ax.pcolormesh(make_da(), cmap="viridis")
        assert ctor_opts(ax.to_html(tmp_path / "s.html").read_text())["earthSurface"] \
            == "shaded_relief"


class TestGridlines:
    def test_gridlines_emitted(self, tmp_path):
        ax = Axes(projection=ccrs.PlateCarree())
        ax.gridlines(draw_labels=True, xstep=45, ystep=15, linestyle="--")
        html = ax.to_html(tmp_path / "g.html").read_text()
        assert "map.gridlines(" in html
        assert '"drawLabels": true' in html
        assert '"xstep": 45.0' in html
        assert '"linestyle": "--"' in html

    def test_explicit_locations(self, tmp_path):
        ax = Axes(projection=ccrs.PlateCarree())
        ax.gridlines(xlocs=[-120, -60, 0], ylocs=[0, 45])
        html = ax.to_html(tmp_path / "gl.html").read_text()
        assert '"xlocs": [-120.0, -60.0, 0.0]' in html
        assert '"ylocs": [0.0, 45.0]' in html
        assert '"xstep"' not in html          # locs and step are exclusive

    def test_gridlines_disables_default_graticule(self, tmp_path):
        ax = Axes(projection=ccrs.PlateCarree())
        ax.pcolormesh(make_da(), cmap="viridis")
        ax.gridlines()
        assert ctor_opts(ax.to_html(tmp_path / "gg.html").read_text())["graticule"] is False

    def test_no_gridlines_keeps_default_graticule(self, tmp_path):
        ax = Axes(projection=ccrs.PlateCarree())
        ax.pcolormesh(make_da(), cmap="viridis")
        assert "graticule" not in ctor_opts(ax.to_html(tmp_path / "ng.html").read_text())

    def test_gridlines_rejected_on_globe(self):
        with pytest.raises(NotImplementedError, match="2D-projection only"):
            Axes().gridlines()


class TestExtent:
    def test_set_extent(self, tmp_path):
        ax = Axes(projection=ccrs.PlateCarree())
        ax.pcolormesh(make_da(), cmap="viridis")
        ax.set_extent([-141, -52, 41, 84])
        opts = ctor_opts(ax.to_html(tmp_path / "e.html").read_text())
        assert opts["extent"] == [-141.0, -52.0, 41.0, 84.0]

    def test_set_global_clears_extent(self):
        ax = Axes(projection=ccrs.Mercator())     # ships a default extent
        assert ax.get_extent() is not None
        ax.set_global()
        assert ax.get_extent() is None

    def test_set_extent_wrong_length(self):
        with pytest.raises(ValueError, match=r"lon0, lon1, lat0, lat1"):
            Axes(projection=ccrs.PlateCarree()).set_extent([0, 10, 20])

    def test_set_extent_rejects_projected_crs(self):
        with pytest.raises(NotImplementedError, match="lon/lat degrees"):
            Axes(projection=ccrs.PlateCarree()).set_extent(
                [0, 1, 2, 3], crs=ccrs.Mercator())

    def test_platecarree_crs_accepted(self):
        ax = Axes(projection=ccrs.PlateCarree())
        ax.set_extent([-10, 10, -5, 5], crs=ccrs.PlateCarree())
        assert ax.get_extent() == (-10.0, 10.0, -5.0, 5.0)


class TestPointLineText:
    def test_plot_with_fmt_string(self, tmp_path):
        ax = Axes(projection=ccrs.PlateCarree())
        ax.plot([-140, -100, -60], [30, 60, 45], "r--o")
        html = ax.to_html(tmp_path / "p.html").read_text()
        assert "map.plot([[-140.0, 30.0]" in html
        assert '"color": "#d62728"' in html
        assert '"linestyle": "--"' in html
        assert '"marker": "o"' in html

    def test_plot_kwargs_beat_fmt(self, tmp_path):
        ax = Axes(projection=ccrs.PlateCarree())
        ax.plot([0, 10], [0, 10], "r-", color="#00ff00", linewidth=3)
        html = ax.to_html(tmp_path / "p2.html").read_text()
        assert '"color": "#00ff00"' in html
        assert '"linewidth": 3' in html

    def test_geodetic_transform_sets_flag(self, tmp_path):
        ax = Axes(projection=ccrs.PlateCarree())
        ax.plot([-170, 170], [20, 20], transform=ccrs.Geodetic())
        assert '"geodesic": true' in ax.to_html(tmp_path / "gd.html").read_text()

    def test_platecarree_transform_is_straight(self, tmp_path):
        ax = Axes(projection=ccrs.PlateCarree())
        ax.plot([-170, 170], [20, 20], transform=ccrs.PlateCarree())
        assert '"geodesic"' not in ax.to_html(tmp_path / "st.html").read_text()

    def test_unsupported_transform_rejected(self):
        with pytest.raises(NotImplementedError, match="transform must be"):
            Axes(projection=ccrs.PlateCarree()).plot([0], [0],
                                                     transform=ccrs.Mercator())

    def test_plot_length_mismatch(self):
        with pytest.raises(ValueError, match="same length"):
            Axes(projection=ccrs.PlateCarree()).plot([0, 1, 2], [0, 1])

    def test_scatter_with_values_and_cmap(self, tmp_path):
        ax = Axes(projection=ccrs.PlateCarree())
        ax.scatter([-95, -75], [45, 20], c=[1.0, 5.0], cmap="plasma", s=8)
        html = ax.to_html(tmp_path / "sc.html").read_text()
        assert "map.scatter([-95.0, -75.0], [45.0, 20.0]" in html
        assert '"values": [1.0, 5.0]' in html
        assert '"cmap": "plasma"' in html
        assert '"size": 8.0' in html

    def test_scatter_per_point_sizes(self, tmp_path):
        ax = Axes(projection=ccrs.PlateCarree())
        ax.scatter([0, 10], [0, 10], s=[3, 9])
        assert '"sizes": [3.0, 9.0]' in ax.to_html(tmp_path / "ss.html").read_text()

    def test_text(self, tmp_path):
        ax = Axes(projection=ccrs.PlateCarree())
        ax.text(-100, 70, "Canada", fontsize=14, ha="left")
        html = ax.to_html(tmp_path / "t.html").read_text()
        assert 'map.text(-100.0, 70.0, "Canada"' in html
        assert '"fontSize": 14' in html
        assert '"anchor": "start"' in html

    def test_annotate_offsets(self, tmp_path):
        ax = Axes(projection=ccrs.PlateCarree())
        ax.annotate("here", (10, 20))
        html = ax.to_html(tmp_path / "an.html").read_text()
        assert 'map.text(10.0, 20.0, "here"' in html
        assert '"dy": -12' in html

    def test_2d_only_primitives_rejected_on_globe(self):
        ax = Axes()
        for call in (lambda: ax.plot([0], [0]),
                     lambda: ax.scatter([0], [0]),
                     lambda: ax.text(0, 0, "x"),
                     lambda: ax.tissot(),
                     lambda: ax.add_geometries({"type": "Point",
                                                "coordinates": [0, 0]})):
            with pytest.raises(NotImplementedError, match="2D-projection only"):
                call()


class TestVectors:
    def test_barbs(self, tmp_path):
        u, v = make_uv()
        ax = Axes(projection=ccrs.PlateCarree())
        ax.barbs(u, v, length=9, flip=True)
        html = ax.to_html(tmp_path / "bb.html").read_text()
        assert "map.barbs(P0.lons, P0.lats, P0.field, P1.field" in html
        assert '"length": 9' in html
        assert '"flip": true' in html

    def test_quiver_and_barbs_share_payloads(self, tmp_path):
        u, v = make_uv()
        ax = Axes(projection=ccrs.PlateCarree())
        ax.quiver(u, v)
        ax.barbs(u, v)
        html = ax.to_html(tmp_path / "qb.html").read_text()
        script = html.rsplit("const map = ", 1)[1]
        assert script.count("await unpackField(") == 2   # u and v, once each

    def test_barbs_rejects_non_dataarray(self):
        with pytest.raises(TypeError, match="expects xarray DataArrays"):
            Axes(projection=ccrs.PlateCarree()).barbs([1, 2], [3, 4])

    def test_streamplot_emits_geometry(self, tmp_path):
        u, v = make_uv()
        ax = Axes(projection=ccrs.PlateCarree())
        ax.streamplot(u, v, density=0.6, color="#8fd")
        html = ax.to_html(tmp_path / "sp.html").read_text()
        assert "map.addGeoJSON(" in html
        assert '"type": "MultiLineString"' in html
        assert '"color": "#8fd"' in html

    def test_streamplot_of_a_zero_field_draws_nothing(self, tmp_path):
        u, v = make_uv()
        zero = xr.zeros_like(u)
        ax = Axes(projection=ccrs.PlateCarree())
        ax.streamplot(zero, xr.zeros_like(v))
        assert "map.addGeoJSON(" not in ax.to_html(tmp_path / "z.html").read_text()


class TestGeometries:
    def test_tissot_emits_multipolygon(self, tmp_path):
        ax = Axes(projection=ccrs.Mollweide())
        ax.tissot(rad_km=500, lons=[0, 90], lats=[0, 45])
        html = ax.to_html(tmp_path / "ti.html").read_text()
        assert '"type": "MultiPolygon"' in html

    def test_add_geometries_geojson(self, tmp_path):
        ax = Axes(projection=ccrs.PlateCarree())
        ax.add_geometries({"type": "LineString",
                           "coordinates": [[0, 0], [10, 10]]},
                          facecolor="none", edgecolor="#f0f")
        html = ax.to_html(tmp_path / "gj.html").read_text()
        assert '"type": "LineString"' in html
        assert '"color": "#f0f"' in html

    def test_add_geometries_geo_interface(self, tmp_path):
        class FakeShape:
            __geo_interface__ = {"type": "Point", "coordinates": [5.0, 5.0]}

        ax = Axes(projection=ccrs.PlateCarree())
        ax.add_geometries(FakeShape())
        assert '"coordinates": [5.0, 5.0]' in ax.to_html(tmp_path / "gi.html").read_text()

    def test_add_geometries_rejects_projected_crs(self):
        with pytest.raises(NotImplementedError, match="lon/lat"):
            Axes(projection=ccrs.PlateCarree()).add_geometries(
                {"type": "Point", "coordinates": [0, 0]}, crs=ccrs.Robinson())

    def test_add_geometries_bad_type(self):
        with pytest.raises(TypeError, match="__geo_interface__"):
            Axes(projection=ccrs.PlateCarree()).add_geometries(42)


class TestLegendAndTitle:
    def test_legend_lists_labelled_layers(self, tmp_path):
        ax = Axes(projection=ccrs.PlateCarree())
        ax.plot([0, 10], [0, 10], color="#f00", label="track")
        ax.scatter([5], [5], color="#0f0", label="site")
        ax.legend(title="key")
        html = ax.to_html(tmp_path / "lg.html").read_text()
        assert "tp-legend" in html
        assert ">track<" in html and ">site<" in html
        assert "key" in html

    def test_no_legend_without_labels(self, tmp_path):
        ax = Axes(projection=ccrs.PlateCarree())
        ax.plot([0, 10], [0, 10])
        ax.legend()
        assert 'class="tp-legend"' not in ax.to_html(tmp_path / "nl.html").read_text()

    def test_legend_escapes_labels(self, tmp_path):
        ax = Axes(projection=ccrs.PlateCarree())
        ax.plot([0, 1], [0, 1], label="<script>x</script>")
        ax.legend()
        html = ax.to_html(tmp_path / "esc.html").read_text()
        assert "&lt;script&gt;" in html

    def test_set_title_alias(self, tmp_path):
        ax = Axes(projection=ccrs.PlateCarree())
        ax.pcolormesh(make_da(), cmap="viridis")
        ax.set_title("hello")
        assert "hello" in ax.to_html(tmp_path / "ti2.html").read_text()

    def test_savefig_alias(self, tmp_path):
        ax = Axes(projection=ccrs.PlateCarree())
        ax.pcolormesh(make_da(), cmap="viridis")
        out = ax.savefig(tmp_path / "sf.html")
        assert out.exists()


class TestChaining:
    def test_every_primitive_returns_self(self):
        u, v = make_uv()
        da = make_da()
        ax = Axes(projection=ccrs.PlateCarree())
        result = (ax.pcolormesh(da, cmap="viridis")
                    .contour(da, levels=5, color="#000")
                    .coastlines()
                    .borders()
                    .gridlines()
                    .plot([0, 1], [0, 1])
                    .scatter([0], [0])
                    .text(0, 0, "x")
                    .quiver(u, v)
                    .barbs(u, v)
                    .tissot()
                    .set_extent([-10, 10, -10, 10])
                    .set_global()
                    .set_title("t")
                    .legend())
        assert result is ax


def test_module_exports():
    assert tp.crs is ccrs
    assert tp.feature is cfeature
