"""Tests for the matplotlib-style Figure / subplots layer."""
import numpy as np
import pytest
import xarray as xr

import pyterraplot as tp
import pyterraplot.crs as ccrs
from pyterraplot import Axes, Figure


def make_da(name="air", seed=1):
    rng = np.random.default_rng(seed)
    lats = np.linspace(90, -90, 18)
    lons = np.linspace(-180, 170, 36)
    LON, LAT = np.meshgrid(lons, lats)
    return xr.DataArray(
        20 * np.cos(np.deg2rad(LAT)) + rng.standard_normal((18, 36)) * 0.1,
        dims=("lat", "lon"), coords={"lat": lats, "lon": lons},
        name=name, attrs={"units": "degC", "long_name": "test field"},
    )


class TestSubplots:
    def test_single_panel_squeezes_to_axes(self):
        fig, ax = tp.subplots(projection=ccrs.PlateCarree())
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_row_squeezes_to_1d(self):
        fig, axes = tp.subplots(1, 3, projection=ccrs.Robinson())
        assert axes.shape == (3,)
        assert all(isinstance(a, Axes) for a in axes)

    def test_grid_returns_2d(self):
        fig, axes = tp.subplots(2, 2, projection=ccrs.PlateCarree())
        assert axes.shape == (2, 2)
        assert len(fig.axes) == 4

    def test_squeeze_false_keeps_2d(self):
        fig, axes = tp.subplots(1, 1, squeeze=False,
                                projection=ccrs.PlateCarree())
        assert axes.shape == (1, 1)

    def test_subplot_kw_projection(self):
        fig, axes = tp.subplots(1, 2,
                                subplot_kw={"projection": ccrs.Mollweide()})
        assert all(a.crs.tp_name == "mollweide" for a in axes)

    def test_subplot_kw_passes_other_axes_options(self):
        fig, ax = tp.subplots(subplot_kw={"projection": ccrs.PlateCarree(),
                                          "tooltip": False})
        assert ax.tooltip is False

    def test_panels_default_to_globe(self):
        fig, axes = tp.subplots(1, 2)
        assert all(a.globe for a in axes)


class TestAddSubplot:
    def test_three_argument_form(self):
        fig = tp.figure()
        ax = fig.add_subplot(2, 2, 3, projection=ccrs.PlateCarree())
        assert fig._positions == [(1, 0, 1, 1)]     # row 1, col 0
        assert isinstance(ax, Axes)

    def test_three_digit_shorthand(self):
        fig = tp.figure()
        fig.add_subplot(224, projection=ccrs.PlateCarree())
        assert fig._positions == [(1, 1, 1, 1)]
        assert (fig.nrows, fig.ncols) == (2, 2)

    def test_no_args_appends(self):
        fig = Figure(nrows=1, ncols=3)
        fig.add_subplot(projection=ccrs.PlateCarree())
        fig.add_subplot(projection=ccrs.PlateCarree())
        assert fig._positions == [(0, 0, 1, 1), (0, 1, 1, 1)]

    def test_spans(self, tmp_path):
        fig = Figure(nrows=2, ncols=2)
        fig.add_subplot(2, 2, 1, colspan=2, projection=ccrs.PlateCarree())
        html = fig.to_html(tmp_path / "span.html").read_text()
        assert "grid-column: 1 / span 2" in html

    def test_bad_positional_args(self):
        with pytest.raises(TypeError, match="add_subplot takes"):
            tp.figure().add_subplot(1, 2)


class TestFigureRendering:
    def test_one_map_per_panel(self, tmp_path):
        fig, axes = tp.subplots(2, 2, projection=ccrs.PlateCarree())
        for ax in axes.flat:
            ax.pcolormesh(make_da(), cmap="viridis")
        html = fig.to_html(tmp_path / "f.html").read_text()
        assert html.count("new GeoMap(") == 4
        assert html.count('class="tp-panel"') == 4
        for i in range(4):
            assert f'id="map{i}"' in html
            assert f"const map{i} = " in html

    def test_shared_payload_registry(self, tmp_path):
        da = make_da()                       # one array, drawn in both panels
        fig, axes = tp.subplots(1, 2, projection=ccrs.PlateCarree())
        for ax in axes:
            ax.pcolormesh(da, cmap="viridis")
        html = fig.to_html(tmp_path / "sh.html").read_text()
        assert html.count("= await unpackField(") == 1
        assert html.count("const P0 = ") == 1
        assert "const P1 = " not in html

    def test_distinct_arrays_get_distinct_payloads(self, tmp_path):
        fig, axes = tp.subplots(1, 2, projection=ccrs.PlateCarree())
        axes[0].pcolormesh(make_da("a", seed=1), cmap="viridis")
        axes[1].pcolormesh(make_da("b", seed=2), cmap="viridis")
        html = fig.to_html(tmp_path / "d.html").read_text()
        assert html.count("= await unpackField(") == 2
        assert "const P1 = " in html

    def test_mixed_projections_per_panel(self, tmp_path):
        fig = tp.figure()
        fig.add_subplot(1, 2, 1, projection=ccrs.NorthPolarStereo())
        fig.add_subplot(1, 2, 2, projection=ccrs.Mollweide())
        html = fig.to_html(tmp_path / "mp.html").read_text()
        assert '"projection": "stereographic"' in html
        assert '"projection": "mollweide"' in html

    def test_globe_panel_alongside_map_panel(self, tmp_path):
        fig = tp.figure()
        fig.add_subplot(1, 2, 1)                                  # globe
        fig.add_subplot(1, 2, 2, projection=ccrs.Robinson())      # flat
        html = fig.to_html(tmp_path / "gm.html").read_text()
        assert "new GeoSphere('#map0'" in html
        assert "new GeoMap('#map1'" in html

    def test_panel_titles(self, tmp_path):
        fig, axes = tp.subplots(1, 2, projection=ccrs.PlateCarree())
        axes[0].set_title("left")
        axes[1].set_title("right")
        html = fig.to_html(tmp_path / "pt.html").read_text()
        assert ">left<" in html and ">right<" in html

    def test_suptitle(self, tmp_path):
        fig, ax = tp.subplots(projection=ccrs.PlateCarree())
        fig.suptitle("the whole thing")
        html = fig.to_html(tmp_path / "st.html").read_text()
        assert '<div id="suptitle">the whole thing</div>' in html
        assert "<title>the whole thing</title>" in html

    def test_figsize_sets_canvas(self, tmp_path):
        fig, ax = tp.subplots(figsize=(12, 6), projection=ccrs.PlateCarree())
        html = fig.to_html(tmp_path / "fs.html").read_text()
        assert "width: min(100vw, 1200px)" in html
        assert "height: 600px" in html

    def test_grid_template(self, tmp_path):
        fig, axes = tp.subplots(3, 2, projection=ccrs.PlateCarree())
        html = fig.to_html(tmp_path / "gt.html").read_text()
        assert "grid-template-rows: repeat(3, 1fr)" in html
        assert "grid-template-columns: repeat(2, 1fr)" in html

    def test_empty_figure_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="no panels"):
            tp.figure().to_html(tmp_path / "e.html")

    def test_savefig_alias(self, tmp_path):
        fig, ax = tp.subplots(projection=ccrs.PlateCarree())
        ax.pcolormesh(make_da(), cmap="viridis")
        assert fig.savefig(tmp_path / "sf.html").exists()

    def test_repr_html_is_iframe(self):
        fig, ax = tp.subplots(projection=ccrs.PlateCarree())
        ax.pcolormesh(make_da(), cmap="viridis")
        assert fig._repr_html_().startswith("<iframe srcdoc=")

    def test_json_mode(self, tmp_path):
        fig, ax = tp.subplots(projection=ccrs.PlateCarree())
        ax.pcolormesh(make_da(), cmap="viridis")
        html = fig.to_html(tmp_path / "j.html", binary=False).read_text()
        # The inlined bundle always defines unpackField; what matters is that
        # the payload declaration is a JSON literal rather than a call to it.
        assert "const P0 = await unpackField(" not in html
        assert 'const P0 = {"lons"' in html


class TestFigureColorbar:
    def test_shared_colorbar_inherits_field_settings(self, tmp_path):
        fig, axes = tp.subplots(1, 2, projection=ccrs.PlateCarree())
        for ax in axes:
            ax.pcolormesh(make_da(), cmap="RdBu_r", vmin=-30, vmax=30)
        fig.colorbar()
        html = fig.to_html(tmp_path / "cb.html").read_text()
        assert 'id="fig-cbar"' in html
        assert '"cmap": "RdBu_r"' in html
        assert '"vmin": -30' in html
        assert "test field [degC]" in html

    def test_explicit_settings_win(self, tmp_path):
        fig, ax = tp.subplots(projection=ccrs.PlateCarree())
        ax.pcolormesh(make_da(), cmap="viridis")
        fig.colorbar(cmap="magma", vmin=0, vmax=1, label="custom")
        html = fig.to_html(tmp_path / "cb2.html").read_text()
        assert '"cmap": "magma"' in html
        assert '"label": "custom"' in html

    def test_no_colorbar_by_default(self, tmp_path):
        fig, ax = tp.subplots(projection=ccrs.PlateCarree())
        ax.pcolormesh(make_da(), cmap="viridis")
        assert 'id="fig-cbar"' not in fig.to_html(tmp_path / "nc.html").read_text()

    def test_per_panel_colorbar_targets_its_panel(self, tmp_path):
        fig, axes = tp.subplots(1, 2, projection=ccrs.PlateCarree())
        for ax in axes:
            ax.pcolormesh(make_da(), cmap="viridis")
            ax.colorbar(orientation="vertical")
        html = fig.to_html(tmp_path / "pc.html").read_text()
        assert "document.querySelector('#cbar-map0')" in html
        assert "document.querySelector('#cbar-map1')" in html


class TestFigureAnimation:
    def test_animated_panels_get_distinct_frame_vars(self, tmp_path):
        da = make_da().expand_dims(step=range(2))
        fig, axes = tp.subplots(1, 2, projection=ccrs.PlateCarree())
        for ax in axes:
            ax.animate(da, dim="step", kind="pcolormesh")
        html = fig.to_html(tmp_path / "an.html").read_text()
        assert "const F0 = await unpackFrames(" in html
        assert "const F1 = await unpackFrames(" in html
        assert "map0.animate(F0" in html
        assert "map1.animate(F1" in html


def test_exported_from_package():
    assert tp.Figure is Figure
    assert callable(tp.figure) and callable(tp.subplots)
