"""
xarray Dataset accessor — `ds.tp`.

For wider operations than DataArray.tp:
  - vector field plotting (u, v components together)
  - side-by-side comparison plots (two variables, e.g. model vs reanalysis)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from .serialize import serialize
from .binary import pack_field
from .accessor import _load_terraplot_bundle, _SHARED_CSS, _IMPORTMAP, _js


@xr.register_dataset_accessor("tp")
class TerraplotDatasetAccessor:
    """
    Dataset-level helpers complementing DataArray.tp.

    Examples
    --------
    >>> ds = xr.open_dataset("era5_winds.nc")
    >>> ds.tp.quiver_html("winds.html", u="u10", v="v10",
    ...                   background="t2m", projection="orthographic")

    >>> ds.tp.compare_html("compare.html", a="t2m_model", b="t2m_era5",
    ...                    projection="naturalEarth")
    """

    def __init__(self, ds: xr.Dataset) -> None:
        self._ds = ds

    def serve_viewer(
        self,
        port: int = 8765,
        host: str = "127.0.0.1",
        open_browser: bool = True,
    ) -> None:
        """
        Start an interactive visual NetCDF viewer server for this dataset.
        """
        from .server import start_viewer
        start_viewer(self._ds, host=host, port=port, open_browser=open_browser)

    # ── Vector field (quiver) HTML export ────────────────────────────────────

    def quiver_html(
        self,
        path: str | Path,
        *,
        u: str,
        v: str,
        background: str | None = None,
        title: str = "terraplot quiver",
        cmap: str = "viridis",
        quiver_color: str = "rgba(255,255,255,0.85)",
        quiver_cmap: str | None = None,
        quiver_density: int | None = None,
        quiver_scale: float = 22,
        alpha: float = 0.7,
        vmin: float | None = None,
        vmax: float | None = None,
        projection: str = "equirectangular",
        coastlines: bool = True,
        center: tuple[float, float] = (0, 0),
        extent: tuple[float, float, float, float] | None = None,
        terraplot_bundle: str | Path | None = None,
        lon_dim: str | None = None,
        lat_dim: str | None = None,
        wrap_lon: bool = True,
    ) -> Path:
        """
        Render an HTML page showing wind/current arrows over an optional
        scalar background field.

        Parameters
        ----------
        path           : output .html path
        u, v           : variable names of east-west / north-south components
        background     : optional scalar variable name (e.g. wind speed, temperature)
        quiver_color   : single colour for all arrows
        quiver_cmap    : cmap name to colour arrows by magnitude (overrides quiver_color)
        quiver_density : subsample factor (1 = every grid point); auto if None
        quiver_scale   : pixels per max-magnitude arrow
        projection     : 2-D projection (defaults to equirectangular)

        See `da.tp.to_html` for descriptions of the colour-field params.
        """
        ds = self._ds
        u_da = ds[u]
        v_da = ds[v]

        u_payload = serialize(u_da, lon_dim=lon_dim, lat_dim=lat_dim, wrap_lon=wrap_lon)
        v_payload = serialize(v_da, lon_dim=lon_dim, lat_dim=lat_dim, wrap_lon=wrap_lon)

        if background is not None:
            bg_payload = serialize(ds[background], lon_dim=lon_dim, lat_dim=lat_dim, wrap_lon=wrap_lon)
            bg_b64 = pack_field(bg_payload)
            bg_long = bg_payload.get("long_name") or bg_payload.get("name") or background
            bg_units = bg_payload.get("units", "")
            label = f"{bg_long} [{bg_units}]" if bg_units else bg_long
        else:
            bg_payload = None
            bg_b64 = None
            label = f"{u} / {v}"

        u_b64 = pack_field(u_payload)
        v_b64 = pack_field(v_payload)
        bundle_js = _load_terraplot_bundle(terraplot_bundle)

        center_js = f"[{center[0]}, {center[1]}]"
        extent_js = (f", extent: [{extent[0]}, {extent[1]}, {extent[2]}, {extent[3]}]"
                     if extent else "")
        coastlines_js = "map.addFeature('coastlines');" if coastlines else ""

        bg_init = ""
        if bg_b64 is not None:
            bg_init = f"""
const bg = await unpackField("{bg_b64}");
map.pcolormesh(bg.lons, bg.lats, bg.field, {{
  cmap: '{cmap}', alpha: {alpha},
  vmin: {_js(vmin)}, vmax: {_js(vmax)},
  name: bg.name, units: bg.units,
}});
"""

        cbar_block = ""
        if bg_b64 is not None:
            cbar_block = f"""
<div id="colorbar"></div>
<script type="module">
import {{ Colorbar }} from 'data:text/javascript;base64,';
</script>
"""

        quiver_opts = (
            f"density: {quiver_density if quiver_density is not None else 'null'}, "
            f"scale: {quiver_scale}, "
            f"color: '{quiver_color}', "
            f"cmap: {('null' if quiver_cmap is None else repr(quiver_cmap))}"
        )

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>{title}</title>
<style>
{_SHARED_CSS}
</style>
</head>
<body>
<div id="map"></div>
<div id="label">{label}</div>
{_IMPORTMAP}
<script type="module">
{bundle_js}

const map = new GeoMap('#map', {{
  projection: '{projection}',
  center: {center_js}{extent_js},
  tooltip: true,
}});
{coastlines_js}
{bg_init}

const uData = await unpackField("{u_b64}");
const vData = await unpackField("{v_b64}");
map.quiver(uData.lons, uData.lats, uData.field, vData.field, {{ {quiver_opts} }});
</script>
</body>
</html>"""
        p = Path(path)
        p.write_text(html)
        return p

    # ── Two-panel side-by-side comparison ────────────────────────────────────

    def compare_html(
        self,
        path: str | Path,
        *,
        a: str,
        b: str,
        title: str = "terraplot comparison",
        cmap: str = "RdBu_r",
        alpha: float = 0.85,
        vmin: float | None = None,
        vmax: float | None = None,
        symmetric: bool = True,
        projection: str = "equirectangular",
        coastlines: bool = True,
        center: tuple[float, float] = (0, 0),
        extent: tuple[float, float, float, float] | None = None,
        terraplot_bundle: str | Path | None = None,
        lon_dim: str | None = None,
        lat_dim: str | None = None,
        wrap_lon: bool = True,
    ) -> Path:
        """
        Render two GeoMaps side-by-side: variable `a` on the left, `b` on the right.
        Useful for model-vs-reanalysis comparisons.

        Sets a single shared color scale across both panels.
        """
        ds = self._ds
        pa = serialize(ds[a], lon_dim=lon_dim, lat_dim=lat_dim, wrap_lon=wrap_lon)
        pb = serialize(ds[b], lon_dim=lon_dim, lat_dim=lat_dim, wrap_lon=wrap_lon)

        # Compute shared range across both fields
        if vmin is None or vmax is None:
            arr_a = np.asarray([v for row in pa["field"] for v in row if v is not None])
            arr_b = np.asarray([v for row in pb["field"] for v in row if v is not None])
            both = np.concatenate([arr_a.ravel(), arr_b.ravel()])
            mn, mx = float(both.min()), float(both.max())
            if symmetric:
                a_abs = max(abs(mn), abs(mx))
                if vmin is None: vmin = -a_abs
                if vmax is None: vmax = +a_abs
            else:
                if vmin is None: vmin = mn
                if vmax is None: vmax = mx

        a_b64 = pack_field(pa)
        b_b64 = pack_field(pb)
        bundle_js = _load_terraplot_bundle(terraplot_bundle)

        center_js = f"[{center[0]}, {center[1]}]"
        extent_js = (f", extent: [{extent[0]}, {extent[1]}, {extent[2]}, {extent[3]}]"
                     if extent else "")
        coastlines_js = "  m.addFeature('coastlines');\n" if coastlines else ""

        units = pa.get("units", "") or pb.get("units", "")

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>{title}</title>
<style>
{_SHARED_CSS}
  body {{ display: flex; flex-direction: column; }}
  #row {{ display: flex; flex: 1; min-height: 0; }}
  .panel {{ flex: 1; position: relative; border-right: 1px solid #1d1d2a; min-height: 0; }}
  .panel:last-child {{ border-right: none; }}
  .panel .panel-label {{
    position: absolute; top: .8rem; left: .8rem;
    background: rgba(0,0,0,.6); padding: .25rem .6rem; border-radius: 4px;
    font-size: .75rem; color: #cbd5e1; pointer-events: none; z-index: 4;
  }}
  .panel-map {{ width: 100%; height: 100%; }}
  #shared-cbar {{
    position: fixed; bottom: 1rem; left: 50%; transform: translateX(-50%);
  }}
</style>
</head>
<body>
<div id="row">
  <div class="panel">
    <div class="panel-label">{a}</div>
    <div id="map-a" class="panel-map"></div>
  </div>
  <div class="panel">
    <div class="panel-label">{b}</div>
    <div id="map-b" class="panel-map"></div>
  </div>
</div>
<div id="shared-cbar"></div>
{_IMPORTMAP}
<script type="module">
{bundle_js}

const aData = await unpackField("{a_b64}");
const bData = await unpackField("{b_b64}");

const opts = {{ cmap: '{cmap}', alpha: {alpha}, vmin: {vmin}, vmax: {vmax} }};

for (const [el, data] of [['#map-a', aData], ['#map-b', bData]]) {{
  const m = new GeoMap(el, {{
    projection: '{projection}',
    center: {center_js}{extent_js},
    tooltip: true,
  }});
{coastlines_js}  m.pcolormesh(data.lons, data.lats, data.field, {{
    ...opts, name: data.name, units: data.units,
  }});
}}

new Colorbar('#shared-cbar', {{
  cmap: '{cmap}', vmin: {vmin}, vmax: {vmax},
  label: 'shared scale {units}', ticks: 5,
}});
</script>
</body>
</html>"""
        p = Path(path)
        p.write_text(html)
        return p
