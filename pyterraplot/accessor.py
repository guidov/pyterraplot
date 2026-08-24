"""
xarray accessor: da.tp.*

Registered automatically when pyterraplot is imported.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import xarray as xr

from .serialize import serialize
from .binary import pack_field, pack_frames


@xr.register_dataarray_accessor("tp")
class TerraplotAccessor:
    """
    Cartopy-style plotting on a terraplot 3D globe, from xarray.

    Usage
    -----
    import pyterraplot                # registers .tp accessor
    import xarray as xr

    ds  = xr.open_dataset("ecmwf_s2s.nc")
    t2m = ds["2m_temperature"].isel(time=0)

    # Export for the browser
    t2m.tp.to_json("field.json")

    # Live-serve from Python (requires pyterraplot[serve])
    t2m.tp.serve(port=8765)

    # Get raw dict (pass directly to your own FastAPI route etc.)
    payload = t2m.tp.to_dict()
    """

    def __init__(self, da: xr.DataArray) -> None:
        self._da = da

    # ── Serialisation ──────────────────────────────────────────────────────────

    def to_dict(
        self,
        lon_dim: str | None = None,
        lat_dim: str | None = None,
        wrap_lon: bool = True,
    ) -> dict[str, Any]:
        """Return the terraplot JSON payload as a Python dict."""
        return serialize(self._da, lon_dim=lon_dim, lat_dim=lat_dim, wrap_lon=wrap_lon)

    def to_json(
        self,
        path: str | Path,
        lon_dim: str | None = None,
        lat_dim: str | None = None,
        wrap_lon: bool = True,
    ) -> Path:
        """Write JSON file consumable by terraplot FieldLayer."""
        p = Path(path)
        serialize(self._da, lon_dim=lon_dim, lat_dim=lat_dim, wrap_lon=wrap_lon, path=p)
        return p

    def to_cog(
        self,
        path: str | Path | None = None,
        *,
        lon_dim: str | None = None,
        lat_dim: str | None = None,
        wrap_lon: bool = True,
        crs: str = "EPSG:4326",
    ):
        """Export to a Cloud-Optimized GeoTIFF (requires ``pyterraplot[raster]``).

        Returns the output ``Path`` when ``path`` is given, else the COG bytes.
        """
        from .cog import to_cog as _to_cog
        return _to_cog(self._da, path, lon_dim=lon_dim, lat_dim=lat_dim,
                       wrap_lon=wrap_lon, crs=crs)

    # ── Live server ───────────────────────────────────────────────────────────

    def serve(
        self,
        port: int = 8765,
        host: str = "127.0.0.1",
        lon_dim: str | None = None,
        lat_dim: str | None = None,
        wrap_lon: bool = True,
        open_browser: bool = False,
    ) -> None:
        """
        Start a local HTTP server that serves this field at GET /field.
        The browser fetches it and passes directly to terraplot FieldLayer.

        Requires: pip install pyterraplot[serve]
        """
        from .server import serve as _serve

        payload = self.to_dict(lon_dim=lon_dim, lat_dim=lat_dim, wrap_lon=wrap_lon)
        _serve(payload, host=host, port=port, open_browser=open_browser)

    # ── Multi-step / ensemble helpers ─────────────────────────────────────────

    def frames(
        self,
        dim: str,
        lon_dim: str | None = None,
        lat_dim: str | None = None,
        wrap_lon: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Serialise each slice along `dim` as a separate frame dict.
        Useful for forecast lead-time animation.

        Example
        -------
        # Returns list of dicts, one per lead time
        steps = ds["t2m"].tp.frames(dim="time")
        json.dump(steps, open("frames.json", "w"))
        """
        slices = [
            serialize(
                self._da.isel({dim: i}),
                lon_dim=lon_dim,
                lat_dim=lat_dim,
                wrap_lon=wrap_lon,
            )
            for i in range(self._da.sizes[dim])
        ]
        # Attach the coordinate value as metadata
        coord_vals = self._da[dim].values
        for i, s in enumerate(slices):
            s["frame"] = i
            s["coord_value"] = str(coord_vals[i])
        return slices

    def frames_to_json(
        self,
        path: str | Path,
        dim: str,
        **kwargs,
    ) -> Path:
        """Write all frames to a single JSON file."""
        p = Path(path)
        p.write_text(json.dumps(self.frames(dim=dim, **kwargs)))
        return p

    def frames_compact(
        self,
        dim: str,
        lon_dim: str | None = None,
        lat_dim: str | None = None,
        wrap_lon: bool = True,
    ) -> dict[str, Any]:
        """
        Compact multi-frame format: lons/lats stored once, only fields repeated.

        Returns { lons, lats, frames: [{ field, frame, coord_value }, ...] }

        This is what terraplot's animate() compact-format branch expects.
        Useful for S2S lead-time animations — avoids duplicating the grid
        per frame, cutting payload size by ~60 %.

        Example
        -------
        payload = ds["t2m"].tp.frames_compact(dim="time")
        json.dump(payload, open("frames_compact.json", "w"))
        """
        first = serialize(self._da.isel({dim: 0}), lon_dim=lon_dim, lat_dim=lat_dim, wrap_lon=wrap_lon)
        coord_vals = self._da[dim].values
        frame_list = []
        for i in range(self._da.sizes[dim]):
            s = serialize(self._da.isel({dim: i}), lon_dim=lon_dim, lat_dim=lat_dim, wrap_lon=wrap_lon)
            frame_list.append({
                "field":       s["field"],
                "frame":       i,
                "coord_value": str(coord_vals[i]),
            })
        return {
            "lons":   first["lons"],
            "lats":   first["lats"],
            "name":   first["name"],
            "units":  first["units"],
            "long_name": first["long_name"],
            "frames": frame_list,
        }

    def frames_compact_to_json(
        self,
        path: str | Path,
        dim: str,
        **kwargs,
    ) -> Path:
        """Write compact frames to a single JSON file."""
        p = Path(path)
        p.write_text(json.dumps(self.frames_compact(dim=dim, **kwargs)))
        return p

    # ── Jupyter inline rendering ──────────────────────────────────────────────

    def _repr_html_(self) -> str:
        """
        Inline rendering inside Jupyter notebooks.
        Only triggers for 2-D arrays; otherwise falls back to xarray's repr.

        The rendered HTML is a self-contained iframe-friendly globe + colorbar.
        """
        if self._da.ndim != 2:
            return None  # let xarray show its default repr

        try:
            import tempfile, base64
            html = self._build_html(
                kind="pcolormesh", title=str(self._da.name or "field"),
                cmap="viridis", alpha=0.85, vmin=None, vmax=None, levels=12,
                projection=None, coastlines=False, center=(0, 0), extent=None,
                binary=True, lon_dim=None, lat_dim=None, wrap_lon=True,
                terraplot_bundle=None, height_px=420,
            )
            # Use srcdoc-encoded iframe so the script runs in a sandboxed context
            srcdoc = html.replace("&", "&amp;").replace('"', "&quot;")
            return (
                f'<iframe srcdoc="{srcdoc}" width="100%" height="440" '
                f'style="border:1px solid rgba(255,255,255,.1); border-radius:6px;"></iframe>'
            )
        except Exception as e:
            return f"<pre>pyterraplot _repr_html_ failed: {e}</pre>"

    # ── Self-contained HTML export ────────────────────────────────────────────

    def to_html(
        self,
        path: str | Path,
        *,
        kind: str = "pcolormesh",
        title: str = "terraplot",
        cmap: str = "viridis",
        alpha: float = 0.7,
        vmin: float | None = None,
        vmax: float | None = None,
        levels: int = 12,
        projection: str | None = None,
        coastlines: bool | None = None,
        coastline_color: str = "#39FF14",
        coastline_width: float = 2.0,
        center: tuple[float, float] = (0, 0),
        extent: tuple[float, float, float, float] | None = None,
        binary: bool = True,
        lon_dim: str | None = None,
        lat_dim: str | None = None,
        wrap_lon: bool = True,
        terraplot_bundle: str | Path | None = None,
        earth_surface: str = "satellite",
        spin: bool = True,
        colorbar: dict | None = None,
        borders: bool = False,
        borders_color: str = "#000000",
        borders_width: float = 0.8,
        cities: bool = False,
        cities_color: str = "#ffffff",
    ) -> Path:
        """
        Export a self-contained HTML file that renders the field.

        With ``projection=None`` (default) renders a 3D globe (GeoSphere).
        With a projection name renders a 2D flat map (GeoMap) with interactive
        hover tooltips showing lat, lon, and field value.

        Parameters
        ----------
        path        : output file path (.html)
        kind        : 'pcolormesh' (smooth) or 'contourf' (banded)
        title       : page title
        cmap        : colormap name (any terraplot Colormaps key)
        alpha       : field opacity (0-1)
        vmin, vmax  : colormap range; auto-detected from data if None
        levels      : number of discrete bands for contourf (default 12)
        projection  : 2D map projection name, or None for 3D globe.
                      Supported: 'equirectangular' (= 'PlateCarree'), 'mercator',
                      'orthographic', 'naturalEarth', 'stereographic',
                      'azimuthalEqualArea', 'albers', 'lambertConformal',
                      'gnomonic'.
        coastlines  : draw continental outlines (default True for 2D maps,
                      False for the 3D globe — pass True for the neon-outline
                      look over a plain globe).
        coastline_color : stroke color for the coastlines (default neon green '#39FF14').
        coastline_width : stroke width in pixels (default 2). Values > 1 render
                      as proper thick lines; 1 gives a hairline.
        center      : (lon, lat) map centre for 2D projection (default (0, 0))
        extent      : (lon0, lon1, lat0, lat1) regional zoom, like cartopy set_extent.
                      Only used with 2D projection.
        binary      : use gzip-compressed float32 binary instead of JSON (default True).
                      Reduces HTML size 3-6× for typical climate grids.
                      Requires DecompressionStream (Chrome 80+, Firefox 113+, Safari 16.4+).
        terraplot_bundle : path to terraplot dist/terraplot.js; auto-detected
                           if None (looks for sibling repo ../terraplot)
        earth_surface : globe/map base style: 'satellite' (night-lights, default),
                      'shaded_relief' | 'stock' (blue marble), 'outline' | 'none'
                      (plain dark sphere — use with ``coastlines=True`` for the
                      neon-outline look), or an image URL.
        spin        : auto-rotate the 3D globe (default True). Ignored for 2D
                      projections. The globe stays draggable/zoomable either way.
        colorbar    : dict of Colorbar widget options — orientation
                      ('horizontal'|'vertical'), position ('bottom'|'top'|
                      'left'|'right'), panel, background, ticks (count or
                      values), format, scale ('linear'|'log'|'symlog'|'power'|
                      'sqrt'), power, linthresh, width, height, label.
        borders     : draw political country borders (default False).
        borders_color : stroke color for country borders (default '#000000').
        borders_width : stroke width in pixels for country borders (default 0.8).
        cities      : draw city markers and labels (default False).
        cities_color : text color for city labels (default '#ffffff').
        """
        if coastlines is None:
            coastlines = projection is not None  # default: 2D yes, 3D no
        html = self._build_html(
            kind=kind, title=title, cmap=cmap, alpha=alpha,
            vmin=vmin, vmax=vmax, levels=levels,
            projection=projection, coastlines=coastlines,
            center=center, extent=extent, binary=binary,
            lon_dim=lon_dim, lat_dim=lat_dim, wrap_lon=wrap_lon,
            terraplot_bundle=terraplot_bundle,
            earth_surface=earth_surface, spin=spin,
            coastline_color=coastline_color, coastline_width=coastline_width,
            cbar_opts=colorbar,
            borders=borders, borders_color=borders_color, borders_width=borders_width,
            cities=cities, cities_color=cities_color,
        )
        p = Path(path)
        p.write_text(html)
        return p

    def _build_html(
        self, *, kind, title, cmap, alpha, vmin, vmax, levels,
        projection, coastlines, center, extent, binary,
        lon_dim, lat_dim, wrap_lon, terraplot_bundle, earth_surface="satellite", height_px=None,
        spin: bool = True, coastline_color: str = "#39FF14", coastline_width: float = 2.0,
        cbar_opts: dict | None = None,
        borders: bool = False, borders_color: str = "#000000", borders_width: float = 0.8,
        cities: bool = False, cities_color: str = "#ffffff",
    ) -> str:
        """Shared HTML builder used by to_html() and _repr_html_()."""
        if kind not in ("pcolormesh", "contourf", "contour"):
            raise ValueError(f"kind must be 'pcolormesh', 'contourf', or 'contour', got {kind!r}")
        payload = self.to_dict(lon_dim=lon_dim, lat_dim=lat_dim, wrap_lon=wrap_lon)
        long_name = payload.get("long_name") or payload.get("name") or title
        units = payload.get("units", "")
        label = f"{long_name} [{units}]" if units else long_name

        bundle_js = _load_terraplot_bundle(terraplot_bundle)

        use_2d = projection is not None
        cbar_id = "cbar"

        if binary:
            b64 = pack_field(payload)
            payload_js = f'await unpackField("{b64}")'
        else:
            payload_js = json.dumps(payload)

        if use_2d:
            map_init = _render_geomap_js(kind, cmap, alpha, vmin, vmax, levels,
                                         projection, coastlines, center, units, extent,
                                         earth_surface=earth_surface, coastline_color=coastline_color,
                                         coastline_width=coastline_width,
                                         borders=borders, borders_color=borders_color, borders_width=borders_width,
                                         cities=cities, cities_color=cities_color)
        else:
            map_init = _render_geosphere_js(kind, cmap, alpha, vmin, vmax, levels,
                                            earth_surface=earth_surface, spin=spin,
                                            coastlines=coastlines, coastline_color=coastline_color,
                                            coastline_width=coastline_width,
                                            borders=borders, borders_color=borders_color, borders_width=borders_width,
                                            cities=cities, cities_color=cities_color)

        return _html_single(
            title=title, label=label, units=units,
            cbar_id=cbar_id, bundle_js=bundle_js,
            payload_js=payload_js, map_init=map_init,
            cmap=cmap, vmin=vmin, vmax=vmax,
            height_px=height_px,
            cbar_opts=cbar_opts,
        )

    # ── Animation HTML export ─────────────────────────────────────────────────

    def frames_to_html(
        self,
        path: str | Path,
        dim: str,
        *,
        kind: str = "pcolormesh",
        title: str = "terraplot",
        cmap: str = "viridis",
        alpha: float = 0.7,
        vmin: float | None = None,
        vmax: float | None = None,
        levels: int = 12,
        projection: str | None = None,
        coastlines: bool | None = None,
        center: tuple[float, float] = (0, 0),
        extent: tuple[float, float, float, float] | None = None,
        interval: int = 700,
        lon_dim: str | None = None,
        lat_dim: str | None = None,
        wrap_lon: bool = True,
        terraplot_bundle: str | Path | None = None,
        earth_surface: str = "satellite",
        spin: bool = True,
        coastline_color: str = "#39FF14",
        coastline_width: float = 2.0,
        colorbar: dict | None = None,
        borders: bool = False,
        borders_color: str = "#000000",
        borders_width: float = 0.8,
        cities: bool = False,
        cities_color: str = "#ffffff",
    ) -> Path:
        """
        Export a self-contained animated HTML file.

        Uses gzip-compressed float32 binary (DecompressionStream) so even
        12-month global animations stay under ~3 MB.

        Includes play/pause, frame scrubber, and frame label overlay.

        Parameters
        ----------
        path       : output .html path
        dim        : dimension to animate over (e.g. 'time', 'lead_time')
        kind       : 'pcolormesh' | 'contourf'
        projection : 2D map projection name, or None for 3D globe
        coastlines : draw continental outlines (default True for 2D, False for 3D)
        coastline_color : stroke color for the coastlines (default '#39FF14')
        coastline_width : stroke width in pixels (default 2)
        earth_surface : 'satellite' | 'shaded_relief' | 'outline'/'none' | URL
        interval   : ms between frames (default 700)
        spin       : auto-rotate the 3D globe (default True). Ignored for 2D.
        colorbar   : dict of Colorbar widget options (orientation, position,
                     panel, ticks, format, scale, …) — see to_html.
        borders     : draw political country borders (default False).
        borders_color : stroke color for country borders (default '#000000').
        borders_width : stroke width in pixels for country borders (default 0.8).
        cities      : draw city markers and labels (default False).
        cities_color : text color for city labels (default '#ffffff').
        """
        if kind not in ("pcolormesh", "contourf", "contour"):
            raise ValueError(f"kind must be 'pcolormesh', 'contourf', or 'contour', got {kind!r}")

        compact   = self.frames_compact(dim=dim, lon_dim=lon_dim, lat_dim=lat_dim, wrap_lon=wrap_lon)
        b64       = pack_frames(compact)
        n_frames  = len(compact["frames"])
        long_name = compact.get("long_name") or compact.get("name") or title
        units     = compact.get("units", "")
        label     = f"{long_name} [{units}]" if units else long_name

        bundle_js = _load_terraplot_bundle(terraplot_bundle)
        if coastlines is None:
            coastlines = projection is not None  # default: 2D yes, 3D no
        use_2d    = projection is not None
        levels_js = _js_levels(levels, kind)
        center_js = f"[{center[0]}, {center[1]}]"
        extent_js = (f", extent: [{extent[0]}, {extent[1]}, {extent[2]}, {extent[3]}]"
                     if extent else "")

        map_ctor = (
            f"new GeoMap('#map', {{ projection: '{projection}', center: {center_js}{extent_js}, background: 'transparent', earthSurface: '{earth_surface}', tooltip: true }})"
            if use_2d else
            f"new GeoSphere('#map', {{ earthSurface: '{earth_surface}', autoRotate: {str(spin).lower()} }})"
        )
        features_js = ""
        if coastlines:
            features_js += f"map.addFeature('coastlines', {{ color: '{coastline_color}', opacity: 0.9, linewidth: {coastline_width} }});\n"
        if borders:
            features_js += f"map.addFeature('borders', {{ color: '{borders_color}', opacity: 0.8, linewidth: {borders_width} }});\n"
        if cities:
            features_js += f"map.addFeature('cities', {{ color: '{cities_color}', opacity: 0.9 }});\n"

        html = _html_animation(
            title=title, label=label, units=units, n_frames=n_frames,
            bundle_js=bundle_js, b64=b64,
            kind=kind, cmap=cmap, alpha=alpha,
            vmin=vmin, vmax=vmax, levels_js=levels_js,
            interval=interval, map_ctor=map_ctor,
            coastlines_line=features_js,
            cbar_opts=_animation_cbar_opts(cmap, vmin, vmax, label, colorbar),
        )
        p = Path(path)
        p.write_text(html)
        return p


def _animation_cbar_opts(cmap, vmin, vmax, label, overrides=None):
    import json as _json
    opts = {
        "cmap": cmap, "vmin": vmin, "vmax": vmax, "label": label, "ticks": 5,
        **{k: v for k, v in (overrides or {}).items() if v is not None},
    }
    return _json.dumps(opts)


# ── helpers ───────────────────────────────────────────────────────────────────

def _js(v: float | None) -> str:
    """Format a Python float/None as a JS literal (null or number)."""
    return "null" if v is None else repr(float(v))


def _js_levels(levels, kind) -> str:
    if kind not in ("contourf", "contour"):
        return "null"
    if hasattr(levels, "tolist"):
        val = levels.tolist()
    elif isinstance(levels, (list, tuple)):
        val = list(levels)
    else:
        val = levels
    return json.dumps(val)


def _render_geosphere_js(kind, cmap, alpha, vmin, vmax, levels, earth_surface,
                         spin=True, coastlines=False, center=(0, 0),
                         coastline_color="#39FF14", coastline_width=2.0,
                         borders=False, borders_color="#000000", borders_width=0.8,
                         cities=False, cities_color="#ffffff") -> str:
    """JS snippet that creates a GeoSphere and plots a field."""
    levels_js = _js_levels(levels, kind)
    coastlines_js = (
        f"map.addFeature('coastlines', {{ color: '{coastline_color}', opacity: 0.9, linewidth: {coastline_width} }});"
        if coastlines else "")
    borders_js = (
        f"map.addFeature('borders', {{ color: '{borders_color}', opacity: 0.8, linewidth: {borders_width} }});"
        if borders else "")
    cities_js = (
        f"map.addFeature('cities', {{ color: '{cities_color}', opacity: 0.9 }});"
        if cities else "")
    return f"""
const map = new GeoSphere('#map', {{ earthSurface: '{earth_surface}', autoRotate: {str(spin).lower()} }});
map.setPointOfView({{ lat: {center[1]}, lng: {center[0]}, altitude: 2.5 }});
const opts = {{
  cmap:   '{cmap}',
  alpha:  {alpha},
  vmin:   {_js(vmin)},
  vmax:   {_js(vmax)},
  levels: {levels_js},
}};
map.{kind}(payload.lons, payload.lats, payload.field, opts);
{coastlines_js}
{borders_js}
{cities_js}
"""


def _render_geomap_js(kind, cmap, alpha, vmin, vmax, levels,
                      projection, coastlines, center, units,
                      extent=None, earth_surface="satellite",
                      coastline_color="#39FF14", coastline_width=2.0,
                      borders=False, borders_color="#000000", borders_width=0.8,
                      cities=False, cities_color="#ffffff") -> str:
    """JS snippet that creates a GeoMap (2D projection) and plots a field."""
    levels_js = _js_levels(levels, kind)
    center_js = f"[{center[0]}, {center[1]}]"
    extent_js = (f", extent: [{extent[0]}, {extent[1]}, {extent[2]}, {extent[3]}]"
                 if extent else "")
    coastlines_js = (
        f"map.addFeature('coastlines', {{ color: '{coastline_color}', opacity: 0.9, linewidth: {coastline_width} }});"
        if coastlines else "")
    borders_js = (
        f"map.addFeature('borders', {{ color: '{borders_color}', opacity: 0.8, linewidth: {borders_width} }});"
        if borders else "")
    cities_js = (
        f"map.addFeature('cities', {{ color: '{cities_color}', opacity: 0.9 }});"
        if cities else "")
    return f"""
const map = new GeoMap('#map', {{
  projection: '{projection}',
  center:     {center_js}{extent_js},
  background: 'transparent',
  earthSurface: '{earth_surface}',
  tooltip:    true,
}});
const opts = {{
  cmap:   '{cmap}',
  alpha:  {alpha},
  vmin:   {_js(vmin)},
  vmax:   {_js(vmax)},
  levels: {levels_js},
  name:   payload.name,
  units:  payload.units,
}};
map.{kind}(payload.lons, payload.lats, payload.field, opts);
{coastlines_js}
{borders_js}
{cities_js}
"""


def _load_terraplot_bundle(bundle_path: str | Path | None) -> str:
    """
    Read the terraplot ESM bundle and transform it for inline use:
      - strip the `export { ... }` block
      - re-expose all public names as plain `const` declarations

    Search order for bundle_path=None:
      1. TERRAPLOT_BUNDLE env var
      2. sibling repo:  <this_file>/../../.. / terraplot/dist/terraplot.js
    """
    if bundle_path is None:
        bundle_path = os.environ.get("TERRAPLOT_BUNDLE")

    if bundle_path is None:
        # Sibling repo layout: /home/user/pyterraplot  &  /home/user/terraplot
        candidate = Path(__file__).resolve().parent.parent.parent / "terraplot" / "dist" / "terraplot.js"
        if candidate.exists():
            bundle_path = candidate

    if bundle_path is None:
        raise FileNotFoundError(
            "Cannot find terraplot bundle. Pass terraplot_bundle='/path/to/terraplot/dist/terraplot.js' "
            "or set the TERRAPLOT_BUNDLE environment variable."
        )

    js = Path(bundle_path).read_text()

    # Find the export block at the end: export { a as B, c as D, ... };
    m = re.search(r'export\s*\{([^}]+)\}\s*;?\s*$', js, re.DOTALL)
    if not m:
        return js  # no export block — return as-is

    export_block = m.group(1)
    aliases: list[str] = []
    for entry in export_block.split(','):
        entry = entry.strip()
        if not entry:
            continue
        if ' as ' in entry:
            min_name, pub_name = entry.split(' as ', 1)
            aliases.append(f"const {pub_name.strip()} = {min_name.strip()};")
        else:
            aliases.append(f"const {entry} = {entry};")

    # Strip the export block and append const aliases
    return js[:m.start()].rstrip() + '\n' + '\n'.join(aliases)


# ── HTML builders ─────────────────────────────────────────────────────────────

_SHARED_CSS = """
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #090912; color: #e0e0e0; font-family: system-ui, sans-serif; }
  #map { width: 100vw; height: 100vh; position: relative; }
  #label {
    position: fixed; top: .8rem; left: 50%; transform: translateX(-50%);
    background: rgba(0,0,0,.6); padding: .35rem .9rem; border-radius: 6px;
    font-size: .82rem; white-space: nowrap; pointer-events: none;
    border: 1px solid rgba(255,255,255,.12);
  }
  #colorbar { /* positioned & styled by the Colorbar widget itself */ }
"""

_IMPORTMAP = """\
<script type="importmap">
{"imports": {
  "three": "https://cdn.jsdelivr.net/npm/three@0.184.0/build/three.module.js",
  "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.184.0/examples/jsm/"
}}
</script>"""

_COLORBAR_JS = """\
// ── Colorbar (terraplot widget — orientation/panel/scale configurable) ────
(function drawColorbar(field, cbarOpts) {
  let lo = cbarOpts.vmin, hi = cbarOpts.vmax;
  if (lo == null || hi == null) {
    lo = Infinity; hi = -Infinity;
    // Handle both flat TypedArray (binary mode) and nested 2D array (JSON mode)
    const iterable = ArrayBuffer.isView(field) ? field : field.flat(Infinity);
    for (const v of iterable) {
      if (v != null && isFinite(v)) { if (v < lo) lo = v; if (v > hi) hi = v; }
    }
    cbarOpts.vmin = lo; cbarOpts.vmax = hi;
  }
  const host = document.getElementById('colorbar');
  host.replaceChildren();
  new Colorbar(host, cbarOpts);
})(payload.field, CBAR_OPTS);"""


def _cbar_opts_js(cbar_opts: dict, cmap: str, vmin, vmax, label: str) -> str:
    """Serialize the widget options for the HTML template's CBAR_OPTS slot.
    Values explicitly given win; sensible defaults fill the rest."""
    import json as _json
    opts = {
        "cmap": cmap,
        "vmin": vmin,
        "vmax": vmax,
        "label": label,
        "ticks": 5,
        **{k: v for k, v in cbar_opts.items() if v is not None},
    }
    return _json.dumps(opts)


def _html_single(*, title, label, units, cbar_id, bundle_js, payload_js,
                 map_init, cmap, vmin, vmax, height_px=None,
                 cbar_opts=None) -> str:
    colorbar_js = (
        _COLORBAR_JS
        .replace("CBAR_OPTS", _cbar_opts_js(cbar_opts or {}, cmap, vmin, vmax, label))
    )
    map_height = f"{height_px}px" if height_px else "100vh"
    vertical = (cbar_opts or {}).get("orientation") == "vertical"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>{title}</title>
<style>
{_SHARED_CSS}
  #map {{ width: 100vw; height: {map_height}; }}
</style>
</head>
<body>
<div id="map"></div>
<div id="label">{label}</div>
<div id="colorbar"></div>
{_IMPORTMAP}
<script type="module">
{bundle_js}

const payload = {payload_js};
{map_init}
{colorbar_js}
</script>
</body>
</html>"""


def _html_animation(*, title, label, units, n_frames, bundle_js, b64,
                    kind, cmap, alpha, vmin, vmax, levels_js, interval,
                    map_ctor, coastlines_line, cbar_opts=None) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>{title}</title>
<style>
{_SHARED_CSS}
  #cbar {{ /* styled by the Colorbar widget */ }}
  #controls {{
    position: fixed; bottom: 5rem; left: 50%; transform: translateX(-50%);
    display: flex; align-items: center; gap: .6rem;
    background: rgba(0,0,0,.65); padding: .4rem .8rem; border-radius: 8px;
    border: 1px solid rgba(255,255,255,.12); user-select: none;
  }}
  #play-btn {{
    background: rgba(255,255,255,.12); border: 1px solid rgba(255,255,255,.2);
    color: #e2e8f0; border-radius: 4px; padding: 2px 10px; cursor: pointer;
    font-size: .78rem; transition: background .15s;
  }}
  #play-btn:hover {{ background: rgba(255,255,255,.22); }}
  #scrubber {{
    width: 180px; accent-color: #60a5fa; cursor: pointer;
  }}
  #frame-label {{
    font-size: .72rem; color: #94a3b8; min-width: 80px; text-align: center;
  }}
</style>
</head>
<body>
<div id="map"></div>
<div id="label">{label}</div>
<div id="colorbar"></div>
<div id="controls">
  <button id="play-btn">⏸ Pause</button>
  <input id="scrubber" type="range" min="0" max="{n_frames - 1}" value="0" step="1"/>
  <span id="frame-label">Frame 0</span>
</div>
{_IMPORTMAP}
<script type="module">
{bundle_js}

const data = await unpackFrames("{b64}");

// Colorbar (terraplot widget)
(() => {{
  const o = {cbar_opts};
  if (o.vmin == null || o.vmax == null) {{
    let lo = Infinity, hi = -Infinity;
    for (const fr of data.frames) for (const v of fr.field)
      if (isFinite(v)) {{ if (v < lo) lo = v; if (v > hi) hi = v; }}
    o.vmin = lo; o.vmax = hi;
  }}
  new Colorbar(document.getElementById('colorbar'), o);
}})();

const map = {map_ctor};
{coastlines_line}
const opts = {{
  cmap:   '{cmap}',
  alpha:  {alpha},
  vmin:   {_js(vmin)},
  vmax:   {_js(vmax)},
  levels: {levels_js},
}};

const scrubber   = document.getElementById('scrubber');
const playBtn    = document.getElementById('play-btn');
const frameLabel = document.getElementById('frame-label');

const anim = map.animate(data, {{
  type:         '{kind}',
  interval:     {interval},
  layerOptions: opts,
  onFrame: (i, f) => {{
    scrubber.value    = i;
    frameLabel.textContent = f.coord_value || `Frame ${{i}}`;
  }},
}});

// Sync scrubber → animation
scrubber.addEventListener('input', () => {{
  anim.pause();
  playBtn.textContent = '▶ Play';
  anim.seek(parseInt(scrubber.value, 10));
}});

// Play/pause button
playBtn.addEventListener('click', () => {{
  if (playBtn.textContent.startsWith('▶')) {{
    anim.play();
    playBtn.textContent = '⏸ Pause';
  }} else {{
    anim.pause();
    playBtn.textContent = '▶ Play';
  }}
}});

// ── Colorbar (handled above via the widget) ───────────────────────────────
</script>
</body>
</html>"""
