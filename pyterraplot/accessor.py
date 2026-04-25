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

    # ── Self-contained HTML export ────────────────────────────────────────────

    def to_html(
        self,
        path: str | Path,
        *,
        title: str = "terraplot",
        cmap: str = "viridis",
        alpha: float = 0.7,
        lon_dim: str | None = None,
        lat_dim: str | None = None,
        wrap_lon: bool = True,
        terraplot_bundle: str | Path | None = None,
    ) -> Path:
        """
        Export a self-contained HTML file that renders the field in a 3D globe.

        The JSON payload and the terraplot bundle are both embedded inline —
        no server required. Open the output file directly in a browser.

        Parameters
        ----------
        path               : output file path (.html)
        title              : page title
        cmap               : colormap name (any terraplot Colormaps key)
        alpha              : field opacity (0-1)
        terraplot_bundle   : path to terraplot dist/terraplot.js; auto-detected
                             if None (looks for sibling repo ../terraplot)
        """
        payload = self.to_dict(lon_dim=lon_dim, lat_dim=lat_dim, wrap_lon=wrap_lon)
        payload_json = json.dumps(payload)
        long_name = payload.get("long_name") or payload.get("name") or title
        units = payload.get("units", "")
        label = f"{long_name} [{units}]" if units else long_name

        bundle_js = _load_terraplot_bundle(terraplot_bundle)

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>{title}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: #090912; color: #e0e0e0; font-family: system-ui, sans-serif; }}
  #globe {{ width: 100vw; height: 100vh; }}
  #label {{
    position: fixed; bottom: 1rem; left: 50%; transform: translateX(-50%);
    background: rgba(0,0,0,.55); padding: .35rem .8rem; border-radius: 6px;
    font-size: .85rem; pointer-events: none;
  }}
</style>
</head>
<body>
<div id="globe"></div>
<div id="label">{label}</div>
<script type="importmap">
{{"imports": {{
  "three": "https://cdn.jsdelivr.net/npm/three@0.184.0/build/three.module.js",
  "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.184.0/examples/jsm/"
}}}}
</script>
<script type="module">
{bundle_js}

const payload = {payload_json};

const globe = new GeoSphere('#globe');
globe.pcolormesh(payload.lons, payload.lats, payload.field, {{
  cmap:  '{cmap}',
  alpha: {alpha},
}});
</script>
</body>
</html>"""

        p = Path(path)
        p.write_text(html)
        return p


# ── helpers ───────────────────────────────────────────────────────────────────

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
