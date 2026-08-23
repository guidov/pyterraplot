"""
Axes — matplotlib/cartopy-style compositional API for terraplot HTML exports.

Instead of exporting one field with one render style, an Axes collects a
sequence of layer primitives (filled contours, isolines, coastlines, …) and
renders them stacked in call order — the same mental model as

    fig, ax = plt.subplots()
    ax.contourf(...)
    ax.contour(...)
    ax.coastlines()

Example
-------
>>> import pyterraplot as tp
>>> ax = tp.Axes(spin=False, earth_surface="none")
>>> ax.contourf(air, levels=14, cmap="viridis", vmin=-30, vmax=30)
>>> ax.contour(air, levels=14, color="black", linewidth=1.5)   # outline the patches
>>> ax.coastlines(color="#39FF14")
>>> ax.to_html("plot.html")

Passing the same DataArray to several primitives embeds the payload only once.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import xarray as xr

from .binary import pack_field, pack_frames
from .accessor import (
    _COLORBAR_JS, _IMPORTMAP, _SHARED_CSS, _load_terraplot_bundle,
)

_FIELD_KINDS = ("pcolormesh", "contourf", "contour")


class Axes:
    """Composable plot container. See module docstring for usage."""

    def __init__(
        self,
        *,
        globe: bool = True,
        projection: str | None = None,
        center: tuple[float, float] = (0, 0),
        extent: tuple[float, float, float, float] | None = None,
        spin: bool = True,
        earth_surface: str = "satellite",
        background: str | None = None,
        height_px: int | None = None,
    ):
        if not globe and projection is None:
            raise ValueError("Set globe=False only together with projection=...")
        self.globe = globe
        self.projection = projection
        self.center = center
        self.extent = extent
        self.spin = spin
        self.earth_surface = earth_surface
        self.background = background
        self.height_px = height_px

        # id(da) -> {"da": da, "var": "P0", "payload": dict} — dedupes payloads
        # shared between primitives (e.g. contourf + contour of one field).
        self._payloads: dict[int, dict[str, Any]] = {}
        self._calls: list[tuple[str, ...]] = []   # sequential JS-generating ops
        self._title: str | None = None
        self._cbar_opts: dict[str, Any] | None = None
        self._anim: dict[str, Any] | None = None

    # ── layer primitives ──────────────────────────────────────────────────────

    def pcolormesh(self, da: xr.DataArray, **opts) -> "Axes":
        """Smooth gradient fill. opts: cmap, alpha, vmin, vmax."""
        return self._field("pcolormesh", da, opts)

    def contourf(self, da: xr.DataArray, **opts) -> "Axes":
        """Banded contour fill. opts: cmap, alpha, vmin, vmax, levels."""
        return self._field("contourf", da, opts)

    def contour(self, da: xr.DataArray, **opts) -> "Axes":
        """Unfilled isolines. opts: levels, cmap or color, alpha, vmin, vmax,
        linewidth (px, >1 = fat line), zorder."""
        return self._field("contour", da, opts)

    def quiver(self, u: xr.DataArray, v: xr.DataArray, **opts) -> "Axes":
        """Vector arrows (2D projections only). opts: color, scale, width."""
        if self.globe:
            raise NotImplementedError("quiver is currently 2D-projection only")
        for arr in (u, v):
            if not hasattr(arr, "tp"):
                raise TypeError("quiver(u, v) expects xarray DataArrays")
        self._register(u)
        self._register(v)
        self._calls.append(("quiver", id(u), id(v), opts))
        return self

    def coastlines(self, color: str = "#39FF14", width: float = 2.0,
                   opacity: float = 0.9) -> "Axes":
        self._calls.append(("feature", "coastlines", dict(color=color, linewidth=width, opacity=opacity)))
        return self

    def borders(self, color: str = "rgba(180,180,210,0.5)", width: float = 1.0,
                opacity: float = 1.0) -> "Axes":
        self._calls.append(("feature", "borders", dict(color=color, linewidth=width, opacity=opacity)))
        return self

    def marker(self, lat: float, lon: float, label: str | None = None,
               color: str = "#fbbf24", size: float = 5) -> "Axes":
        """Point marker (2D projections only)."""
        if self.globe:
            raise NotImplementedError("marker is currently 2D-projection only")
        self._calls.append(("marker", dict(lat=lat, lon=lon, label=label, color=color, size=size)))
        return self

    def title(self, text: str) -> "Axes":
        self._title = text
        return self

    def colorbar(self, **opts) -> "Axes":
        """Override the auto colorbar: cmap, vmin, vmax, label."""
        self._cbar_opts = opts
        return self

    def animate(self, da: xr.DataArray, dim: str, *, kind: str = "pcolormesh",
                interval: int = 800, **layer_opts) -> "Axes":
        """Animate a multi-step DataArray; static layers stay on top."""
        if self._anim is not None:
            raise ValueError("Only one animate() per Axes")
        self._anim = dict(
            compact=da.tp.frames_compact(dim=dim),
            kind=kind, interval=interval, layer_opts=layer_opts,
        )
        return self

    # ── rendering ─────────────────────────────────────────────────────────────

    def to_html(self, path: str | Path, *, title: str = "terraplot",
                binary: bool = True, terraplot_bundle: str | Path | None = None) -> Path:
        html = self._render_html(title=title, binary=binary,
                                 terraplot_bundle=terraplot_bundle)
        p = Path(path)
        p.write_text(html)
        return p

    def _repr_html_(self) -> str:
        try:
            html = self._render_html(title=self._title or "terraplot", height_px=420)
            srcdoc = html.replace("&", "&amp;").replace('"', "&quot;")
            return (
                f'<iframe srcdoc="{srcdoc}" width="100%" height="440" '
                f'style="border:1px solid rgba(255,255,255,.1); border-radius:6px;"></iframe>'
            )
        except Exception as e:  # pragma: no cover — Jupyter must never crash
            return f"<pre>pyterraplot Axes render failed: {e}</pre>"

    # ── internals ─────────────────────────────────────────────────────────────

    def _register(self, da: xr.DataArray) -> str:
        key = id(da)
        if key not in self._payloads:
            self._payloads[key] = {
                "da": da,                                   # pin the object: keeps id() valid
                "var": f"P{len(self._payloads)}",
                "payload": da.tp.to_dict(),
            }
        return self._payloads[key]["var"]

    def _field(self, kind: str, da: xr.DataArray, opts: dict) -> "Axes":
        if kind not in _FIELD_KINDS:
            raise ValueError(f"unknown field kind {kind!r}")
        if not hasattr(da, "tp"):
            raise TypeError(f"{kind}() expects an xarray DataArray")
        var = self._register(da)
        self._calls.append(("field", kind, var, opts))
        return self

    def _render_html(self, *, title="terraplot", binary=True,
                     terraplot_bundle=None, height_px=None) -> str:
        bundle_js = _load_terraplot_bundle(terraplot_bundle)
        use_2d = not self.globe

        # ── map constructor ──
        if use_2d:
            ctor = (f"new GeoMap('#map', {{ projection: '{self.projection}', "
                    f"center: [{self.center[0]}, {self.center[1]}]"
                    + (f", extent: [{self.extent[0]}, {self.extent[1]}, {self.extent[2]}, {self.extent[3]}]" if self.extent else "")
                    + f", background: '{self.background or 'transparent'}', earthSurface: '{self.earth_surface}', tooltip: true }})")
        else:
            ctor = (f"new GeoSphere('#map', {{ earthSurface: '{self.earth_surface}', "
                    f"autoRotate: {str(self.spin).lower()} }});\n"
                    f"map.setPointOfView({{ lat: {self.center[1]}, lng: {self.center[0]}, altitude: 2.5 }});")

        # ── payload declarations ──
        decls = []
        for entry in self._payloads.values():
            if binary:
                decls.append(f'const {entry["var"]} = await unpackField("{pack_field(entry["payload"])}");')
            else:
                decls.append(f'const {entry["var"]} = {json.dumps(entry["payload"])};')

        # ── sequential layer calls ──
        lines = []
        first_field_opts = None
        for call in self._calls:
            op = call[0]
            if op == "field":
                _, kind, var, opts = call
                js_opts = dict(opts)
                if "levels" in js_opts and js_opts.get("cmap") is None and kind == "contour":
                    pass  # solid-color isolines: levels pass through as-is
                lines.append(f"map.{kind}({var}.lons, {var}.lats, {var}.field, {json.dumps(js_opts)});")
                if first_field_opts is None and kind in ("pcolormesh", "contourf"):
                    first_field_opts = js_opts
            elif op == "feature":
                _, name, opts = call
                lines.append(f"map.addFeature('{name}', {json.dumps(opts)});")
            elif op == "quiver":
                _, uid, vid, opts = call
                u = self._payloads[uid]["var"]; v = self._payloads[vid]["var"]
                lines.append(f"map.quiver({u}.lons, {u}.lats, {u}.field, {v}.field, {json.dumps(opts)});")
            elif op == "marker":
                lines.append(f"map.marker({json.dumps(call[1]['lat'])}, {json.dumps(call[1]['lon'])}, {json.dumps(call[1])});")

        # ── animation ──
        anim_js = ""
        if self._anim:
            a = self._anim
            b64 = pack_frames(a["compact"])
            lopts = dict(a["layer_opts"])
            anim_js = (f'const F0 = await unpackFrames("{b64}");\n'
                       f"map.animate(F0, {{ type: '{a['kind']}', interval: {a['interval']}, "
                       f"layerOptions: {json.dumps(lopts)} }});")

        # ── title / label ──
        first = next(iter(self._payloads.values()), None)
        if first:
            payload = first["payload"]
            long_name = payload.get("long_name") or payload.get("name") or ""
            units = payload.get("units", "")
            label = f"{long_name} [{units}]" if units else long_name
        else:
            label = ""
        if self._title:
            title = self._title
            label = f"{title} — {label}" if label else title

        # ── colorbar ──
        cbar = self._cbar_opts or {}
        cbar_src = dict(cbar)
        if first_field_opts:
            cbar_src.setdefault("cmap", first_field_opts.get("cmap", "viridis"))
            cbar_src.setdefault("vmin", first_field_opts.get("vmin"))
            cbar_src.setdefault("vmax", first_field_opts.get("vmax"))
        cvar = next(iter(self._payloads.values()), None)
        colorbar_js = (
            _COLORBAR_JS
            .replace("'cbar'", "'cbar'")
            .replace("CMAP", json.dumps(cbar_src.get("cmap", "viridis")))
            .replace("VMIN", json.dumps(cbar_src.get("vmin")))
            .replace("VMAX", json.dumps(cbar_src.get("vmax")))
        )
        # _COLORBAR_JS takes the field array; pass the first payload's
        colorbar_js = colorbar_js.replace(
            "})(payload.field,",
            f"}})({cvar['var']}.field," if cvar else "})([],",
        )

        map_height = f"{height_px or self.height_px}px" if (height_px or self.height_px) else "100vh"

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>{title}</title>
<style>
{_SHARED_CSS}
  #map {{ width: 100vw; height: {map_height}; }}
  #cbar {{ width: 220px; height: 12px; border-radius: 3px;
           border: 1px solid rgba(255,255,255,.18); }}
</style>
</head>
<body>
<div id="map"></div>
<div id="label">{label}</div>
<div id="colorbar">
  <canvas id="cbar" width="220" height="12"></canvas>
  <div id="cbar-ticks"></div>
  <div id="cbar-units">{cbar_src.get('label', '')}</div>
</div>
{_IMPORTMAP}
<script type="module">
{bundle_js}

const map = {ctor};
{chr(10).join(decls)}
{chr(10).join(lines)}
{anim_js}
{colorbar_js}
</script>
</body>
</html>"""
