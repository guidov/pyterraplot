"""
Figure — matplotlib-style multi-panel layout for terraplot HTML exports.

``subplots()`` mirrors ``matplotlib.pyplot.subplots``, including the
``subplot_kw={"projection": ...}`` spelling cartopy users reach for::

    import pyterraplot as tp
    import pyterraplot.crs as ccrs

    fig, axes = tp.subplots(1, 2, figsize=(14, 6),
                            subplot_kw={"projection": ccrs.Robinson()})
    for ax, (name, da) in zip(axes.flat, fields.items()):
        ax.contourf(da, levels=14, cmap="RdBu_r", vmin=-30, vmax=30)
        ax.coastlines()
        ax.set_title(name)
    fig.suptitle("Ensemble mean vs reanalysis")
    fig.colorbar(label="2 m temperature [°C]", vmin=-30, vmax=30, cmap="RdBu_r")
    fig.savefig("panels.html")

Every panel is an independent terraplot map with its own projection, extent and
layers. Panels share one payload registry, so a DataArray drawn in several
panels is embedded once.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .axes import Axes, _LEGEND_CSS, _payload_decls, _esc
from .accessor import _IMPORTMAP, _SHARED_CSS, _load_terraplot_bundle

__all__ = ["Figure", "figure", "subplots"]

#: Pixels per inch used to turn a matplotlib-style ``figsize`` into a canvas.
DEFAULT_DPI = 100


class Figure:
    """A grid of :class:`~pyterraplot.Axes` panels rendered into one HTML page."""

    def __init__(self, figsize: tuple[float, float] | None = None, *,
                 dpi: int = DEFAULT_DPI, nrows: int = 1, ncols: int = 1,
                 background: str = "#090912", title: str | None = None):
        self.figsize = figsize
        self.dpi = dpi
        self.nrows = nrows
        self.ncols = ncols
        self.background = background
        self._suptitle = title
        self._axes: list[Axes] = []
        self._positions: list[tuple[int, int, int, int]] = []   # row, col, rowspan, colspan
        # One registry for the whole figure so a field shared between panels
        # is packed into the page a single time.
        self._payloads: dict[int, dict[str, Any]] = {}
        self._cbar: dict[str, Any] | None = None

    # ── panel construction ───────────────────────────────────────────────────

    def add_subplot(self, *args, projection=None, rowspan: int = 1,
                    colspan: int = 1, **kw) -> Axes:
        """Add a panel, matplotlib-style.

        Accepts ``add_subplot(nrows, ncols, index)`` or the three-digit
        ``add_subplot(221)`` shorthand; ``index`` is 1-based and runs across
        rows. With no positional arguments the panel is appended to the next
        free cell of the current grid.
        """
        if len(args) == 1 and isinstance(args[0], int) and args[0] >= 100:
            spec = args[0]
            nrows, ncols, index = spec // 100, (spec // 10) % 10, spec % 10
        elif len(args) == 3:
            nrows, ncols, index = (int(a) for a in args)
        elif not args:
            nrows, ncols = self.nrows, self.ncols
            index = len(self._axes) + 1
        else:
            raise TypeError(
                "add_subplot takes (nrows, ncols, index), a three-digit code "
                f"like 221, or no positional arguments; got {args!r}"
            )

        self.nrows = max(self.nrows, nrows)
        self.ncols = max(self.ncols, ncols)
        row, col = divmod(index - 1, ncols)

        ax = Axes(projection, _payloads=self._payloads, **kw)
        self._axes.append(ax)
        self._positions.append((row, col, rowspan, colspan))
        return ax

    @property
    def axes(self) -> list[Axes]:
        """The panels, in creation order."""
        return list(self._axes)

    # ── figure-level decoration ──────────────────────────────────────────────

    def suptitle(self, text: str) -> "Figure":
        """Set the figure-wide title."""
        self._suptitle = text
        return self

    def colorbar(self, ax: Axes | None = None, *, cmap: str | None = None,
                 vmin: float | None = None, vmax: float | None = None,
                 label: str | None = None, **opts) -> "Figure":
        """Add one colorbar for the whole figure, below the panel grid.

        Without explicit ``cmap``/``vmin``/``vmax`` the settings are taken from
        the first filled field layer of ``ax`` (or of the first panel that has
        one), so a shared scale set on the panels carries through.
        """
        source = ax or next((a for a in self._axes
                             if any(c[0] == "field" for c in a._calls)), None)
        cfg: dict[str, Any] = dict(opts)
        if source is not None:
            field_opts = next(
                (c[3] for c in source._calls
                 if c[0] == "field" and c[1] in ("pcolormesh", "contourf")),
                {},
            )
            cfg.setdefault("cmap", cmap or field_opts.get("cmap", "viridis"))
            cfg.setdefault("vmin", vmin if vmin is not None else field_opts.get("vmin"))
            cfg.setdefault("vmax", vmax if vmax is not None else field_opts.get("vmax"))
        else:
            cfg.setdefault("cmap", cmap or "viridis")
            cfg.setdefault("vmin", vmin)
            cfg.setdefault("vmax", vmax)
        if label is None and source is not None:
            first = next(iter(source._payloads.values()), None)
            if first:
                p = first["payload"]
                ln = p.get("long_name") or p.get("name") or ""
                u = p.get("units", "")
                label = f"{ln} [{u}]" if u else ln
        if label:
            cfg["label"] = label
        cfg.setdefault("ticks", 5)
        self._cbar = cfg
        return self

    # ── rendering ────────────────────────────────────────────────────────────

    def to_html(self, path: str | Path, *, title: str | None = None,
                binary: bool = True,
                terraplot_bundle: str | Path | None = None) -> Path:
        html = self._render_html(title=title, binary=binary,
                                 terraplot_bundle=terraplot_bundle)
        p = Path(path)
        p.write_text(html)
        return p

    def savefig(self, path: str | Path, **kw) -> Path:
        """matplotlib spelling of :meth:`to_html`."""
        return self.to_html(path, **kw)

    def _repr_html_(self) -> str:
        try:
            html = self._render_html(title=self._suptitle, height_px=460)
            srcdoc = html.replace("&", "&amp;").replace('"', "&quot;")
            return (
                f'<iframe srcdoc="{srcdoc}" width="100%" height="480" '
                f'style="border:1px solid rgba(255,255,255,.1); border-radius:6px;"></iframe>'
            )
        except Exception as e:  # pragma: no cover — Jupyter must never crash
            return f"<pre>pyterraplot Figure render failed: {e}</pre>"

    def _render_html(self, *, title: str | None = None, binary: bool = True,
                     terraplot_bundle=None, height_px: int | None = None) -> str:
        if not self._axes:
            raise ValueError(
                "Figure has no panels — call fig.add_subplot(...) or use "
                "pyterraplot.subplots(nrows, ncols)."
            )
        bundle_js = _load_terraplot_bundle(terraplot_bundle)
        page_title = title or self._suptitle or "terraplot"

        panels, ctors, layer_lines, anim_lines = [], [], [], []
        for i, (ax, (row, col, rspan, cspan)) in enumerate(
                zip(self._axes, self._positions)):
            map_id = f"map{i}"
            style = (f"grid-row: {row + 1} / span {rspan}; "
                     f"grid-column: {col + 1} / span {cspan};")
            panel_title = (f'<div class="tp-panel-title">{_esc(ax._title)}</div>'
                           if ax._title else "")
            panels.append(
                f'<div class="tp-panel" style="{style}">{panel_title}'
                f'<div class="tp-map" id="{map_id}"></div>'
                f'<div class="tp-panel-cbar" id="cbar-{map_id}"></div>'
                f'{ax._legend_html()}</div>'
            )
            ctors.append(ax._ctor_js(map_id, f"#{map_id}"))
            lines, _ = ax._layers_js(map_id, cbar_host=f"#cbar-{map_id}")
            layer_lines.extend(lines)
            anim = ax._anim_js(map_id)
            if anim:
                anim_lines.append(anim)

        decls = _payload_decls(self._payloads, binary)

        cbar_js = ""
        if self._cbar is not None:
            cvar = next(iter(self._payloads.values()), None)
            field_ref = f"{cvar['var']}.field" if cvar else "[]"
            cbar_js = f"""(() => {{
  const cbarOpts = {json.dumps(self._cbar)};
  let lo = cbarOpts.vmin, hi = cbarOpts.vmax;
  const f = {field_ref};
  if (lo == null || hi == null) {{
    lo = Infinity; hi = -Infinity;
    const iterable = ArrayBuffer.isView(f) ? f : f.flat(Infinity);
    for (const v of iterable) {{
      if (v != null && isFinite(v)) {{ if (v < lo) lo = v; if (v > hi) hi = v; }}
    }}
    cbarOpts.vmin = lo; cbarOpts.vmax = hi;
  }}
  new Colorbar(document.getElementById('fig-cbar'), cbarOpts);
}})();"""

        # Figure geometry: an explicit figsize pins a canvas, otherwise the grid
        # fills the viewport.
        if self.figsize:
            w = int(self.figsize[0] * self.dpi)
            h = int(height_px or self.figsize[1] * self.dpi)
            frame_css = (f"width: min(100vw, {w}px); height: {h}px; "
                         f"margin: 0 auto;")
        else:
            frame_css = f"width: 100vw; height: {height_px or 0}px;" if height_px \
                else "width: 100vw; height: 100vh;"

        suptitle_html = (f'<div id="suptitle">{_esc(self._suptitle)}</div>'
                         if self._suptitle else "")
        cbar_html = '<div id="fig-cbar"></div>' if self._cbar is not None else ""

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>{_esc(page_title)}</title>
<style>
{_SHARED_CSS}
{_LEGEND_CSS}
  body {{ background: {self.background}; display: flex; flex-direction: column;
         align-items: center; }}
  #suptitle {{
    padding: .7rem 1rem .2rem; font-size: 1rem; font-weight: 600;
    color: #e2e8f0; text-align: center;
  }}
  #figure {{
    {frame_css}
    display: grid;
    grid-template-rows: repeat({self.nrows}, 1fr);
    grid-template-columns: repeat({self.ncols}, 1fr);
    gap: 6px; padding: 6px; min-height: 0;
  }}
  .tp-panel {{ position: relative; min-width: 0; min-height: 0;
               border: 1px solid rgba(255,255,255,.08); border-radius: 4px;
               overflow: hidden; }}
  .tp-map {{ position: absolute; inset: 0; }}
  .tp-panel-title {{
    position: absolute; top: .4rem; left: 50%; transform: translateX(-50%);
    z-index: 110; background: rgba(0,0,0,.55); padding: .2rem .6rem;
    border-radius: 4px; font-size: .74rem; color: #e2e8f0;
    white-space: nowrap; pointer-events: none;
  }}
  .tp-panel-cbar {{
    position: absolute; bottom: .4rem; left: 50%; transform: translateX(-50%);
    z-index: 110;
  }}
  #fig-cbar {{ padding: .4rem 0 1rem; }}
</style>
</head>
<body>
{suptitle_html}
<div id="figure">
{chr(10).join(panels)}
</div>
{cbar_html}
{_IMPORTMAP}
<script type="module">
{bundle_js}

{chr(10).join(decls)}
{chr(10).join(ctors)}
{chr(10).join(layer_lines)}
{chr(10).join(anim_lines)}
{cbar_js}
</script>
</body>
</html>"""


# ── module-level constructors ────────────────────────────────────────────────

def figure(figsize: tuple[float, float] | None = None, *,
           dpi: int = DEFAULT_DPI, background: str = "#090912",
           title: str | None = None) -> Figure:
    """Create an empty :class:`Figure`, matplotlib-style."""
    return Figure(figsize, dpi=dpi, background=background, title=title)


def subplots(nrows: int = 1, ncols: int = 1, *,
             figsize: tuple[float, float] | None = None,
             dpi: int = DEFAULT_DPI,
             projection=None,
             subplot_kw: dict[str, Any] | None = None,
             squeeze: bool = True,
             background: str = "#090912",
             title: str | None = None):
    """Create a figure and a grid of panels.

    Returns ``(fig, axes)``. With ``squeeze=True`` (the default) a 1×1 grid
    returns a bare :class:`~pyterraplot.Axes` and a single row or column
    returns a 1D array, matching matplotlib.

    ``projection`` applies to every panel; ``subplot_kw`` takes the same
    keywords :class:`~pyterraplot.Axes` does, including a per-call
    ``projection``, and is the spelling cartopy users will recognise.
    """
    kw = dict(subplot_kw or {})
    proj = kw.pop("projection", projection)

    fig = Figure(figsize, dpi=dpi, nrows=nrows, ncols=ncols,
                 background=background, title=title)
    grid = np.empty((nrows, ncols), dtype=object)
    for r in range(nrows):
        for c in range(ncols):
            grid[r, c] = fig.add_subplot(nrows, ncols, r * ncols + c + 1,
                                         projection=proj, **kw)

    if squeeze:
        if nrows == 1 and ncols == 1:
            return fig, grid[0, 0]
        if nrows == 1 or ncols == 1:
            return fig, grid.ravel()
    return fig, grid
