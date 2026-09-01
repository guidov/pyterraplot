"""
Axes — matplotlib/cartopy-style compositional API for terraplot HTML exports.

Instead of exporting one field with one render style, an Axes collects a
sequence of layer primitives (filled contours, isolines, coastlines, …) and
renders them stacked in call order — the same mental model as matplotlib::

    import pyterraplot as tp
    import pyterraplot.crs as ccrs
    import pyterraplot.feature as cfeature

    ax = tp.Axes(projection=ccrs.Robinson(central_longitude=-100))
    ax.contourf(air, levels=14, cmap="RdBu_r")
    ax.contour(air, levels=14, color="black", linewidth=1.5)
    ax.add_feature(cfeature.LAND)
    ax.coastlines(resolution="50m")
    ax.gridlines(draw_labels=True)
    ax.set_extent([-141, -52, 41, 84])
    ax.set_title("2 m air temperature")
    ax.to_html("plot.html")

Passing a projection object switches to a 2D map; the default (no projection,
or ``ccrs.Globe3D()``) renders the interactive 3D globe.

Layers render stacked in call order, and passing the same DataArray to several
primitives embeds the payload only once.
"""
from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Iterable
from typing import Any, Sequence

import numpy as np
import xarray as xr

from . import crs as _crs
from . import feature as _feature
from . import geodesy
from .binary import pack_field, pack_frames
from .accessor import (
    _COLORBAR_JS, _IMPORTMAP, _SHARED_CSS, _load_terraplot_bundle,
)

_FIELD_KINDS = ("pcolormesh", "contourf", "contour")

# matplotlib single-letter colour shorthands.
_COLOR_CODES = {
    "b": "#1f77b4", "g": "#2ca02c", "r": "#d62728", "c": "#17becf",
    "m": "#e377c2", "y": "#bcbd22", "k": "#000000", "w": "#ffffff",
}
_LINESTYLES = ("--", "-.", ":", "-")
_MARKERS = "o.,^v<>sp*+xdD|_"


class Axes:
    """Composable plot container. See the module docstring for usage.

    Parameters
    ----------
    projection : CRS instance, projection name, or None
        ``None`` (default) or :class:`~pyterraplot.crs.Globe3D` renders the
        interactive 3D globe. Any other :mod:`pyterraplot.crs` projection —
        or a bare terraplot projection name for backwards compatibility —
        renders a flat 2D map.
    globe : bool, optional
        Legacy switch, inferred from ``projection`` when omitted.
    center : (lon, lat), optional
        Map centre; defaults to the projection's central longitude/latitude.
    extent : (lon0, lon1, lat0, lat1), optional
        Initial regional extent, as in cartopy's ``set_extent``.
    spin : bool
        3D globe only — auto-rotate. Default True.
    earth_surface : str
        ``'satellite'`` (default), ``'shaded_relief'`` / ``'stock'``,
        ``'none'`` / ``'outline'``, or an image URL.
    background : str, optional
        Page/map background colour.
    height_px : int, optional
        Fixed map height; defaults to the full viewport.
    tooltip : bool
        2D only — hover readout of lat/lon/value. Default True.
    graticule : bool, optional
        Draw terraplot's default 10° grid. Defaults to True for 2D unless
        :meth:`gridlines` is called, which replaces it.
    """

    def __init__(
        self,
        projection: "_crs.CRS | str | None" = None,
        *,
        globe: bool | None = None,
        center: tuple[float, float] | None = None,
        extent: tuple[float, float, float, float] | None = None,
        spin: bool = True,
        earth_surface: str = "satellite",
        background: str | None = None,
        height_px: int | None = None,
        tooltip: bool = True,
        graticule: bool | None = None,
        _payloads: dict[int, dict[str, Any]] | None = None,
    ):
        self.crs, self.projection, self.globe = self._resolve_projection(projection, globe)

        if center is None and self.crs is not None and not self.crs.is_globe:
            center = self.crs.center
        self.center = center or (0.0, 0.0)

        if extent is None and self.crs is not None:
            extent = getattr(self.crs, "default_extent", None)
        self.extent = extent

        self.spin = spin
        self.earth_surface = earth_surface
        self.background = background
        self.height_px = height_px
        self.tooltip = tooltip
        self.graticule = graticule

        # id(da) -> {"da": da, "var": "P0", "payload": dict} — dedupes payloads
        # shared between primitives (e.g. contourf + contour of one field).
        # A Figure passes its own dict so panels share one registry.
        self._payloads: dict[int, dict[str, Any]] = (
            _payloads if _payloads is not None else {}
        )
        self._calls: list[tuple] = []          # sequential JS-generating ops
        self._title: str | None = None
        self._legend: dict[str, Any] | None = None
        self._legend_entries: list[dict[str, Any]] = []
        self._cbar_opts: dict[str, Any] | None = None
        self._anim: dict[str, Any] | None = None

    # ── projection resolution ────────────────────────────────────────────────

    @staticmethod
    def _resolve_projection(projection, globe):
        """Normalise the projection argument into (CRS|None, name|None, is_globe)."""
        if isinstance(projection, _crs.CRS):
            if projection.is_transform_only:
                raise TypeError(
                    f"{type(projection).__name__}() is a transform, not a map "
                    "projection — pass it as transform=... to plot()/text(), "
                    "not as projection= to Axes()."
                )
            is_globe = projection.is_globe
            if globe is True and not is_globe:
                raise ValueError(
                    f"globe=True conflicts with projection={projection!r}. "
                    "Use projection=ccrs.Globe3D() for the 3D globe."
                )
            if globe is False and is_globe:
                raise ValueError("globe=False conflicts with projection=Globe3D()")
            return (projection, None if is_globe else projection.tp_name, is_globe)

        if isinstance(projection, str):
            # Bare projection names stay supported; look up a CRS when the name
            # matches one so parallels and clip angles still apply.
            if globe is True:
                raise ValueError("globe=True conflicts with a 2D projection name")
            try:
                resolved = _crs.get(projection)
            except (ValueError, TypeError):
                resolved = None
            if resolved is not None and (resolved.is_globe
                                         or resolved.is_transform_only):
                resolved = None          # not a drawable flat projection
            return (resolved, projection, False)

        if projection is None:
            if globe is False:
                raise ValueError("Set globe=False only together with projection=...")
            return (None, None, True)

        raise TypeError(
            "projection must be a pyterraplot.crs CRS, a projection name, or None; "
            f"got {type(projection).__name__}"
        )

    def _require_2d(self, what: str) -> None:
        if self.globe:
            raise NotImplementedError(
                f"{what} is currently 2D-projection only — construct the Axes "
                "with e.g. projection=ccrs.PlateCarree()"
            )

    # ── field primitives ─────────────────────────────────────────────────────

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

    def imshow(self, da: xr.DataArray, **opts) -> "Axes":
        """Alias for :meth:`pcolormesh` — matplotlib's name for a raster field."""
        return self._field("pcolormesh", da, opts)

    def quiver(self, u: xr.DataArray, v: xr.DataArray, **opts) -> "Axes":
        """Vector arrows (2D projections only). opts: color, cmap, scale,
        density, linewidth, headSize."""
        self._require_2d("quiver")
        self._check_vector(u, v, "quiver")
        self._register(u)
        self._register(v)
        self._calls.append(("quiver", id(u), id(v), _js_keys(opts)))
        return self

    def barbs(self, u: xr.DataArray, v: xr.DataArray, **opts) -> "Axes":
        """Wind barbs in the standard meteorological convention — a pennant per
        50, a full barb per 10, a half barb per 5, in the units of ``u``/``v``.

        opts: color, cmap, length, density, linewidth, flip (southern-hemisphere
        barb side), and the ``pennant``/``full``/``half`` speed thresholds.
        """
        self._require_2d("barbs")
        self._check_vector(u, v, "barbs")
        self._register(u)
        self._register(v)
        self._calls.append(("barbs", id(u), id(v), _js_keys(opts)))
        return self

    def streamplot(self, u: xr.DataArray, v: xr.DataArray, *, density: float = 1.0,
                   color: str = "rgba(190,215,255,0.85)", linewidth: float = 1.0,
                   step_deg: float = 0.5, max_steps: int = 400,
                   label: str | None = None, **opts) -> "Axes":
        """Streamlines through a vector field.

        Streamlines are integrated in Python (RK4 over the lon/lat grid, see
        :func:`pyterraplot.geodesy.streamlines`) and embedded as geometry, so
        the browser only has to project and draw them.
        """
        self._require_2d("streamplot")
        self._check_vector(u, v, "streamplot")
        lons, lats, uu, vv = _vector_grid(u, v)
        lines = geodesy.streamlines(
            lons, lats, uu, vv,
            density=density, step_deg=step_deg, max_steps=max_steps,
        )
        if not lines:
            return self
        geom = {"type": "MultiLineString", "coordinates": lines}
        style = {"color": color, "linewidth": linewidth, **_js_keys(opts)}
        self._calls.append(("geojson", geom, style))
        self._add_legend_entry(label, "line", style)
        return self

    # ── geographic features ──────────────────────────────────────────────────

    def add_feature(self, feature: "_feature.Feature | str", **style) -> "Axes":
        """Add a cartopy-style map feature.

        ``feature`` is a :mod:`pyterraplot.feature` constant (``cfeature.LAND``)
        or a bare name (``"coastlines"``). Style keywords override the feature's
        own: ``edgecolor``/``color``, ``facecolor``, ``linewidth``, ``alpha``.
        """
        if isinstance(feature, str):
            feature = _feature.Feature(feature)
        opts = feature.to_js(**style)
        if opts.get("scale") == "110m":
            opts.pop("scale")                      # the default needs no mention
        self._calls.append(("feature", feature.name, opts))
        return self

    def coastlines(self, resolution: str = "110m", *, color: str = "#39FF14",
                   linewidth: float | None = None, alpha: float = 0.9,
                   width: float = 2.0) -> "Axes":
        """Continental outlines. ``resolution`` is ``'110m'``, ``'50m'`` or ``'10m'``."""
        return self._line_feature("coastlines", resolution, color,
                                  linewidth if linewidth is not None else width, alpha)

    def borders(self, resolution: str = "110m", *,
                color: str = "rgba(180,180,210,0.5)",
                linewidth: float | None = None, alpha: float = 1.0,
                width: float = 1.0) -> "Axes":
        """National boundaries."""
        return self._line_feature("borders", resolution, color,
                                  linewidth if linewidth is not None else width, alpha)

    def states(self, resolution: str = "110m", *,
               color: str = "rgba(180,180,200,0.35)",
               linewidth: float = 0.5, alpha: float = 1.0) -> "Axes":
        """State / province boundaries."""
        return self._line_feature("states", resolution, color, linewidth, alpha)

    def rivers(self, resolution: str = "110m", *,
               color: str = "rgba(120,170,230,0.7)",
               linewidth: float = 0.6, alpha: float = 1.0) -> "Axes":
        """River centrelines."""
        return self._line_feature("rivers", resolution, color, linewidth, alpha)

    def lakes(self, resolution: str = "110m", *, facecolor: str = "#22406b",
              alpha: float = 1.0) -> "Axes":
        """Filled lakes."""
        return self.add_feature(_feature.LAKES.with_scale(resolution),
                                facecolor=facecolor, alpha=alpha)

    def land(self, resolution: str = "110m", *, facecolor: str = "#3b3b32",
             alpha: float = 1.0) -> "Axes":
        """Filled land polygons, drawn beneath the field layers."""
        return self.add_feature(_feature.LAND.with_scale(resolution),
                                facecolor=facecolor, alpha=alpha)

    def ocean(self, resolution: str = "110m", *, facecolor: str = "#1a2a45",
              alpha: float = 1.0) -> "Axes":
        """Filled ocean polygons, drawn beneath the field layers."""
        return self.add_feature(_feature.OCEAN.with_scale(resolution),
                                facecolor=facecolor, alpha=alpha)

    def cities(self, color: str = "#ffffff", opacity: float = 0.9,
               resolution: str = "110m") -> "Axes":
        """Major cities and populated places, with text labels."""
        opts: dict[str, Any] = {"color": color, "opacity": opacity}
        if resolution != "110m":
            opts["scale"] = resolution
        self._calls.append(("feature", "cities", opts))
        return self

    def stock_img(self) -> "Axes":
        """Blue-marble background image, as in cartopy's ``ax.stock_img()``."""
        self.earth_surface = "shaded_relief"
        return self

    def background_img(self, url: str) -> "Axes":
        """Use an arbitrary equirectangular image as the map background."""
        self.earth_surface = url
        return self

    def _line_feature(self, name: str, resolution: str, color: str,
                      linewidth: float, alpha: float) -> "Axes":
        # Key order matters: colour first keeps the emitted JS readable and
        # stable for callers that assert on it.
        opts: dict[str, Any] = {"color": color, "linewidth": linewidth,
                                "opacity": alpha}
        if resolution != "110m":
            opts["scale"] = resolution
        self._calls.append(("feature", name, opts))
        return self

    # ── gridlines ────────────────────────────────────────────────────────────

    def gridlines(self, *, draw_labels: bool = False,
                  xlocs: Sequence[float] | None = None,
                  ylocs: Sequence[float] | None = None,
                  xstep: float = 30.0, ystep: float = 30.0,
                  color: str = "rgba(255,255,255,0.25)",
                  linewidth: float = 0.6,
                  linestyle: str | None = None,
                  alpha: float = 1.0,
                  dms: bool = False,
                  label_size: float = 10,
                  label_color: str = "rgba(226,232,240,0.92)") -> "Axes":
        """Draw a lat/lon graticule, cartopy-style.

        ``xlocs``/``ylocs`` give explicit meridian/parallel positions; without
        them the grid is regular at ``xstep``/``ystep`` degrees. ``draw_labels``
        annotates each line where it meets the map edge. ``linestyle`` accepts
        matplotlib spellings (``'--'``, ``':'``, ``'-.'``).

        Replaces terraplot's default graticule rather than drawing over it.
        """
        self._require_2d("gridlines")
        opts: dict[str, Any] = {
            "color": color, "linewidth": linewidth, "opacity": alpha,
            "drawLabels": draw_labels, "labelSize": label_size,
            "labelColor": label_color, "dms": dms,
        }
        if xlocs is not None:
            opts["xlocs"] = [float(x) for x in xlocs]
        else:
            opts["xstep"] = float(xstep)
        if ylocs is not None:
            opts["ylocs"] = [float(y) for y in ylocs]
        else:
            opts["ystep"] = float(ystep)
        if linestyle:
            opts["linestyle"] = linestyle
        self._calls.append(("gridlines", opts))
        return self

    # ── point / line / text primitives ───────────────────────────────────────

    def plot(self, lons: Sequence[float] | float, lats: Sequence[float] | float,
             fmt: str | None = None, *,
             transform: "_crs.CRS | None" = None,
             color: str | None = None, linewidth: float | None = None,
             linestyle: str | None = None, alpha: float | None = None,
             marker: str | None = None, markersize: float | None = None,
             fill: str | None = None, closed: bool = False,
             label: str | None = None, **opts) -> "Axes":
        """Draw a line through geographic coordinates.

        ``fmt`` accepts matplotlib's shorthand (``'r--o'``). Pass
        ``transform=ccrs.Geodetic()`` to follow great circles between vertices
        instead of straight lines in lon/lat.
        """
        f_color, f_style, f_marker = _parse_fmt(fmt)
        coords = _as_coords(lons, lats)

        style: dict[str, Any] = {
            "color": color or f_color or "#fbbf24",
            "linewidth": 1.5 if linewidth is None else linewidth,
        }
        ls = linestyle or f_style
        if ls:
            style["linestyle"] = ls
        mk = marker or f_marker
        if mk:
            style["marker"] = mk
            style["markerSize"] = 4 if markersize is None else markersize
        if alpha is not None:
            style["opacity"] = alpha
        if fill:
            style["fill"] = fill
        if closed:
            style["closed"] = True
        if _is_geodetic(transform):
            style["geodesic"] = True
        style.update(_js_keys(opts))

        self._calls.append(("plot", coords, style))
        self._add_legend_entry(label, "line", style)
        return self

    def scatter(self, lons: Sequence[float] | float, lats: Sequence[float] | float,
                *, c: Sequence[float] | None = None,
                s: float | Sequence[float] | None = None,
                color: str | None = None, cmap: str | None = None,
                vmin: float | None = None, vmax: float | None = None,
                alpha: float | None = None, marker: str = "o",
                style_3d: str = "vertical_line", height: float | None = None,
                tooltip: str | dict | None = None,
                tooltips: Sequence[str | dict] | None = None,
                edgecolor: str | None = None, edgewidth: float | None = None,
                label: str | None = None, **opts) -> "Axes":
        """Scatter points or 3D markers at geographic coordinates.

        ``c`` colours points by value through ``cmap``; ``s`` sets the radius.
        ``tooltips`` / ``tooltip`` adds interactive hover metadata cards.
        On 3D globes, ``style_3d`` selects ``'vertical_line'`` (needle), ``'dot'``,
        or ``'pin'``.
        """
        lon_list = [float(x) for x in np.atleast_1d(np.asarray(lons, dtype=float))]
        lat_list = [float(y) for y in np.atleast_1d(np.asarray(lats, dtype=float))]

        style: dict[str, Any] = {"marker": marker, "style": style_3d}
        if height is not None:
            style["height"] = float(height)
        if tooltips is not None:
            style["tooltips"] = list(tooltips)
        elif tooltip is not None:
            style["tooltip"] = tooltip

        if c is not None:
            style["values"] = [float(v) for v in np.asarray(c, dtype=float).ravel()]
            style["cmap"] = cmap or "viridis"
            if vmin is not None:
                style["vmin"] = vmin
            if vmax is not None:
                style["vmax"] = vmax
        else:
            style["color"] = color or "#ef4444"
        if s is not None:
            arr = np.atleast_1d(np.asarray(s, dtype=float))
            if arr.size == 1:
                style["size"] = float(arr[0])
            else:
                style["sizes"] = [float(x) for x in arr]
        if alpha is not None:
            style["alpha"] = alpha
        if edgecolor:
            style["edgeColor"] = edgecolor
        if edgewidth is not None:
            style["edgeWidth"] = edgewidth
        style.update(_js_keys(opts))

        self._calls.append(("scatter", lon_list, lat_list, style))
        self._add_legend_entry(label, "point", style)
        return self

    def text(self, lon: float, lat: float, s: str, *,
             color: str = "#e2e8f0", fontsize: float = 12,
             weight: str | None = None, ha: str = "center",
             va: str = "middle", rotation: float | None = None,
             dx: float = 0, dy: float = 0, outline: bool = True,
             **opts) -> "Axes":
        """Place a text label at a geographic position (2D maps and 3D globe)."""
        anchor = {"center": "middle", "left": "start", "right": "end"}.get(ha, ha)
        baseline = {"center": "middle", "top": "hanging",
                    "bottom": "auto"}.get(va, va)
        style: dict[str, Any] = {
            "color": color, "fontSize": fontsize, "anchor": anchor,
            "baseline": baseline, "dx": dx, "dy": dy, "outline": outline,
        }
        if weight:
            style["fontWeight"] = weight
        if rotation:
            style["rotation"] = rotation
        style.update(_js_keys(opts))
        self._calls.append(("text", float(lon), float(lat), str(s), style))
        return self

    def annotate(self, text: str, xy: tuple[float, float], *,
                 xytext: tuple[float, float] | None = None, **kw) -> "Axes":
        """Label the point ``xy = (lon, lat)``, optionally offset in pixels."""
        dx, dy = xytext if xytext else (0, -12)
        return self.text(xy[0], xy[1], text, dx=dx, dy=dy, **kw)

    def marker(self, lat: float, lon: float, label: str | None = None,
               tooltip: str | dict | None = None,
               style: str = "vertical_line",
               height: float = 2.4,
               color: str = "#ef4444", size: float = 0.22, **opts) -> "Axes":
        """Point or 3D vertical needle marker with optional tooltip and label."""
        m_opts = {
            "lat": float(lat),
            "lon": float(lon),
            "label": label,
            "tooltip": tooltip,
            "style": style,
            "height": height,
            "color": color,
            "size": size,
            **_js_keys(opts)
        }
        self._calls.append(("marker", m_opts))
        return self

    def tissot(self, rad_km: float = 500.0, *,
               lons: Sequence[float] | None = None,
               lats: Sequence[float] | None = None,
               n_samples: int = 80, color: str = "rgba(255,120,120,0.9)",
               facecolor: str = "rgba(255,120,120,0.25)",
               linewidth: float = 1.0, **opts) -> "Axes":
        """Tissot's indicatrices — circles of constant ground radius.

        Distortion shows up as the departure of each projected circle from a
        circle. Defaults to a 30° × 30° global lattice.
        """
        self._require_2d("tissot")
        if lons is None:
            lons = np.arange(-180, 180, 30, dtype=float)
        if lats is None:
            lats = np.arange(-60, 90, 30, dtype=float)
        lon_arr = np.atleast_1d(np.asarray(lons, dtype=float))
        lat_arr = np.atleast_1d(np.asarray(lats, dtype=float))
        if lon_arr.size != lat_arr.size:
            centres = [(x, y) for y in lat_arr for x in lon_arr]
        else:
            centres = list(zip(lon_arr, lat_arr))

        rings = [[geodesy.geodesic_circle(float(x), float(y),
                                          rad_km * 1000.0, n_samples)]
                 for x, y in centres]
        geom = {"type": "MultiPolygon", "coordinates": rings}
        style = {"color": color, "fill": facecolor, "linewidth": linewidth,
                 **_js_keys(opts)}
        self._calls.append(("geojson", geom, style))
        return self

    def add_geometries(self, geoms, crs: "_crs.CRS | None" = None, *,
                       facecolor: str = "none",
                       edgecolor: str = "rgba(200,220,255,0.85)",
                       linewidth: float = 1.0, alpha: float | None = None,
                       label: str | None = None, **opts) -> "Axes":
        """Draw arbitrary geometries through the map projection or 3D globe.

        Accepts a GeoJSON mapping, anything exposing ``__geo_interface__``
        (shapely geometries, geopandas rows), or an iterable of either.
        ``crs`` is accepted for cartopy parity and must describe lon/lat
        coordinates, which is all terraplot consumes.
        """
        if crs is not None and not isinstance(crs, (_crs.PlateCarree, _crs.Geodetic)):
            raise NotImplementedError(
                "add_geometries only reads lon/lat coordinates — pass "
                "crs=ccrs.PlateCarree() (the default) and reproject beforehand "
                f"if your geometries are in {type(crs).__name__}."
            )
        geojson = _to_geojson(geoms)
        style: dict[str, Any] = {"color": edgecolor, "fill": facecolor,
                                 "linewidth": linewidth}
        if alpha is not None:
            style["opacity"] = alpha
        style.update(_js_keys(opts))
        self._calls.append(("geojson", geojson, style))
        self._add_legend_entry(label, "patch", style)
        return self

    # ── extent / view ────────────────────────────────────────────────────────

    def set_extent(self, extents: Sequence[float],
                   crs: "_crs.CRS | None" = None) -> "Axes":
        """Zoom to ``[lon0, lon1, lat0, lat1]``, as in cartopy."""
        if len(extents) != 4:
            raise ValueError(
                "extents must be [lon0, lon1, lat0, lat1]; got "
                f"{len(extents)} values"
            )
        if crs is not None and not isinstance(crs, (_crs.PlateCarree, _crs.Geodetic)):
            raise NotImplementedError(
                "set_extent reads lon/lat degrees — pass crs=ccrs.PlateCarree() "
                f"(the default) rather than {type(crs).__name__}."
            )
        self.extent = tuple(float(x) for x in extents)  # type: ignore[assignment]
        return self

    def get_extent(self) -> tuple[float, float, float, float] | None:
        """The current extent, or None when the view is global."""
        return self.extent

    def set_global(self) -> "Axes":
        """Drop any regional extent and show the whole world."""
        self.extent = None
        return self

    def set_center(self, lon: float, lat: float = 0.0) -> "Axes":
        """Recentre the projection (or the globe's point of view)."""
        self.center = (float(lon), float(lat))
        return self

    # ── labels ───────────────────────────────────────────────────────────────

    def title(self, text: str) -> "Axes":
        """Set the plot title (page title and overlay label)."""
        self._title = text
        return self

    def set_title(self, text: str) -> "Axes":
        """matplotlib spelling of :meth:`title`."""
        return self.title(text)

    def legend(self, *, loc: str = "upper right", title: str | None = None,
               background: str = "rgba(8,8,20,0.78)") -> "Axes":
        """Show a legend for every layer that was given a ``label=``.

        ``loc`` is one of ``'upper left'``, ``'upper right'``, ``'lower left'``,
        ``'lower right'``.
        """
        self._legend = {"loc": loc, "title": title, "background": background}
        return self

    def colorbar(self, da: xr.DataArray | None = None, **opts) -> "Axes":
        """Add a colorbar primitive to the axes.

        Parameters
        ----------
        da : optional DataArray whose field values and metadata (cmap, vmin, vmax,
             units, long_name) provide defaults for the colorbar. If omitted,
             inherits from the preceding or first plotted field layer.
        **opts : Colorbar widget options:
            orientation : 'horizontal' | 'vertical' (default 'horizontal')
            position    : 'bottom' | 'top' | 'left' | 'right' (default 'bottom')
            panel       : bool (default True, translucent backing)
            background  : str (default 'rgba(0,0,0,0.65)')
            scale       : 'linear' | 'log' | 'symlog' | 'power' | 'sqrt'
            power       : float (exponent for power scale)
            linthresh   : float (threshold for symlog scale)
            ticks       : int or list of numeric tick values
            format      : tick label formatter
            width       : int (length in px)
            height      : int (thickness in px)
            label       : str (colorbar title text)
            cmap        : str (colormap name)
            vmin, vmax  : float (data range)

        Example
        -------
        >>> ax.pcolormesh(precip, cmap='YlGnBu')
        >>> ax.colorbar(orientation='vertical', position='right', scale='sqrt',
        ...             ticks=[0, 1, 2, 5, 10, 20, 35])
        """
        target_id = None
        if da is not None:
            if not hasattr(da, "tp"):
                raise TypeError("colorbar(da) expects an xarray DataArray")
            self._register(da)
            target_id = id(da)
        self._calls.append(("colorbar", target_id, opts))
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

    # ── rendering ────────────────────────────────────────────────────────────

    def to_html(self, path: str | Path, *, title: str = "terraplot",
                binary: bool = True, terraplot_bundle: str | Path | None = None) -> Path:
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
            html = self._render_html(title=self._title or "terraplot", height_px=420)
            srcdoc = html.replace("&", "&amp;").replace('"', "&quot;")
            return (
                f'<iframe srcdoc="{srcdoc}" width="100%" height="440" '
                f'style="border:1px solid rgba(255,255,255,.1); border-radius:6px;"></iframe>'
            )
        except Exception as e:  # pragma: no cover — Jupyter must never crash
            return f"<pre>pyterraplot Axes render failed: {e}</pre>"

    # ── internals ────────────────────────────────────────────────────────────

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
        label = opts.pop("label", None)
        var = self._register(da)
        js_opts = _js_keys(opts)
        self._calls.append(("field", kind, var, js_opts))
        self._add_legend_entry(label, "line" if kind == "contour" else "patch",
                               js_opts)
        return self

    @staticmethod
    def _check_vector(u, v, what: str) -> None:
        for arr in (u, v):
            if not hasattr(arr, "tp"):
                raise TypeError(f"{what}(u, v) expects xarray DataArrays")

    def _add_legend_entry(self, label, kind: str, style: dict) -> None:
        if not label:
            return
        self._legend_entries.append({
            "label": label,
            "kind": kind,
            "color": style.get("color") or style.get("fill") or "#fbbf24",
            "linestyle": style.get("linestyle"),
        })

    # ── JS emission ──────────────────────────────────────────────────────────

    def _ctor_js(self, map_var: str, selector: str) -> str:
        """The `const <map_var> = new GeoMap(...)` (or GeoSphere) statement."""
        if self.globe:
            cen_lon, cen_lat = self.center[0], self.center[1]
            alt = 2.5
            if self.extent:
                cen_lon = (self.extent[0] + self.extent[1]) / 2.0
                cen_lat = (self.extent[2] + self.extent[3]) / 2.0
                span = max(abs(self.extent[1] - self.extent[0]), abs(self.extent[3] - self.extent[2]))
                # Close-up altitude for regional extents
                alt = max(1.06, min(2.5, 1.02 + (span / 60.0) * 0.8))
            ctor = (f"new GeoSphere('{selector}', {{ earthSurface: "
                    f"'{self.earth_surface}', autoRotate: {str(self.spin).lower()} }});\n"
                    f"{map_var}.setPointOfView({{ lat: {cen_lat:.2f}, "
                    f"lng: {cen_lon:.2f}, altitude: {alt:.2f} }});")
            return f"const {map_var} = {ctor}"

        opts: dict[str, Any] = {}
        if self.crs is not None and not self.crs.is_globe:
            opts.update(self.crs.to_js())
        else:
            opts["projection"] = self.projection or "equirectangular"
            opts["center"] = [self.center[0], self.center[1]]
        # An explicit center= argument wins over the projection's own.
        if "rotate" not in opts:
            opts["center"] = [self.center[0], self.center[1]]
        if self.extent:
            opts["extent"] = list(self.extent)
        opts["background"] = self.background or "transparent"
        opts["earthSurface"] = self.earth_surface
        opts["tooltip"] = bool(self.tooltip)
        if self.graticule is not None:
            opts["graticule"] = bool(self.graticule)
        elif any(c[0] == "gridlines" for c in self._calls):
            # gridlines() supersedes the built-in grid; don't draw both.
            opts["graticule"] = False
        return f"const {map_var} = new GeoMap('{selector}', {json.dumps(opts)});"

    def _layers_js(self, map_var: str,
                   cbar_host: str | None = None) -> tuple[list[str], dict]:
        """Emit the per-layer statements. Returns (lines, colorbar metadata)."""
        lines: list[str] = []
        last_field_opts = None
        last_field_var = None
        first_field_opts = None
        first_field_var = None
        has_explicit_cbar = any(call[0] == "colorbar" for call in self._calls)

        for call in self._calls:
            op = call[0]
            if op == "field":
                _, kind, var, opts = call
                lines.append(f"{map_var}.{kind}({var}.lons, {var}.lats, {var}.field, "
                             f"{json.dumps(opts)});")
                last_field_opts = opts
                last_field_var = var
                if first_field_opts is None and kind in ("pcolormesh", "contourf"):
                    first_field_opts = opts
                    first_field_var = var
            elif op == "feature":
                _, name, opts = call
                lines.append(f"{map_var}.addFeature('{name}', {json.dumps(opts)});")
            elif op == "gridlines":
                lines.append(f"{map_var}.gridlines({json.dumps(call[1])});")
            elif op == "quiver":
                _, uid, vid, opts = call
                u = self._payloads[uid]["var"]
                v = self._payloads[vid]["var"]
                lines.append(f"{map_var}.quiver({u}.lons, {u}.lats, {u}.field, "
                             f"{v}.field, {json.dumps(opts)});")
            elif op == "barbs":
                _, uid, vid, opts = call
                u = self._payloads[uid]["var"]
                v = self._payloads[vid]["var"]
                lines.append(f"{map_var}.barbs({u}.lons, {u}.lats, {u}.field, "
                             f"{v}.field, {json.dumps(opts)});")
            elif op == "plot":
                _, coords, style = call
                lines.append(f"{map_var}.plot({json.dumps(coords)}, {json.dumps(style)});")
            elif op == "scatter":
                _, lon_list, lat_list, style = call
                lines.append(f"{map_var}.scatter({json.dumps(lon_list)}, "
                             f"{json.dumps(lat_list)}, {json.dumps(style)});")
            elif op == "text":
                _, lon, lat, s, style = call
                lines.append(f"{map_var}.text({lon}, {lat}, {json.dumps(s)}, "
                             f"{json.dumps(style)});")
            elif op == "geojson":
                _, geom, style = call
                lines.append(f"{map_var}.addGeoJSON({json.dumps(geom)}, "
                             f"{json.dumps(style)});")
            elif op == "marker":
                m = call[1]
                lat_v = int(m['lat']) if isinstance(m['lat'], (int, float)) and float(m['lat']).is_integer() else m['lat']
                lon_v = int(m['lon']) if isinstance(m['lon'], (int, float)) and float(m['lon']).is_integer() else m['lon']
                lines.append(f"{map_var}.marker({lat_v}, {lon_v}, {json.dumps(m)});\n")
            elif op == "colorbar":
                _, target_id, cbar_opts = call
                cbar_cfg = dict(cbar_opts)
                target_var = (self._payloads[target_id]["var"]
                              if target_id and target_id in self._payloads
                              else (last_field_var or first_field_var))
                f_opts = last_field_opts or first_field_opts or {}
                for key in ("cmap", "vmin", "vmax"):
                    if key not in cbar_cfg and key in f_opts:
                        cbar_cfg[key] = f_opts[key]
                if "label" not in cbar_cfg:
                    if target_id and target_id in self._payloads:
                        p = self._payloads[target_id]["payload"]
                        ln = p.get("long_name") or p.get("name") or ""
                        u = p.get("units", "")
                        cbar_cfg["label"] = f"{ln} [{u}]" if u else ln
                field_ref = f"{target_var}.field" if target_var else "[]"
                host_js = (f"document.querySelector('{cbar_host}')"
                           if cbar_host else "document.body.appendChild("
                                             "document.createElement('div'))")
                lines.append(f"""(() => {{
  const cbarOpts = {json.dumps(cbar_cfg)};
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
  const host = {host_js};
  new Colorbar(host, cbarOpts);
}})();""")

        meta = {
            "has_explicit_cbar": has_explicit_cbar,
            "first_field_opts": first_field_opts,
            "first_field_var": first_field_var,
        }
        return lines, meta

    def _anim_js(self, map_var: str) -> str:
        if not self._anim:
            return ""
        a = self._anim
        b64 = pack_frames(a["compact"])
        lopts = _js_keys(dict(a["layer_opts"]))
        # "map" -> F0 (single Axes), "map3" -> F3 (panel 3 of a Figure)
        var = "F" + (map_var[3:] or "0")
        return (f'const {var} = await unpackFrames("{b64}");\n'
                f"{map_var}.animate({var}, {{ type: '{a['kind']}', "
                f"interval: {a['interval']}, layerOptions: {json.dumps(lopts)} }});")

    def _label_text(self) -> str:
        first = next(iter(self._payloads.values()), None)
        if first:
            payload = first["payload"]
            long_name = payload.get("long_name") or payload.get("name") or ""
            units = payload.get("units", "")
            label = f"{long_name} [{units}]" if units else long_name
        else:
            label = ""
        if self._title:
            label = f"{self._title} — {label}" if label else self._title
        return label

    def _legend_html(self) -> str:
        if not self._legend or not self._legend_entries:
            return ""
        loc = self._legend["loc"]
        vert, horiz = (loc.split() + ["right"])[:2]
        pos = (f"{'top' if vert == 'upper' else 'bottom'}: 3.2rem; "
               f"{'left' if horiz == 'left' else 'right'}: 1rem;")
        rows = []
        for e in self._legend_entries:
            if e["kind"] == "point":
                swatch = (f'<span class="tp-swatch-dot" '
                          f'style="background:{e["color"]}"></span>')
            elif e["kind"] == "line":
                dash = "dashed" if e.get("linestyle") in ("--", "dashed") else \
                       "dotted" if e.get("linestyle") in (":", "dotted") else "solid"
                swatch = (f'<span class="tp-swatch-line" '
                          f'style="border-top:2px {dash} {e["color"]}"></span>')
            else:
                swatch = (f'<span class="tp-swatch-box" '
                          f'style="background:{e["color"]}"></span>')
            rows.append(f'<div class="tp-legend-row">{swatch}'
                        f'<span>{_esc(e["label"])}</span></div>')
        title = (f'<div class="tp-legend-title">{_esc(self._legend["title"])}</div>'
                 if self._legend.get("title") else "")
        return (f'<div class="tp-legend" style="{pos} '
                f'background:{self._legend["background"]}">{title}'
                + "".join(rows) + "</div>")

    def _render_html(self, *, title="terraplot", binary=True,
                     terraplot_bundle=None, height_px=None) -> str:
        bundle_js = _load_terraplot_bundle(terraplot_bundle)

        ctor = self._ctor_js("map", "#map")
        decls = _payload_decls(self._payloads, binary)
        lines, meta = self._layers_js("map")
        anim_js = self._anim_js("map")

        label = self._label_text()
        if self._title:
            title = self._title

        colorbar_js = _fallback_colorbar_js(
            meta, self._payloads, self._cbar_opts, label)

        map_height = (f"{height_px or self.height_px}px"
                      if (height_px or self.height_px) else "100vh")

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>{title}</title>
<style>
{_SHARED_CSS}
{_LEGEND_CSS}
  #map {{ width: 100vw; height: {map_height}; }}
</style>
</head>
<body>
<div id="map"></div>
<div id="label">{label}</div>
<div id="colorbar"></div>
{self._legend_html()}
{_IMPORTMAP}
<script type="module">
{bundle_js}

{ctor}
{chr(10).join(decls)}
{chr(10).join(lines)}
{anim_js}
{colorbar_js}
</script>
</body>
</html>"""


# ── shared rendering helpers (also used by Figure) ───────────────────────────

_LEGEND_CSS = """
  .tp-legend {
    position: fixed; z-index: 120; padding: .5rem .7rem; border-radius: 6px;
    border: 1px solid rgba(255,255,255,.14); font-size: .74rem; line-height: 1.6;
    pointer-events: none; color: #e2e8f0;
  }
  .tp-legend-title { font-weight: 600; margin-bottom: .25rem; opacity: .85; }
  .tp-legend-row { display: flex; align-items: center; gap: .45rem; }
  .tp-swatch-dot  { width: 9px; height: 9px; border-radius: 50%; display: inline-block; }
  .tp-swatch-box  { width: 12px; height: 9px; border-radius: 2px; display: inline-block; }
  .tp-swatch-line { width: 14px; height: 0; display: inline-block; }
"""


def _payload_decls(payloads: dict[int, dict[str, Any]], binary: bool) -> list[str]:
    """`const P0 = …` declarations for every registered field payload."""
    decls = []
    for entry in payloads.values():
        if binary:
            decls.append(f'const {entry["var"]} = '
                         f'await unpackField("{pack_field(entry["payload"])}");')
        else:
            decls.append(f'const {entry["var"]} = {json.dumps(entry["payload"])};')
    return decls


def _fallback_colorbar_js(meta: dict, payloads: dict, cbar_opts, label: str) -> str:
    """The auto-derived colorbar, used when no colorbar() call was made."""
    first_field_opts = meta.get("first_field_opts")
    if meta.get("has_explicit_cbar") or not (first_field_opts or cbar_opts):
        return ""
    cbar_src = dict(cbar_opts or {})
    if first_field_opts:
        cbar_src.setdefault("cmap", first_field_opts.get("cmap", "viridis"))
        cbar_src.setdefault("vmin", first_field_opts.get("vmin"))
        cbar_src.setdefault("vmax", first_field_opts.get("vmax"))
    if label and "label" not in cbar_src:
        cbar_src["label"] = label
    cvar = next(iter(payloads.values()), None)
    return (
        _COLORBAR_JS
        .replace("})(payload.field,",
                 f"}})({cvar['var']}.field," if cvar else "})([],", 1)
        .replace("CBAR_OPTS", json.dumps(cbar_src))
    )


def _js_keys(opts: dict[str, Any]) -> dict[str, Any]:
    """Translate snake_case Python keywords into terraplot's camelCase options."""
    mapping = {
        "line_width": "linewidth",
        "head_size": "headSize",
        "marker_size": "markerSize",
        "edge_color": "edgeColor",
        "edge_width": "edgeWidth",
        "font_size": "fontSize",
        "font_weight": "fontWeight",
        "draw_labels": "drawLabels",
        "label_size": "labelSize",
        "label_color": "labelColor",
        "facecolor": "fill",
        "edgecolor": "color",
    }
    return {mapping.get(k, k): v for k, v in opts.items()}


def _parse_fmt(fmt: str | None) -> tuple[str | None, str | None, str | None]:
    """Split a matplotlib format string like ``'r--o'`` into its three parts."""
    if not fmt:
        return (None, None, None)
    rest = fmt
    color = linestyle = marker = None

    # A colour code only counts in matplotlib's leading position, so 'k--o' is
    # black-dashed-circle while 'o' alone stays a marker.
    if rest and rest[0] in _COLOR_CODES:
        color = _COLOR_CODES[rest[0]]
        rest = rest[1:]
    for ls in _LINESTYLES:                      # '--' before '-' matters
        if ls in rest:
            linestyle = ls
            rest = rest.replace(ls, "", 1)
            break
    for ch in rest:
        if ch in _MARKERS:
            marker = ch
            break
    return (color, linestyle, marker)


def _as_coords(lons, lats) -> list[list[float]]:
    """Zip lon/lat sequences (or scalars) into ``[[lon, lat], …]``."""
    lon_arr = np.atleast_1d(np.asarray(lons, dtype=float))
    lat_arr = np.atleast_1d(np.asarray(lats, dtype=float))
    if lon_arr.size != lat_arr.size:
        raise ValueError(
            f"lons and lats must be the same length; got {lon_arr.size} and "
            f"{lat_arr.size}"
        )
    return [[float(a), float(b)] for a, b in zip(lon_arr, lat_arr)]


def _is_geodetic(transform) -> bool:
    if transform is None:
        return False
    if isinstance(transform, _crs.Geodetic):
        return True
    if isinstance(transform, _crs.PlateCarree):
        return False
    raise NotImplementedError(
        "transform must be ccrs.PlateCarree() (straight in lon/lat) or "
        f"ccrs.Geodetic() (great circles); got {type(transform).__name__}."
    )


def _to_geojson(geoms) -> dict[str, Any]:
    """Coerce GeoJSON mappings, ``__geo_interface__`` objects, or an iterable
    of either into a single GeoJSON object."""
    if hasattr(geoms, "__geo_interface__"):
        return dict(geoms.__geo_interface__)
    if isinstance(geoms, dict):
        return geoms
    if isinstance(geoms, Iterable) and not isinstance(geoms, (str, bytes)):
        parts = [_to_geojson(g) for g in geoms]
        if not parts:
            raise ValueError("add_geometries got an empty geometry collection")
        if len(parts) == 1:
            return parts[0]
        return {"type": "GeometryCollection",
                "geometries": [p.get("geometry", p) for p in parts]}
    raise TypeError(
        "add_geometries expects GeoJSON, an object with __geo_interface__ "
        f"(e.g. a shapely geometry), or an iterable of those; got "
        f"{type(geoms).__name__}"
    )


def _vector_grid(u: xr.DataArray, v: xr.DataArray):
    """Extract aligned (lons, lats, u, v) numpy arrays from two DataArrays."""
    pu = u.tp.to_dict()
    pv = v.tp.to_dict()
    lons = np.asarray(pu["lons"], dtype=float)
    lats = np.asarray(pu["lats"], dtype=float)
    uu = np.array([[np.nan if x is None else x for x in row] for row in pu["field"]],
                  dtype=float)
    vv = np.array([[np.nan if x is None else x for x in row] for row in pv["field"]],
                  dtype=float)
    if uu.shape != vv.shape:
        raise ValueError(
            f"u and v must share a grid; got {uu.shape} and {vv.shape}"
        )
    return lons, lats, uu, vv


def _esc(text: str) -> str:
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))
