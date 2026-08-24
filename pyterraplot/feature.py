"""
Map features — cartopy-style ``NaturalEarthFeature`` objects.

Mirrors ``cartopy.feature``, so the familiar constants work::

    import pyterraplot.feature as cfeature

    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN, facecolor="#0b1a30")
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linestyle=":")

Geometry is fetched in the browser from Natural Earth GeoJSON, at the scale
tier the feature carries (``110m`` coarse, ``50m`` regional, ``10m`` detailed).
Nothing is downloaded at Python time — a feature is just a style + scale
record that the exported HTML turns into a terraplot ``addFeature`` call.
"""
from __future__ import annotations

from typing import Any

__all__ = [
    "Feature", "NaturalEarthFeature",
    "COASTLINE", "COASTLINES", "BORDERS", "LAND", "OCEAN",
    "LAKES", "RIVERS", "STATES", "CITIES",
    "SCALES",
]

#: Natural Earth resolution tiers, coarsest first.
SCALES = ("110m", "50m", "10m")

# Features that are polygons rather than line work — these fill by default.
_AREAL = {"land", "ocean", "lakes"}

# Roughly cartopy's cfeature palette, tuned for terraplot's dark background.
_DEFAULT_STYLE: dict[str, dict[str, Any]] = {
    "coastlines": {"edgecolor": "rgba(200,220,255,0.85)", "linewidth": 0.8},
    "borders":    {"edgecolor": "rgba(180,180,200,0.55)", "linewidth": 0.7},
    "states":     {"edgecolor": "rgba(180,180,200,0.35)", "linewidth": 0.5},
    "rivers":     {"edgecolor": "rgba(120,170,230,0.7)",  "linewidth": 0.6},
    "land":       {"facecolor": "#3b3b32", "edgecolor": "none", "linewidth": 0},
    "ocean":      {"facecolor": "#1a2a45", "edgecolor": "none", "linewidth": 0},
    "lakes":      {"facecolor": "#22406b", "edgecolor": "none", "linewidth": 0},
    "cities":     {"edgecolor": "#ffffff", "linewidth": 0},
}


class Feature:
    """A named map feature with a scale tier and a style.

    Features are immutable: :meth:`with_scale` and :meth:`with_style` return
    new instances, so the module-level constants stay safe to share.
    """

    def __init__(self, name: str, scale: str = "110m", **style):
        if scale not in SCALES:
            raise ValueError(
                f"scale must be one of {SCALES}, got {scale!r}"
            )
        self.name = name
        self.scale = scale
        self.style: dict[str, Any] = {**_DEFAULT_STYLE.get(name, {}), **style}

    # ── derivation ────────────────────────────────────────────────────────────

    def with_scale(self, scale: str) -> "Feature":
        """Return this feature at a different Natural Earth resolution."""
        return type(self)(self.name, scale, **self.style)

    def with_style(self, **style) -> "Feature":
        """Return this feature with extra or overridden style keys."""
        return type(self)(self.name, self.scale, **{**self.style, **style})

    @property
    def is_areal(self) -> bool:
        """True for polygon features, which fill rather than stroke."""
        return self.name in _AREAL

    # ── translation to the browser ────────────────────────────────────────────

    def to_js(self, **overrides) -> dict[str, Any]:
        """Options for terraplot's ``GeoMap.addFeature(name, opts)``."""
        style = {**self.style, **{k: v for k, v in overrides.items()
                                  if v is not None}}
        opts: dict[str, Any] = {"scale": self.scale}

        edge = style.get("edgecolor", style.get("color"))
        face = style.get("facecolor")
        if edge is not None:
            opts["color"] = edge
        if face is not None:
            opts["fill"] = face
        elif self.is_areal:
            opts["fill"] = _DEFAULT_STYLE[self.name]["facecolor"]
        if "linewidth" in style:
            opts["linewidth"] = style["linewidth"]
        if "alpha" in style:
            opts["opacity"] = style["alpha"]
        if "opacity" in style:
            opts["opacity"] = style["opacity"]
        if style.get("url"):
            opts["url"] = style["url"]
        return opts

    def __repr__(self) -> str:
        extra = "".join(f", {k}={v!r}" for k, v in self.style.items())
        return f"Feature({self.name!r}, scale={self.scale!r}{extra})"

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, Feature) and self.name == other.name
                and self.scale == other.scale and self.style == other.style)

    def __hash__(self) -> int:
        return hash((self.name, self.scale, tuple(sorted(self.style.items()))))


class NaturalEarthFeature(Feature):
    """A Natural Earth feature named by cartopy's ``(category, name, scale)``.

    ``category`` is accepted for cartopy parity; terraplot resolves the layer
    from ``name`` alone, so the two spellings below are equivalent::

        NaturalEarthFeature("physical", "coastline", "50m")
        Feature("coastlines", "50m")
    """

    #: cartopy's Natural Earth layer names → terraplot's feature keys
    _ALIASES = {
        "coastline": "coastlines",
        "admin_0_boundary_lines_land": "borders",
        "admin_1_states_provinces_lines": "states",
        "rivers_lake_centerlines": "rivers",
        "populated_places": "cities",
    }

    def __init__(self, category: str, name: str, scale: str = "110m", **style):
        resolved = self._ALIASES.get(name, name)
        super().__init__(resolved, scale, **style)
        self.category = category

    def with_scale(self, scale: str) -> "NaturalEarthFeature":
        return NaturalEarthFeature(self.category, self.name, scale, **self.style)

    def with_style(self, **style) -> "NaturalEarthFeature":
        return NaturalEarthFeature(self.category, self.name, self.scale,
                                   **{**self.style, **style})


# ── cartopy-compatible constants ─────────────────────────────────────────────

COASTLINE = Feature("coastlines")
COASTLINES = COASTLINE          # cartopy spells it COASTLINE; both read fine
BORDERS = Feature("borders")
STATES = Feature("states")
LAND = Feature("land")
OCEAN = Feature("ocean")
LAKES = Feature("lakes")
RIVERS = Feature("rivers")
CITIES = Feature("cities")
