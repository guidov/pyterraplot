"""
Colormap names accepted by the terraplot JS bundle.

pyterraplot passes cmap names through to terraplot's resolveColormap(),
which resolves two families:

  - matplotlib / d3-scale-chromatic style (viridis, RdBu_r, YlGnBu, ...)
  - cmocean (https://matplotlib.org/cmocean/) — 22 perceptually-uniform
    oceanographic colormaps plus their _r reversals (thermal, haline,
    ice, balance, ...)

The lists here must stay in sync with terraplot's src/colormaps.js and
src/cmocean.js (ColormapGroups is the canonical grouping on the JS side).
"""
from __future__ import annotations

CMOCEAN = (
    "algae", "amp", "balance", "curl", "deep", "delta", "dense", "diff",
    "gray", "haline", "ice", "matter", "oxy", "phase", "rain", "solar",
    "speed", "tarn", "tempo", "thermal", "topo", "turbid",
)
CMOCEAN_REVERSED = tuple(f"{n}_r" for n in CMOCEAN)

DIVERGING = (
    "RdBu_r", "RdBu", "RdYlBu_r", "RdYlBu", "BrBG", "BrBG_r",
    "Spectral_r", "Spectral",
)
PERCEPTUALLY_UNIFORM = ("viridis", "plasma", "inferno", "magma")
SEQUENTIAL = (
    "YlGnBu", "PuBuGn", "BuPu", "OrRd", "YlOrRd",
    "Blues", "Greens", "Greys",
)

COLORMAP_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Diverging (anomalies)",      DIVERGING),
    ("Perceptually uniform",       PERCEPTUALLY_UNIFORM),
    ("Sequential (intensity)",     SEQUENTIAL),
    ("cmocean (oceanographic)",    CMOCEAN),
    ("cmocean reversed (_r)",      CMOCEAN_REVERSED),
)

ALL_COLORMAPS: tuple[str, ...] = tuple(
    name for _, names in COLORMAP_GROUPS for name in names
)


def is_valid_cmap(name: str) -> bool:
    return name in ALL_COLORMAPS


def cmap_options_html(selected: str = "RdBu_r") -> str:
    """Render <option> markup (with <optgroup> per family) for a <select>."""
    parts: list[str] = []
    for label, names in COLORMAP_GROUPS:
        parts.append(f'<optgroup label="{label}">')
        for name in names:
            mark = " selected" if name == selected else ""
            parts.append(f'<option value="{name}"{mark}>{name}</option>')
        parts.append("</optgroup>")
    return "\n".join(parts)


def format_cmap_help() -> str:
    """Human-readable grouped listing for --help / --list-cmaps output."""
    lines = ["Available colormaps:"]
    for label, names in COLORMAP_GROUPS:
        lines.append(f"  {label}:")
        lines.append("    " + ", ".join(names))
    return "\n".join(lines)
