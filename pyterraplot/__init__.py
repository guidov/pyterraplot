"""
pyterraplot — matplotlib + cartopy, rendered by terraplot in the browser.

Two layers sit on top of the xarray ``.tp`` accessor:

* :class:`~pyterraplot.Axes` — a compositional plot container with the
  cartopy primitive set (``contourf``, ``coastlines``, ``gridlines``,
  ``set_extent``, ``quiver``, ``barbs``, ``streamplot``, ``tissot``, …).
* :func:`~pyterraplot.subplots` — matplotlib's multi-panel figure layout.

Projections use cartopy's instantiation style::

    import pyterraplot as tp
    import pyterraplot.crs as ccrs
    import pyterraplot.feature as cfeature

    fig, ax = tp.subplots(projection=ccrs.NorthPolarStereo())
    ax.contourf(t2m, levels=14, cmap="RdBu_r")
    ax.add_feature(cfeature.LAND)
    ax.gridlines(draw_labels=True)
    fig.savefig("arctic.html")
"""
from . import crs                              # noqa: F401 — pyterraplot.crs
from . import feature                          # noqa: F401 — pyterraplot.feature
from . import geodesy                          # noqa: F401
from .accessor import TerraplotAccessor        # noqa: F401 — registers .tp on DataArray
from .dataset import TerraplotDatasetAccessor  # noqa: F401 — registers .tp on Dataset
from .axes import Axes
from .figure import Figure, figure, subplots
from .serialize import serialize
from .server import serve
from .binary import pack_field, pack_frames
from .cog import to_cog

# Projection classes are also re-exported at the top level, so both
# `ccrs.PlateCarree()` and `tp.PlateCarree()` work.
from .crs import (  # noqa: F401
    CRS, Geodetic, Globe3D,
    PlateCarree, Mercator, TransverseMercator, Miller, LambertCylindrical,
    Robinson, Mollweide, Sinusoidal, EqualEarth, NaturalEarth, NaturalEarth2,
    EckertI, EckertII, EckertIII, EckertIV, EckertV, EckertVI,
    Orthographic, Stereographic, NorthPolarStereo, SouthPolarStereo,
    LambertAzimuthalEqualArea, AzimuthalEquidistant, Gnomonic,
    NearsidePerspective,
    AlbersEqualArea, LambertConformal, EquidistantConic,
    Aitoff, Hammer, WinkelTripel, Bonne, Polyconic, VanDerGrinten,
    Lagrange, Times,
    InterruptedGoodeHomolosine, InterruptedMollweide, InterruptedSinusoidal,
    InterruptedBoggs, RotatedPole,
)

__version__ = "0.4.0"
__all__ = [
    "TerraplotAccessor",
    "TerraplotDatasetAccessor",
    "Axes",
    "Figure",
    "figure",
    "subplots",
    "crs",
    "feature",
    "geodesy",
    "serialize",
    "serve",
    "pack_field",
    "pack_frames",
    "to_cog",
    # projections
    "CRS", "Geodetic", "Globe3D",
    "PlateCarree", "Mercator", "TransverseMercator", "Miller",
    "LambertCylindrical", "Robinson", "Mollweide", "Sinusoidal", "EqualEarth",
    "NaturalEarth", "NaturalEarth2",
    "EckertI", "EckertII", "EckertIII", "EckertIV", "EckertV", "EckertVI",
    "Orthographic", "Stereographic", "NorthPolarStereo", "SouthPolarStereo",
    "LambertAzimuthalEqualArea", "AzimuthalEquidistant", "Gnomonic",
    "NearsidePerspective", "AlbersEqualArea", "LambertConformal",
    "EquidistantConic", "Aitoff", "Hammer", "WinkelTripel", "Bonne",
    "Polyconic", "VanDerGrinten", "Lagrange", "Times",
    "InterruptedGoodeHomolosine", "InterruptedMollweide",
    "InterruptedSinusoidal", "InterruptedBoggs", "RotatedPole",
]
