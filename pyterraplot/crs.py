"""
Coordinate reference systems — cartopy-style projection objects.

Mirrors the ``cartopy.crs`` instantiation pattern, so code reads the same::

    import pyterraplot as tp
    import pyterraplot.crs as ccrs

    ax = tp.Axes(projection=ccrs.Orthographic(central_longitude=-95,
                                              central_latitude=60))
    ax.contourf(t2m, levels=14, cmap="RdBu_r")
    ax.coastlines()
    ax.gridlines(draw_labels=True)
    ax.to_html("plot.html")

Each class carries the parameters cartopy uses and knows how to translate them
into the d3-geo projection terraplot drives in the browser: a projection name,
a rotation, standard parallels, a clip angle.

Two entries are not map projections in the drawing sense:

``Geodetic()``
    A transform-only CRS. Passing ``transform=ccrs.Geodetic()`` to
    :meth:`~pyterraplot.Axes.plot` densifies the line along great circles;
    ``transform=ccrs.PlateCarree()`` (the default) keeps segments straight in
    lon/lat.

``Globe3D()``
    The interactive 3D sphere (terraplot's ``GeoSphere``) rather than a flat
    projection, so ``Axes(projection=ccrs.Globe3D())`` is the explicit spelling
    of the default globe view.

Ellipsoids and datums are not modelled — terraplot renders on a sphere, so
cartopy's ``globe=ccrs.Globe(ellipse=...)`` argument has no counterpart here.
"""
from __future__ import annotations

from typing import Any

__all__ = [
    "CRS", "Geodetic", "Globe3D",
    # cylindrical
    "PlateCarree", "Mercator", "TransverseMercator", "Miller",
    "LambertCylindrical",
    # pseudocylindrical
    "Robinson", "Mollweide", "Sinusoidal", "EqualEarth",
    "NaturalEarth", "NaturalEarth2",
    "EckertI", "EckertII", "EckertIII", "EckertIV", "EckertV", "EckertVI",
    # azimuthal
    "Orthographic", "Stereographic", "NorthPolarStereo", "SouthPolarStereo",
    "LambertAzimuthalEqualArea", "AzimuthalEquidistant", "Gnomonic",
    "NearsidePerspective",
    # conic
    "AlbersEqualArea", "LambertConformal", "EquidistantConic",
    # other
    "Aitoff", "Hammer", "WinkelTripel", "Bonne", "Polyconic",
    "VanDerGrinten", "Lagrange", "Times",
    "InterruptedGoodeHomolosine", "InterruptedMollweide",
    "InterruptedSinusoidal", "InterruptedBoggs",
    "RotatedPole",
    "PROJECTIONS", "get",
]

# Earth radius used to convert a satellite height in metres into the
# multiple-of-Earth-radii distance d3-geo's satellite projection expects.
_EARTH_RADIUS_M = 6_378_137.0


class CRS:
    """Base class for all projections.

    Subclasses set :attr:`tp_name` — the projection key terraplot resolves in
    the browser — and may override :meth:`to_js` to add parameters.
    """

    #: terraplot / d3-geo projection key
    tp_name: str = "equirectangular"
    #: True for the 3D globe rather than a flat projection
    is_globe: bool = False
    #: True for CRSs that only describe how coordinates are interpreted
    is_transform_only: bool = False

    def __init__(self, central_longitude: float = 0.0,
                 central_latitude: float = 0.0):
        self.central_longitude = float(central_longitude)
        self.central_latitude = float(central_latitude)

    # ── translation to the browser ────────────────────────────────────────────

    @property
    def center(self) -> tuple[float, float]:
        """``(central_longitude, central_latitude)`` — the map centre."""
        return (self.central_longitude, self.central_latitude)

    def to_js(self) -> dict[str, Any]:
        """Constructor options for terraplot's ``GeoMap``."""
        opts: dict[str, Any] = {
            "projection": self.tp_name,
            "center": [self.central_longitude, self.central_latitude],
        }
        return opts

    # ── plumbing ──────────────────────────────────────────────────────────────

    def _params(self) -> dict[str, Any]:
        """Constructor parameters, for repr and equality."""
        return {k: v for k, v in vars(self).items() if not k.startswith("_")}

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self._params().items())
        return f"{type(self).__name__}({args})"

    def __eq__(self, other: object) -> bool:
        return (type(self) is type(other)
                and self._params() == other._params())   # type: ignore[attr-defined]

    def __hash__(self) -> int:
        return hash((type(self).__name__, tuple(sorted(
            (k, tuple(v) if isinstance(v, (list, tuple)) else v)
            for k, v in self._params().items()
        ))))


# ── transform-only / non-flat ────────────────────────────────────────────────

class Geodetic(CRS):
    """Spherical lon/lat coordinates, interpolated along great circles.

    Only meaningful as ``transform=`` on :meth:`~pyterraplot.Axes.plot`,
    where it makes line segments follow the shortest path over the sphere
    instead of a straight line in lon/lat.
    """

    is_transform_only = True

    def __init__(self):
        super().__init__()

    def to_js(self) -> dict[str, Any]:
        raise TypeError(
            "Geodetic() is a transform, not a map projection — pass it as "
            "transform=ccrs.Geodetic() to plot(), not as projection= to Axes()."
        )


class Globe3D(CRS):
    """The interactive 3D globe (terraplot ``GeoSphere``) instead of a flat map."""

    tp_name = "globe3d"
    is_globe = True

    def __init__(self):
        super().__init__()

    def to_js(self) -> dict[str, Any]:
        raise TypeError("Globe3D() renders via GeoSphere, not GeoMap options.")


# ── cylindrical ──────────────────────────────────────────────────────────────

class PlateCarree(CRS):
    """Equidistant cylindrical (equirectangular). The lon/lat identity map."""

    tp_name = "equirectangular"

    def __init__(self, central_longitude: float = 0.0):
        super().__init__(central_longitude, 0.0)


class Mercator(CRS):
    """Mercator, the conformal cylindrical projection.

    ``min_latitude`` / ``max_latitude`` are accepted for cartopy parity and
    used as the default extent, since Mercator diverges at the poles.
    """

    tp_name = "mercator"

    def __init__(self, central_longitude: float = 0.0,
                 min_latitude: float = -80.0, max_latitude: float = 84.0):
        super().__init__(central_longitude, 0.0)
        self.min_latitude = float(min_latitude)
        self.max_latitude = float(max_latitude)

    @property
    def default_extent(self) -> tuple[float, float, float, float]:
        lon = self.central_longitude
        return (lon - 180.0, lon + 180.0, self.min_latitude, self.max_latitude)


class TransverseMercator(CRS):
    """Transverse Mercator — conformal, true along a central meridian."""

    tp_name = "transversemercator"

    def __init__(self, central_longitude: float = 0.0,
                 central_latitude: float = 0.0):
        super().__init__(central_longitude, central_latitude)


class Miller(CRS):
    """Miller cylindrical — a compromise that tames Mercator's polar stretch."""

    tp_name = "miller"

    def __init__(self, central_longitude: float = 0.0):
        super().__init__(central_longitude, 0.0)


class LambertCylindrical(CRS):
    """Lambert cylindrical equal-area."""

    tp_name = "cylindricalequalarea"

    def __init__(self, central_longitude: float = 0.0):
        super().__init__(central_longitude, 0.0)


# ── pseudocylindrical ────────────────────────────────────────────────────────

class Robinson(CRS):
    """Robinson — the classic compromise world projection."""

    tp_name = "robinson"

    def __init__(self, central_longitude: float = 0.0):
        super().__init__(central_longitude, 0.0)


class Mollweide(CRS):
    """Mollweide — equal-area, elliptical."""

    tp_name = "mollweide"

    def __init__(self, central_longitude: float = 0.0):
        super().__init__(central_longitude, 0.0)


class Sinusoidal(CRS):
    """Sinusoidal — equal-area, true along every parallel."""

    tp_name = "sinusoidal"

    def __init__(self, central_longitude: float = 0.0):
        super().__init__(central_longitude, 0.0)


class EqualEarth(CRS):
    """Equal Earth — equal-area with Robinson-like proportions."""

    tp_name = "equalearth"

    def __init__(self, central_longitude: float = 0.0):
        super().__init__(central_longitude, 0.0)


class NaturalEarth(CRS):
    """Natural Earth I — a rounded compromise world projection."""

    tp_name = "naturalearth1"

    def __init__(self, central_longitude: float = 0.0):
        super().__init__(central_longitude, 0.0)


class NaturalEarth2(CRS):
    """Natural Earth II — more strongly rounded meridians than Natural Earth I."""

    tp_name = "naturalearth2"

    def __init__(self, central_longitude: float = 0.0):
        super().__init__(central_longitude, 0.0)


class _Eckert(CRS):
    def __init__(self, central_longitude: float = 0.0):
        super().__init__(central_longitude, 0.0)


class EckertI(_Eckert):
    """Eckert I — straight-line pseudocylindrical."""
    tp_name = "eckert1"


class EckertII(_Eckert):
    """Eckert II — equal-area, straight meridians."""
    tp_name = "eckert2"


class EckertIII(_Eckert):
    """Eckert III — elliptical meridians, pole lines."""
    tp_name = "eckert3"


class EckertIV(_Eckert):
    """Eckert IV — equal-area with elliptical meridians."""
    tp_name = "eckert4"


class EckertV(_Eckert):
    """Eckert V — sinusoidal/plate carrée average."""
    tp_name = "eckert5"


class EckertVI(_Eckert):
    """Eckert VI — equal-area, sinusoidal meridians."""
    tp_name = "eckert6"


# ── azimuthal ────────────────────────────────────────────────────────────────

class Orthographic(CRS):
    """Orthographic — the view of a globe from infinite distance."""

    tp_name = "orthographic"

    def __init__(self, central_longitude: float = 0.0,
                 central_latitude: float = 0.0):
        super().__init__(central_longitude, central_latitude)


class Stereographic(CRS):
    """Stereographic — conformal azimuthal.

    ``true_scale_latitude`` is accepted for cartopy parity; terraplot renders
    on a sphere with d3's fixed scaling, so it does not alter the geometry.
    """

    tp_name = "stereographic"

    def __init__(self, central_latitude: float = 0.0,
                 central_longitude: float = 0.0,
                 true_scale_latitude: float | None = None):
        super().__init__(central_longitude, central_latitude)
        self.true_scale_latitude = true_scale_latitude


class NorthPolarStereo(Stereographic):
    """Stereographic centred on the North Pole."""

    def __init__(self, central_longitude: float = 0.0,
                 true_scale_latitude: float | None = None):
        super().__init__(central_latitude=90.0,
                         central_longitude=central_longitude,
                         true_scale_latitude=true_scale_latitude)

    @property
    def default_extent(self) -> tuple[float, float, float, float]:
        return (-180.0, 180.0, 45.0, 90.0)


class SouthPolarStereo(Stereographic):
    """Stereographic centred on the South Pole."""

    def __init__(self, central_longitude: float = 0.0,
                 true_scale_latitude: float | None = None):
        super().__init__(central_latitude=-90.0,
                         central_longitude=central_longitude,
                         true_scale_latitude=true_scale_latitude)

    @property
    def default_extent(self) -> tuple[float, float, float, float]:
        return (-180.0, 180.0, -90.0, -45.0)


class LambertAzimuthalEqualArea(CRS):
    """Lambert azimuthal equal-area."""

    tp_name = "azimuthalequalarea"

    def __init__(self, central_longitude: float = 0.0,
                 central_latitude: float = 0.0):
        super().__init__(central_longitude, central_latitude)


class AzimuthalEquidistant(CRS):
    """Azimuthal equidistant — distances from the centre are true."""

    tp_name = "azimuthalequidistant"

    def __init__(self, central_longitude: float = 0.0,
                 central_latitude: float = 0.0):
        super().__init__(central_longitude, central_latitude)


class Gnomonic(CRS):
    """Gnomonic — great circles project to straight lines."""

    tp_name = "gnomonic"

    def __init__(self, central_latitude: float = 0.0,
                 central_longitude: float = 0.0):
        super().__init__(central_longitude, central_latitude)


class NearsidePerspective(CRS):
    """The view from a satellite at a finite height above the surface.

    ``satellite_height`` is in metres above the surface, as in cartopy; it is
    converted to the multiple-of-Earth-radii distance d3-geo expects.
    """

    tp_name = "nearsideperspective"

    def __init__(self, central_longitude: float = 0.0,
                 central_latitude: float = 0.0,
                 satellite_height: float = 35_785_831.0):
        super().__init__(central_longitude, central_latitude)
        self.satellite_height = float(satellite_height)

    def to_js(self) -> dict[str, Any]:
        opts = super().to_js()
        opts["distance"] = 1.0 + self.satellite_height / _EARTH_RADIUS_M
        return opts


# ── conic ────────────────────────────────────────────────────────────────────

class _Conic(CRS):
    """Shared behaviour for the conic family: two standard parallels."""

    def __init__(self, central_longitude: float, central_latitude: float,
                 standard_parallels: tuple[float, float]):
        super().__init__(central_longitude, central_latitude)
        if len(standard_parallels) != 2:
            raise ValueError(
                "standard_parallels must be a 2-tuple (lat1, lat2); "
                f"got {standard_parallels!r}"
            )
        self.standard_parallels = (float(standard_parallels[0]),
                                   float(standard_parallels[1]))

    def to_js(self) -> dict[str, Any]:
        opts = super().to_js()
        opts["parallels"] = list(self.standard_parallels)
        return opts


class AlbersEqualArea(_Conic):
    """Albers equal-area conic."""

    tp_name = "conicequalarea"

    def __init__(self, central_longitude: float = 0.0,
                 central_latitude: float = 0.0,
                 standard_parallels: tuple[float, float] = (20.0, 50.0)):
        super().__init__(central_longitude, central_latitude, standard_parallels)


class LambertConformal(_Conic):
    """Lambert conformal conic — the standard for mid-latitude weather charts."""

    tp_name = "conicconformal"

    def __init__(self, central_longitude: float = -96.0,
                 central_latitude: float = 39.0,
                 standard_parallels: tuple[float, float] = (33.0, 45.0)):
        super().__init__(central_longitude, central_latitude, standard_parallels)


class EquidistantConic(_Conic):
    """Equidistant conic — true scale along the standard parallels."""

    tp_name = "conicequidistant"

    def __init__(self, central_longitude: float = 0.0,
                 central_latitude: float = 0.0,
                 standard_parallels: tuple[float, float] = (20.0, 50.0)):
        super().__init__(central_longitude, central_latitude, standard_parallels)


# ── other world projections ──────────────────────────────────────────────────

class Aitoff(CRS):
    """Aitoff — modified azimuthal equidistant, 2:1 world."""

    tp_name = "aitoff"

    def __init__(self, central_longitude: float = 0.0):
        super().__init__(central_longitude, 0.0)


class Hammer(CRS):
    """Hammer (Hammer–Aitoff) — equal-area, elliptical."""

    tp_name = "hammer"

    def __init__(self, central_longitude: float = 0.0):
        super().__init__(central_longitude, 0.0)


class WinkelTripel(CRS):
    """Winkel tripel — the National Geographic compromise projection."""

    tp_name = "winkeltripel"

    def __init__(self, central_longitude: float = 0.0):
        super().__init__(central_longitude, 0.0)


class Bonne(CRS):
    """Bonne — heart-shaped equal-area pseudoconic."""

    tp_name = "bonne"

    def __init__(self, central_latitude: float = 45.0,
                 central_longitude: float = 0.0):
        super().__init__(central_longitude, central_latitude)


class Polyconic(CRS):
    """American polyconic."""

    tp_name = "polyconic"

    def __init__(self, central_longitude: float = 0.0,
                 central_latitude: float = 0.0):
        super().__init__(central_longitude, central_latitude)


class VanDerGrinten(CRS):
    """Van der Grinten — the world in a circle."""

    tp_name = "vandergrinten"

    def __init__(self, central_longitude: float = 0.0):
        super().__init__(central_longitude, 0.0)


class Lagrange(CRS):
    """Lagrange — conformal, the world in a circle."""

    tp_name = "lagrange"

    def __init__(self, central_longitude: float = 0.0):
        super().__init__(central_longitude, 0.0)


class Times(CRS):
    """The Times projection — Moir's compromise for world atlases."""

    tp_name = "times"

    def __init__(self, central_longitude: float = 0.0):
        super().__init__(central_longitude, 0.0)


# ── interrupted ──────────────────────────────────────────────────────────────

class InterruptedGoodeHomolosine(CRS):
    """Goode homolosine — sinusoidal near the equator, Mollweide poleward,
    interrupted across the oceans."""

    tp_name = "interruptedgoodehomolosine"

    def __init__(self, central_longitude: float = 0.0):
        super().__init__(central_longitude, 0.0)


class InterruptedMollweide(CRS):
    """Mollweide, interrupted into ocean-avoiding lobes."""

    tp_name = "interruptedmollweide"

    def __init__(self, central_longitude: float = 0.0):
        super().__init__(central_longitude, 0.0)


class InterruptedSinusoidal(CRS):
    """Sinusoidal, interrupted into lobes."""

    tp_name = "interruptedsinusoidal"

    def __init__(self, central_longitude: float = 0.0):
        super().__init__(central_longitude, 0.0)


class InterruptedBoggs(CRS):
    """Boggs eumorphic, interrupted."""

    tp_name = "interruptedboggs"

    def __init__(self, central_longitude: float = 0.0):
        super().__init__(central_longitude, 0.0)


# ── rotated pole ─────────────────────────────────────────────────────────────

class RotatedPole(CRS):
    """A lat/lon grid whose north pole sits at a chosen geographic point.

    The native grid of many regional climate models (CORDEX, COSMO, HIRHAM).
    ``pole_longitude`` / ``pole_latitude`` give the geographic position of the
    rotated system's north pole, as in cartopy.

    ``central_rotated_longitude`` is not supported: it is a spin about the
    rotated pole applied *after* the other two rotations, which d3-geo's
    three-angle ``rotate()`` composition cannot express.
    """

    tp_name = "equirectangular"

    def __init__(self, pole_longitude: float = 0.0, pole_latitude: float = 90.0,
                 central_rotated_longitude: float = 0.0,
                 projection: str = "equirectangular"):
        super().__init__(0.0, 0.0)
        if central_rotated_longitude:
            raise NotImplementedError(
                "central_rotated_longitude is not supported — it is a spin "
                "about the rotated pole applied after the other rotations, "
                "which d3-geo's rotate([lambda, phi, gamma]) cannot express. "
                "Rotate the data instead, or open an issue if you need it."
            )
        self.pole_longitude = float(pole_longitude)
        self.pole_latitude = float(pole_latitude)
        self.central_rotated_longitude = 0.0
        self.projection = projection
        self.tp_name = projection

    def to_js(self) -> dict[str, Any]:
        # d3 rotate([-pole_lon, 90 - pole_lat, 0]) carries the geographic point
        # (pole_lon, pole_lat) onto the projection's north pole.
        return {
            "projection": self.tp_name,
            "rotate": [-self.pole_longitude, 90.0 - self.pole_latitude, 0.0],
        }


# ── registry / lookup ────────────────────────────────────────────────────────

#: Every concrete projection class, keyed by name.
PROJECTIONS: dict[str, type[CRS]] = {
    name: obj
    for name, obj in list(globals().items())
    if isinstance(obj, type) and issubclass(obj, CRS)
    and obj not in (CRS, _Conic, _Eckert)
}


def get(name: str, **kwargs) -> CRS:
    """Look up a projection class by name and instantiate it.

    Accepts the class name (``"PlateCarree"``) or the terraplot projection key
    (``"equirectangular"``), case- and separator-insensitively::

        >>> get("north_polar_stereo", central_longitude=-100)
        NorthPolarStereo(central_longitude=-100.0, central_latitude=90.0, true_scale_latitude=None)
    """
    key = name.lower().replace("_", "").replace("-", "").replace(" ", "")
    for cls_name, cls in PROJECTIONS.items():
        if cls_name.lower() == key:
            return cls(**kwargs)
    # Fall back to the terraplot projection key. Geodetic, Globe3D and
    # RotatedPole inherit or reuse a tp_name they do not own, so they never win
    # this pass — "equirectangular" must resolve to PlateCarree.
    for cls_name, cls in PROJECTIONS.items():
        if cls.is_transform_only or cls.is_globe or cls_name == "RotatedPole":
            continue
        if getattr(cls, "tp_name", None) == key:
            return cls(**kwargs)
    raise ValueError(
        f"Unknown projection {name!r}. Available: "
        + ", ".join(sorted(PROJECTIONS))
    )
