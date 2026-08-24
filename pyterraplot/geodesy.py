"""
Spherical geometry and vector-field helpers.

Small, dependency-light routines the Axes primitives build on:

* :func:`geodesic_circle` — the locus of points a fixed great-circle distance
  from a centre, used by ``Axes.tissot``.
* :func:`great_circle` — densify a segment along the shortest spherical path.
  ``Axes.plot(..., transform=Geodetic())`` leaves this to the browser, which
  reprojects on resize; this is the equivalent for coordinates you want
  densified in Python before handing them anywhere else.
* :func:`streamlines` — trace streamlines through a lon/lat vector field,
  used by ``Axes.streamplot``.

Everything works on a sphere of radius :data:`EARTH_RADIUS_M`; no ellipsoid is
modelled, matching what terraplot draws in the browser.
"""
from __future__ import annotations

import numpy as np

__all__ = ["EARTH_RADIUS_M", "geodesic_circle", "great_circle", "streamlines"]

#: Mean Earth radius in metres.
EARTH_RADIUS_M = 6_371_008.8


def geodesic_circle(lon: float, lat: float, radius_m: float,
                    n_samples: int = 80) -> list[list[float]]:
    """Points at great-circle distance ``radius_m`` from ``(lon, lat)``.

    Returns a closed ring of ``[lon, lat]`` pairs — the first point is repeated
    at the end, so the result drops straight into a GeoJSON polygon.

    The circle is built by walking every azimuth and applying the direct
    geodesic formula on a sphere::

        sin(lat2) = sin(lat1)cos(d) + cos(lat1)sin(d)cos(theta)

    where ``d = radius_m / R`` is the angular distance and ``theta`` the
    bearing.
    """
    d = radius_m / EARTH_RADIUS_M
    theta = np.linspace(0.0, 2.0 * np.pi, n_samples, endpoint=False)

    lat1 = np.deg2rad(lat)
    lon1 = np.deg2rad(lon)

    sin_lat2 = np.sin(lat1) * np.cos(d) + np.cos(lat1) * np.sin(d) * np.cos(theta)
    lat2 = np.arcsin(np.clip(sin_lat2, -1.0, 1.0))
    lon2 = lon1 + np.arctan2(
        np.sin(theta) * np.sin(d) * np.cos(lat1),
        np.cos(d) - np.sin(lat1) * sin_lat2,
    )

    lons = (np.rad2deg(lon2) + 180.0) % 360.0 - 180.0
    lats = np.rad2deg(lat2)
    ring = [[float(a), float(b)] for a, b in zip(lons, lats)]
    ring.append(ring[0])
    return ring


def great_circle(start: tuple[float, float], end: tuple[float, float],
                 n: int = 32) -> list[list[float]]:
    """Densify ``start`` → ``end`` along the great circle joining them.

    Uses spherical linear interpolation of the 3D unit vectors, which stays
    stable for near-antipodal endpoints where the angular formulation degrades.
    """
    p0 = _to_xyz(*start)
    p1 = _to_xyz(*end)
    omega = float(np.arccos(np.clip(np.dot(p0, p1), -1.0, 1.0)))
    if omega < 1e-12:
        return [list(start), list(end)]

    t = np.linspace(0.0, 1.0, max(2, n))
    s0 = np.sin((1.0 - t) * omega) / np.sin(omega)
    s1 = np.sin(t * omega) / np.sin(omega)
    pts = s0[:, None] * p0[None, :] + s1[:, None] * p1[None, :]
    return [_to_lonlat(p) for p in pts]


def streamlines(
    lons: np.ndarray,
    lats: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    *,
    density: float = 1.0,
    max_steps: int = 400,
    step_deg: float = 0.5,
    min_length: int = 8,
) -> list[list[list[float]]]:
    """Trace streamlines through a regular lon/lat vector field.

    Seeds are laid on a grid whose spacing follows ``density`` (matplotlib's
    convention: 1.0 is the default coverage, larger is denser), then integrated
    both forward and backward with RK4 in degrees of lon/lat. ``u`` is scaled by
    ``1/cos(lat)`` so a constant zonal wind traces a constant-latitude line
    instead of curving, and integration stops when a line leaves the grid,
    stalls, or re-enters a cell an existing line already occupies.

    Parameters
    ----------
    lons, lats : 1D coordinate arrays (ascending or descending)
    u, v : 2D arrays shaped ``(len(lats), len(lons))``
    density : seed density multiplier
    max_steps : cap on integration steps per direction
    step_deg : integration step length, in degrees of arc
    min_length : discard traces shorter than this many points

    Returns
    -------
    list of polylines, each a list of ``[lon, lat]`` pairs.
    """
    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)

    # Work on ascending axes so the interpolator stays simple; flip the field
    # to match rather than sorting inside the inner loop.
    if lats[0] > lats[-1]:
        lats = lats[::-1]
        u = u[::-1, :]
        v = v[::-1, :]
    if lons[0] > lons[-1]:
        lons = lons[::-1]
        u = u[:, ::-1]
        v = v[:, ::-1]

    lon0, lon1 = float(lons[0]), float(lons[-1])
    lat0, lat1 = float(lats[0]), float(lats[-1])
    cyclic = (lon1 - lon0) >= 359.0

    # Normalise magnitude so the step length is a true arc length in degrees.
    # An all-NaN or all-zero field has nothing to trace; check before nanmax so
    # it does not warn about an empty slice.
    speed = np.hypot(u, v)
    finite = np.isfinite(speed)
    if not finite.any():
        return []
    scale = float(speed[finite].max())
    if scale == 0:
        return []

    # Occupancy grid: at most one streamline per cell keeps the picture legible,
    # the same trick matplotlib's streamplot uses.
    nx = max(4, int(30 * density))
    ny = max(4, int(15 * density))
    occupied = np.zeros((ny, nx), dtype=bool)

    def sample(lon: float, lat: float):
        if cyclic:
            lon = lon0 + (lon - lon0) % 360.0
        elif not (lon0 <= lon <= lon1):
            return None
        if not (lat0 <= lat <= lat1):
            return None
        # Bilinear interpolation on the regular grid.
        fi = np.interp(lon, lons, np.arange(lons.size))
        fj = np.interp(lat, lats, np.arange(lats.size))
        i0, j0 = int(np.floor(fi)), int(np.floor(fj))
        i1 = min(i0 + 1, lons.size - 1)
        j1 = min(j0 + 1, lats.size - 1)
        ti, tj = fi - i0, fj - j0
        def bl(a):
            return ((1 - ti) * (1 - tj) * a[j0, i0] + ti * (1 - tj) * a[j0, i1]
                    + (1 - ti) * tj * a[j1, i0] + ti * tj * a[j1, i1])
        uu, vv = bl(u), bl(v)
        if not (np.isfinite(uu) and np.isfinite(vv)):
            return None
        mag = float(np.hypot(uu, vv))
        if mag < 1e-12:
            return None
        # Degrees of lon per degree of lat: zonal steps shrink toward the poles.
        coslat = max(np.cos(np.deg2rad(lat)), 1e-3)
        return (float(uu) / mag / coslat, float(vv) / mag)

    def cell(lon: float, lat: float):
        gx = int((lon - lon0) / max(lon1 - lon0, 1e-9) * (nx - 1))
        gy = int((lat - lat0) / max(lat1 - lat0, 1e-9) * (ny - 1))
        if 0 <= gx < nx and 0 <= gy < ny:
            return gy, gx
        return None

    def integrate(lon: float, lat: float, sign: int):
        pts = [[lon, lat]]
        for _ in range(max_steps):
            k1 = sample(lon, lat)
            if k1 is None:
                break
            h = sign * step_deg
            k2 = sample(lon + 0.5 * h * k1[0], lat + 0.5 * h * k1[1])
            if k2 is None:
                break
            k3 = sample(lon + 0.5 * h * k2[0], lat + 0.5 * h * k2[1])
            if k3 is None:
                break
            k4 = sample(lon + h * k3[0], lat + h * k3[1])
            if k4 is None:
                break
            lon += h * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6.0
            lat += h * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6.0
            if cyclic:
                lon = lon0 + (lon - lon0) % 360.0
            elif not (lon0 <= lon <= lon1):
                break
            if not (lat0 <= lat <= lat1):
                break
            c = cell(lon, lat)
            if c is None:
                break
            pts.append([lon, lat])
        return pts

    out: list[list[list[float]]] = []
    seeds_x = np.linspace(lon0, lon1, nx, endpoint=not cyclic)
    seeds_y = np.linspace(lat0, lat1, ny)
    for sy in seeds_y:
        for sx in seeds_x:
            c = cell(float(sx), float(sy))
            if c is None or occupied[c]:
                continue
            fwd = integrate(float(sx), float(sy), +1)
            bwd = integrate(float(sx), float(sy), -1)
            line = bwd[::-1][:-1] + fwd
            if len(line) < min_length:
                continue
            for p in line:
                cc = cell(p[0], p[1])
                if cc is not None:
                    occupied[cc] = True
            # Split where a cyclic line wraps the dateline so the browser
            # doesn't draw a spurious segment straight across the map.
            out.extend(_split_wrap(line))
    return out


def _split_wrap(line: list[list[float]], threshold: float = 180.0):
    """Break a polyline wherever consecutive longitudes jump the dateline."""
    runs, current = [], [line[0]]
    for prev, pt in zip(line, line[1:]):
        if abs(pt[0] - prev[0]) > threshold:
            if len(current) > 1:
                runs.append(current)
            current = [pt]
        else:
            current.append(pt)
    if len(current) > 1:
        runs.append(current)
    return runs


def _to_xyz(lon: float, lat: float) -> np.ndarray:
    rlon, rlat = np.deg2rad(lon), np.deg2rad(lat)
    return np.array([np.cos(rlat) * np.cos(rlon),
                     np.cos(rlat) * np.sin(rlon),
                     np.sin(rlat)])


def _to_lonlat(p: np.ndarray) -> list[float]:
    p = p / np.linalg.norm(p)
    return [float(np.rad2deg(np.arctan2(p[1], p[0]))),
            float(np.rad2deg(np.arcsin(np.clip(p[2], -1.0, 1.0))))]
