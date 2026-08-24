"""
pyterraplot cartopy demo
========================

The cartopy-style surface end to end: projection objects, map features,
gridlines, regional extents, vector overlays, and matplotlib-style subplots.

Run: python examples/cartopy_demo.py
Writes a set of self-contained .html files into examples/output/.
"""
from pathlib import Path

import numpy as np
import xarray as xr

import pyterraplot as tp
import pyterraplot.crs as ccrs
import pyterraplot.feature as cfeature

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

# ── Synthetic global fields ──────────────────────────────────────────────────

nlat, nlon = 73, 144
lats = np.linspace(90, -90, nlat)
lons = np.linspace(-180, 177.5, nlon)
LON, LAT = np.meshgrid(lons, lats)

t2m = xr.DataArray(
    (30 * np.cos(np.radians(LAT))
     - 15
     + 6 * np.sin(np.radians(3 * LON)) * np.cos(np.radians(2 * LAT))).astype(np.float32),
    dims=["lat", "lon"], coords={"lat": lats, "lon": lons}, name="t2m",
    attrs={"units": "degC", "long_name": "2 m air temperature"},
)

# A zonal jet with a wavy meridional component — enough structure for arrows,
# barbs and streamlines to show something recognisable.
u10 = xr.DataArray(
    (25 * np.exp(-((np.abs(LAT) - 45) / 12) ** 2)
     + 5 * np.sin(np.radians(4 * LON))).astype(np.float32),
    dims=["lat", "lon"], coords={"lat": lats, "lon": lons}, name="u10",
    attrs={"units": "kt", "long_name": "10 m eastward wind"},
)
v10 = xr.DataArray(
    (8 * np.sin(np.radians(3 * LON)) * np.cos(np.radians(LAT))).astype(np.float32),
    dims=["lat", "lon"], coords={"lat": lats, "lon": lons}, name="v10",
    attrs={"units": "kt", "long_name": "10 m northward wind"},
)


# ── 1. The classic cartopy opening move ──────────────────────────────────────

ax = tp.Axes(projection=ccrs.Robinson(central_longitude=-100))
ax.contourf(t2m, levels=16, cmap="RdBu_r", vmin=-30, vmax=30)
ax.contour(t2m, levels=16, color="rgba(0,0,0,0.35)", linewidth=1)
ax.coastlines(resolution="50m", color="#e8f4ff", linewidth=0.9)
ax.gridlines(draw_labels=True, xstep=60, ystep=30, linestyle="--")
ax.colorbar(orientation="horizontal", position="bottom")
ax.set_title("2 m air temperature — Robinson")
ax.to_html(OUT / "01_robinson.html")
print("wrote 01_robinson.html")


# ── 2. Regional extent, filled features, city labels ─────────────────────────

ax = tp.Axes(projection=ccrs.LambertConformal(central_longitude=-96,
                                              central_latitude=50,
                                              standard_parallels=(40, 60)),
             earth_surface="none")
ax.add_feature(cfeature.OCEAN, facecolor="#0d1b2e")
ax.add_feature(cfeature.LAND, facecolor="#2a2a24")
ax.pcolormesh(t2m, cmap="magma", alpha=0.75)
ax.coastlines(resolution="50m")
ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.8)
ax.states(resolution="50m")
ax.set_extent([-141, -52, 41, 84])
ax.gridlines(draw_labels=True, xstep=20, ystep=10)
ax.marker(53.5, -113.5, label="Edmonton")
ax.set_title("Canada — Lambert conformal conic")
ax.to_html(OUT / "02_lambert_regional.html")
print("wrote 02_lambert_regional.html")


# ── 3. Polar stereographic, with its own default extent ──────────────────────

ax = tp.Axes(projection=ccrs.NorthPolarStereo(central_longitude=-100))
ax.contourf(t2m, levels=14, cmap="cividis")
ax.coastlines(color="#7dd3fc")
ax.gridlines(draw_labels=True, xstep=30, ystep=10, dms=False)
ax.set_title("Arctic — north polar stereographic")
ax.to_html(OUT / "03_polar.html")
print("wrote 03_polar.html")


# ── 4. Vectors: quiver, barbs and streamlines over one field ─────────────────

for name, add_vectors in [
    ("quiver", lambda a: a.quiver(u10, v10, scale=26, color="rgba(255,255,255,0.8)")),
    ("barbs", lambda a: a.barbs(u10, v10, length=9, density=6)),
    ("stream", lambda a: a.streamplot(u10, v10, density=1.4,
                                      color="rgba(220,240,255,0.85)")),
]:
    ax = tp.Axes(projection=ccrs.PlateCarree(), earth_surface="none")
    ax.pcolormesh(np.hypot(u10, v10).rename("wspd").assign_attrs(
        units="kt", long_name="10 m wind speed"), cmap="viridis", alpha=0.8)
    add_vectors(ax)
    ax.coastlines(color="rgba(255,255,255,0.55)", linewidth=0.8)
    ax.gridlines(xstep=30, ystep=30)
    ax.set_title(f"10 m wind — {name}")
    ax.to_html(OUT / f"04_{name}.html")
    print(f"wrote 04_{name}.html")


# ── 5. Points, great-circle tracks and labels ────────────────────────────────

cities = {
    "Vancouver": (-123.1, 49.3), "Reykjavík": (-21.9, 64.1),
    "Nairobi": (36.8, -1.3), "Sydney": (151.2, -33.9), "Tokyo": (139.7, 35.7),
}
ax = tp.Axes(projection=ccrs.NaturalEarth(), earth_surface="none")
ax.add_feature(cfeature.LAND, facecolor="#26262a")
ax.coastlines(color="rgba(180,200,230,0.7)", linewidth=0.7)
ax.gridlines(xstep=30, ystep=30, color="rgba(255,255,255,0.12)")

lon_list = [c[0] for c in cities.values()]
lat_list = [c[1] for c in cities.values()]
ax.scatter(lon_list, lat_list, color="#fbbf24", s=5, edgecolor="#000",
           edgewidth=1, label="cities")
for name, (lon, lat) in cities.items():
    ax.text(lon, lat, name, dy=-12, fontsize=11)

# Great circles vs. straight lon/lat lines — the difference is the point.
ax.plot([-123.1, 139.7], [49.3, 35.7], transform=ccrs.Geodetic(),
        color="#38bdf8", linewidth=2, label="great circle")
ax.plot([-123.1, 139.7], [49.3, 35.7], transform=ccrs.PlateCarree(),
        color="#f87171", linewidth=2, linestyle="--", label="straight in lon/lat")
ax.legend(loc="lower left", title="Vancouver → Tokyo")
ax.set_title("Points, labels and geodesics")
ax.to_html(OUT / "05_points_and_tracks.html")
print("wrote 05_points_and_tracks.html")


# ── 6. Tissot's indicatrices: distortion, projection by projection ───────────

# Each panel needs its own projection, so build them with add_subplot rather
# than subplots(), which applies one projection to the whole grid.
fig = tp.figure(figsize=(16, 9))
for i, crs in enumerate([ccrs.PlateCarree(), ccrs.Mercator(),
                         ccrs.Mollweide(), ccrs.Robinson()], start=1):
    ax = fig.add_subplot(2, 2, i, projection=crs, earth_surface="none")
    ax.add_feature(cfeature.LAND, facecolor="#2b2b30")
    ax.coastlines(color="rgba(200,220,255,0.6)", linewidth=0.6)
    ax.tissot(rad_km=500)
    ax.gridlines(xstep=30, ystep=30)
    ax.set_title(type(crs).__name__)
fig.suptitle("Tissot's indicatrices — equal ground circles, unequal on the page")
fig.savefig(OUT / "06_tissot.html")
print("wrote 06_tissot.html")


# ── 7. Multi-panel comparison with one shared colour scale ───────────────────

anomalies = {
    "DJF": t2m - 4 * np.cos(np.radians(LAT)),
    "MAM": t2m - 1,
    "JJA": t2m + 4 * np.cos(np.radians(LAT)),
    "SON": t2m + 1,
}
fig, axes = tp.subplots(2, 2, figsize=(16, 9),
                        subplot_kw={"projection": ccrs.EqualEarth()})
for ax, (season, field) in zip(axes.flat, anomalies.items()):
    ax.contourf(field.rename("t2m").assign_attrs(units="degC",
                                                 long_name="2 m temperature"),
                levels=16, cmap="RdBu_r", vmin=-35, vmax=35)
    ax.coastlines(color="rgba(255,255,255,0.5)", linewidth=0.6)
    ax.gridlines(xstep=60, ystep=30, color="rgba(255,255,255,0.1)")
    ax.set_title(season)
fig.suptitle("Seasonal 2 m temperature — Equal Earth")
fig.colorbar(label="2 m temperature [°C]", ticks=7)
fig.savefig(OUT / "07_seasons.html")
print("wrote 07_seasons.html")


# ── 8. A rotated-pole grid, the native frame of many regional models ─────────

ax = tp.Axes(projection=ccrs.RotatedPole(pole_longitude=-162, pole_latitude=39.25))
ax.pcolormesh(t2m, cmap="turbo", alpha=0.85)
ax.coastlines(color="#fff", linewidth=0.8)
ax.gridlines(xstep=30, ystep=30)
ax.set_title("Rotated pole (CORDEX EUR-11 style)")
ax.to_html(OUT / "08_rotated_pole.html")
print("wrote 08_rotated_pole.html")

print(f"\nAll demos written to {OUT}/ — open any of them in a browser.")
