"""
Mapbox-style 3D Globe Render Demo
=================================

Renders a 3D globe with precipitation anomalies, country borders,
city labels, and a customized vertical colorbar with a square-root scale.

Run:
  python examples/mapbox_demo.py
"""
import numpy as np
import xarray as xr
import pyterraplot

# ── 1. Create a synthetic precipitation anomaly dataset ──────────────────
nlat, nlon = 181, 360
lats = np.linspace(90, -90, nlat)
lons = np.linspace(-180, 180, nlon)
LON, LAT = np.meshgrid(lons, lats)

# Generate a synthetic precipitation patch (heavy-tailed, 0 to 35 mm/day)
precip_data = (
    20 * np.exp(-((LAT - 5)**2 / 400 + (LON + 60)**2 / 1600)) +
    15 * np.exp(-((LAT + 15)**2 / 300 + (LON - 120)**2 / 1200))
).astype(np.float32)

# Add some fine-scale noise
rng = np.random.default_rng(42)
noise = rng.normal(0, 1.5, size=(nlat, nlon)).astype(np.float32)
precip_data = np.clip(precip_data + noise, 0, None)

precip = xr.DataArray(
    precip_data,
    dims=["lat", "lon"],
    coords={"lat": lats, "lon": lons},
    name="precip",
    attrs={"units": "mm/day", "long_name": "Precipitation Anomaly"},
)

# ── 2. Export HTML with Mapbox-style boundaries, labels, and colorbar ─────
out_html = "/tmp/pyterraplot_mapbox.html"

precip.tp.to_html(
    out_html,
    title="Precipitation Anomaly (mm/day)",
    cmap="YlGnBu",      # Nice green-blue sequential colormap for rain
    alpha=0.75,
    vmin=0,
    vmax=35,
    earth_surface="satellite",  # Dark space & satellite imagery background
    coastlines=True,
    coastline_color="#000000",   # Black coastlines
    coastline_width=1.5,
    borders=True,
    borders_color="#000000",     # Black country borders
    borders_width=0.8,
    cities=True,                 # Populated places/city markers & labels
    cities_color="#ffffff",      # White city labels (with automatic black text shadow)
    colorbar={
        "orientation": "vertical",
        "position": "right",
        "panel": True,            # Black translucent panel backing
        "background": "rgba(0,0,0,0.65)",
        "scale": "sqrt",          # Square root scale for precipitation distribution
        "ticks": [0, 1, 2, 5, 10, 20, 35],
        "label": "Precipitation Anomaly (mm/day)",
    }
)

print(f"Successfully generated Mapbox-style globe visualization: {out_html}")
print("Open this file in your browser to interact with the 3D globe.")
