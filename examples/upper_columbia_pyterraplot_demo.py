"""
Upper Columbia Transboundary 3D Globe with Interactive Tooltips.

Demonstrates pyterraplot's new Globe3D capabilities:
  - 3D Globe visualization with deep zoom
  - Green continental coastlines & international boundaries
  - 3D vertical needle gauging stations with rich hover metadata tooltips
  - Direct basin boundary plotting
"""
import xarray as xr
import numpy as np
from parflow.tools.io import read_pfb
import pyterraplot as tp
import pyterraplot.crs as ccrs

# 1. Load ParFlow DEM
model_dir = '/home/guido/hydrogeology/models/upper_columbia_transboundary_1km'
dem_arr = read_pfb(f'{model_dir}/dem.pfb')[0]
ny, nx = dem_arr.shape

# 180x180 spatial sampling for smooth 3D globe rendering
stride_y = max(1, ny // 180)
stride_x = max(1, nx // 180)
dem_sub = dem_arr[::stride_y, ::stride_x]
sub_ny, sub_nx = dem_sub.shape

lats = np.linspace(47.5, 52.5, sub_ny)
lons = np.linspace(-121.0, -113.5, sub_nx)

da_dem = xr.DataArray(
    dem_sub,
    coords={'latitude': lats, 'longitude': lons},
    dims=['latitude', 'longitude'],
    name='elevation',
    attrs={'long_name': 'Surface Elevation', 'units': 'm ASL'}
)

# 2. Setup 3D Globe Figure
fig, ax = tp.subplots(
    figsize=(12, 10),
    subplot_kw={
        'projection': ccrs.Globe3D(),
        'spin': False,
        'earth_surface': 'shaded_relief',
    }
)

# 3. Render Elevation Contours
ax.contourf(da_dem, cmap='terrain', levels=16, alpha=0.92)

# 4. Add Green Continental Coastlines & Borders
ax.coastlines(color='#22c55e', linewidth=2.0)
ax.borders(color='#4ade80', linewidth=1.5)

# 5. Delineate Upper Columbia Basin Extent via ax.plot
box_lons = [-121.0, -113.5, -113.5, -121.0, -121.0]
box_lats = [47.5, 47.5, 52.5, 52.5, 47.5]
ax.plot(box_lons, box_lats, color='#38bdf8', linewidth=2.5, label='Upper Columbia Basin')
ax.set_extent([-125.0, -110.0, 45.0, 55.0])

# 6. Add Transboundary Sentinel Gauging Stations with Rich Tooltips
stations = [
    {
        'name': 'Birchbank',
        'agency': 'WSC (08NE049)',
        'lat': 49.18,
        'lon': -117.72,
        'flow': '1,940 m³/s',
        'type': 'Mainstem Outlet'
    },
    {
        'name': 'Fort Steele',
        'agency': 'WSC (08NH005)',
        'lat': 49.62,
        'lon': -115.75,
        'flow': '245 m³/s',
        'type': 'Kootenay Headwaters'
    },
    {
        'name': 'Waneta',
        'agency': 'WSC (08NL007)',
        'lat': 49.00,
        'lon': -117.62,
        'flow': '750 m³/s',
        'type': 'Pend d\'Oreille Confluence'
    },
    {
        'name': 'Libby Dam',
        'agency': 'USGS (12301933)',
        'lat': 48.40,
        'lon': -115.31,
        'flow': '340 m³/s',
        'type': 'Lake Koocanusa Regulated Outlet'
    },
    {
        'name': 'Bonners Ferry',
        'agency': 'USGS (12305000)',
        'lat': 48.70,
        'lon': -116.31,
        'flow': '450 m³/s',
        'type': 'Kootenai Valley US Inflow'
    },
    {
        'name': 'Grand Coulee',
        'agency': 'USGS (12436500)',
        'lat': 47.96,
        'lon': -118.98,
        'flow': '3,100 m³/s',
        'type': 'Lake Roosevelt Basin Outlet'
    }
]

st_lons = [s['lon'] for s in stations]
st_lats = [s['lat'] for s in stations]
st_tooltips = [
    {
        'name': s['name'],
        'agency': s['agency'],
        'type': s['type'],
        'mean_flow': s['flow'],
        'coordinates': f"{s['lat']}°N, {abs(s['lon'])}°W"
    }
    for s in stations
]

ax.scatter(
    st_lons,
    st_lats,
    color='#ef4444',
    style_3d='vertical_line',
    height=2.8,
    tooltips=st_tooltips,
    label='Sentinel Gauging Stations'
)

# 7. Focus Camera & Save
ax.set_center(lon=-117.25, lat=50.0)
out_html = '/home/guido/hydrogeology/docs/globe/pyterraplot_3d_globe_demo.html'
fig.savefig(out_html)
print(f"✓ Successfully generated pyterraplot 3D Globe demo: {out_html}")
