#!/usr/bin/env python3
"""
examples/render_parflow_globe_example.py
========================================
Example script demonstrating pyterraplot CONTOURING features:
1. Banded Contour Fills (ax.contourf) with discrete interval steps.
2. Unfilled Iso-Contour Lines (ax.contour) for elevation & groundwater isobars.
3. 3D Globe & Orthographic Map Projection exports.

Usage:
    PYTHONPATH=/home/guido/pyterraplot python examples/render_parflow_globe_example.py
"""

import os
import sys
import struct
import numpy as np
import xarray as xr

# Ensure pyterraplot is on Python path
sys.path.insert(0, "/home/guido/pyterraplot")
import pyterraplot as tp

def read_pfb(filename: str) -> np.ndarray:
    """Zero-dependency pure-Python ParFlow binary (.pfb) reader."""
    try:
        from parflow.tools.io import read_pfb as pf_read_pfb
        return pf_read_pfb(filename)
    except ImportError:
        pass

    with open(filename, "rb") as f:
        x, y, z = struct.unpack(">ddd", f.read(24))
        nx, ny, nz = struct.unpack(">iii", f.read(12))
        dx, dy, dz = struct.unpack(">ddd", f.read(24))
        num_subgrids = struct.unpack(">i", f.read(4))[0]

        grid = np.zeros((nz, ny, nx), dtype=np.float64)
        for _ in range(num_subgrids):
            ix, iy, iz = struct.unpack(">iii", f.read(12))
            rx, ry, rz = struct.unpack(">iii", f.read(12))
            rx_p, ry_q, rz_r = struct.unpack(">iii", f.read(12))
            subgrid_data = np.fromfile(f, dtype=">f8", count=rx * ry * rz)
            subgrid_3d = subgrid_data.reshape((rz, ry, rx))
            grid[iz:iz+rz, iy:iy+ry, ix:ix+rx] = subgrid_3d

        return grid

def main():
    print("=" * 75)
    print("  PYTERRAPLOT PARFLOW CONTOURING (contourf & contour isolines)")
    print("=" * 75)

    model_dir = "/home/guido/hydrogeology/models/upper_columbia_transboundary_1km"
    output_dir = "docs/globe"
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n[1/4] Loading ParFlow PFB grids from: {model_dir}")
    dem_arr = read_pfb(os.path.join(model_dir, "dem.pfb"))[0]
    press_arr = read_pfb(os.path.join(model_dir, "press.init.pfb"))[-1]
    ny, nx = dem_arr.shape

    lats = np.linspace(47.5, 52.5, ny)
    lons = np.linspace(-121.0, -113.5, nx)

    da_dem = xr.DataArray(
        dem_arr.astype("float32"),
        coords={"latitude": lats, "longitude": lons},
        dims=["latitude", "longitude"],
        name="elevation",
        attrs={"long_name": "Transboundary DEM Elevation", "units": "m ASL"}
    )

    da_press = xr.DataArray(
        press_arr.astype("float32"),
        coords={"latitude": lats, "longitude": lons},
        dims=["latitude", "longitude"],
        name="pressure_head",
        attrs={"long_name": "Water Table Head (Equipotentials)", "units": "m"}
    )

    # 1. Orthographic 3D Projection with Banded Contours (contourf) + Isolines (contour)
    print("\n[2/4] Generating Contoured DEM Topography (contourf + contour)...")
    fig, ax = tp.subplots(
        projection=tp.Orthographic(central_longitude=-117.25, central_latitude=50.0),
        figsize=(12, 9)
    )
    ax.set_title("Canadian Upper Columbia 1km DEM (Contoured Bands & Isolines)")
    ax.coastlines(color="#334155", linewidth=1.2)
    ax.borders(color="#64748b", linewidth=1.2)
    ax.states(color="#94a3b8", linewidth=0.8)
    ax.rivers(color="#0284c7", linewidth=1.2)

    # Discrete banded contour intervals (16 levels)
    ax.contourf(da_dem, levels=16, cmap="terrain", vmin=100, vmax=3400, alpha=0.90)
    # Overlay distinct contour isolines
    ax.contour(da_dem, levels=12, color="rgba(255,255,255,0.7)", linewidth=1.2)

    dem_contour_html = os.path.join(output_dir, "upper_columbia_contoured_dem.html")
    ax.to_html(dem_contour_html)
    print(f"  ✓ Saved Contoured DEM: {dem_contour_html}")

    # 2. Groundwater Head Contours & Isobars
    print("\n[3/4] Generating Groundwater Head Isobars (contourf + contour)...")
    fig2, ax2 = tp.subplots(
        projection=tp.LambertConformal(central_longitude=-117.25, central_latitude=50.0),
        figsize=(12, 9)
    )
    ax2.set_title("Transboundary Groundwater Table Head Contours (Isobars)")
    ax2.coastlines()
    ax2.borders()
    ax2.states()
    ax2.rivers()

    ax2.contourf(da_press, levels=12, cmap="Blues", vmin=0.5, vmax=11.0, alpha=0.90)
    ax2.contour(da_press, levels=8, color="#0284c7", linewidth=1.8)

    press_contour_html = os.path.join(output_dir, "upper_columbia_contoured_heads.html")
    ax2.to_html(press_contour_html)
    print(f"  ✓ Saved Contoured Heads: {press_contour_html}")

    # 3. Export for envintel
    print("\n[4/4] Packing binary contour field for envintel...")
    packed_blob = tp.pack_field(da_dem.tp.to_dict(), binary=True)
    print(f"  ✓ Packed field: {len(packed_blob)/1024:.1f} KB")

    print("\n" + "=" * 75)
    print("  CONTOURING EXPORTS COMPLETED!")
    print(f"  Live URL: http://localhost:8765/upper_columbia_contoured_dem.html")
    print("=" * 75)

if __name__ == "__main__":
    main()
