"""
Serialize an xarray DataArray to the JSON contract that terraplot FieldLayer expects:
  { lons, lats, field, name, units, long_name }

field is a 2D list [j][i] where:
  j = latitude index  (lats[0] is the first lat value)
  i = longitude index (lons[0] is the first lon value)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

# CF-convention and common alternative names for lat/lon dimensions
_LON_NAMES = {"lon", "longitude", "x", "rlon", "nav_lon", "nlon"}
_LAT_NAMES = {"lat", "latitude", "y", "rlat", "nav_lat", "nlat"}


def _find_dim(da: xr.DataArray, candidates: set[str]) -> str:
    for dim in da.dims:
        if dim.lower() in candidates:
            return dim
    # fall back to cf_xarray if installed
    try:
        import cf_xarray  # noqa: F401
        axis_map = da.cf.axes
        if "X" in axis_map and candidates == _LON_NAMES:
            return axis_map["X"][0]
        if "Y" in axis_map and candidates == _LAT_NAMES:
            return axis_map["Y"][0]
    except (ImportError, Exception):
        pass
    raise ValueError(
        f"Cannot identify {'longitude' if candidates == _LON_NAMES else 'latitude'} "
        f"dimension. dims present: {list(da.dims)}. "
        "Pass lon_dim= / lat_dim= explicitly."
    )


def serialize(
    da: xr.DataArray,
    *,
    lon_dim: str | None = None,
    lat_dim: str | None = None,
    wrap_lon: bool = True,
    path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Convert a 2D DataArray to a terraplot-compatible dict.

    Parameters
    ----------
    da       : 2D xarray DataArray with lat and lon dimensions
    lon_dim  : name of the longitude dimension (auto-detected if None)
    lat_dim  : name of the latitude dimension (auto-detected if None)
    wrap_lon : convert 0→360 longitudes to -180→180 (default True)
    path     : if given, write JSON to this file path

    Returns
    -------
    dict with keys: lons, lats, field, name, units, long_name
    """
    if da.ndim != 2:
        raise ValueError(
            f"DataArray must be 2D (lat × lon). Got shape {da.shape} with dims {da.dims}. "
            "Reduce extra dimensions first (e.g. .isel(time=0, ensemble=0))."
        )

    lon_dim = lon_dim or _find_dim(da, _LON_NAMES)
    lat_dim = lat_dim or _find_dim(da, _LAT_NAMES)

    # Ensure lat is the first axis, lon second
    da = da.transpose(lat_dim, lon_dim)

    lons = da[lon_dim].values.copy().astype(float)
    lats = da[lat_dim].values.copy().astype(float)

    # Wrap 0→360 to -180→180
    if wrap_lon and lons.max() > 180:
        lons = np.where(lons > 180, lons - 360, lons)
        sort_idx = np.argsort(lons)
        lons = lons[sort_idx]
        da = da.isel({lon_dim: sort_idx})

    field = da.values.astype(float)

    # Replace NaN with None so JSON serialises cleanly
    field_list: list = np.where(np.isnan(field), None, field).tolist()

    payload: dict[str, Any] = {
        "lons":      lons.tolist(),
        "lats":      lats.tolist(),
        "field":     field_list,
        "name":      da.name or "",
        "units":     da.attrs.get("units", ""),
        "long_name": da.attrs.get("long_name", da.name or ""),
    }

    if path is not None:
        Path(path).write_text(json.dumps(payload))

    return payload
