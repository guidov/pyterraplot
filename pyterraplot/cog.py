"""
Export a terraplot field to a (Cloud-Optimized) GeoTIFF.

rioxarray / rasterio pull in GDAL, which is heavy, so they are an optional
extra and imported lazily:

    pip install 'pyterraplot[raster]'

The dimension detection and optional 0→360 longitude wrap mirror
pyterraplot.serialize, so a COG written here round-trips back through the same
field contract (terraplot's ``unpackGeoTiff``).
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import xarray as xr

from .serialize import _LAT_NAMES, _LON_NAMES, _find_dim


def to_cog(
    da: xr.DataArray,
    path: str | Path | None = None,
    *,
    lon_dim: str | None = None,
    lat_dim: str | None = None,
    wrap_lon: bool = True,
    crs: str = "EPSG:4326",
) -> bytes | Path:
    """
    Write a 2D DataArray to a Cloud-Optimized GeoTIFF.

    Parameters
    ----------
    da       : 2D DataArray with lat/lon dims (length-1 extra dims are squeezed)
    path     : output path; if None, the COG bytes are returned instead
    lon_dim, lat_dim : override the auto-detected dimension names
    wrap_lon : convert 0→360 longitudes to -180→180 (default True)
    crs      : CRS to tag on the raster (default geographic WGS84)

    Returns
    -------
    The output ``Path`` when ``path`` is given, else the COG file as ``bytes``.
    """
    try:
        import rioxarray  # noqa: F401 — registers the .rio accessor
    except ModuleNotFoundError as e:  # pragma: no cover - import guard
        raise ModuleNotFoundError(
            "to_cog requires rioxarray. Install with: pip install 'pyterraplot[raster]'"
        ) from e

    da = da.squeeze(drop=True)
    if da.ndim != 2:
        raise ValueError(
            f"DataArray must be 2D (lat × lon) for GeoTIFF export. "
            f"Got shape {da.shape} with dims {da.dims}. "
            "Reduce extra dimensions first (e.g. .isel(time=0, ensemble=0))."
        )

    lon_dim = lon_dim or _find_dim(da, _LON_NAMES)
    lat_dim = lat_dim or _find_dim(da, _LAT_NAMES)
    da = da.transpose(lat_dim, lon_dim)

    # Wrap 0→360 longitudes to -180→180 and re-sort, matching serialize().
    lons = da[lon_dim].values.astype(float)
    if wrap_lon and lons.size and lons.max() > 180:
        wrapped = np.where(lons > 180, lons - 360, lons)
        order = np.argsort(wrapped)
        da = da.isel({lon_dim: order}).assign_coords({lon_dim: wrapped[order]})

    # North-up raster: latitude descending (row 0 = north).
    lat_vals = da[lat_dim].values
    if lat_vals.size > 1 and lat_vals[0] < lat_vals[-1]:
        da = da.isel({lat_dim: slice(None, None, -1)})

    da = da.astype("float32")
    da = da.rio.set_spatial_dims(x_dim=lon_dim, y_dim=lat_dim, inplace=False)
    da = da.rio.write_crs(crs, inplace=False)
    da = da.rio.write_nodata(np.nan, inplace=False)

    if path is not None:
        p = Path(path)
        da.rio.to_raster(p, driver="COG", dtype="float32")
        return p

    fd, tmp = tempfile.mkstemp(suffix=".tif")
    os.close(fd)
    try:
        da.rio.to_raster(tmp, driver="COG", dtype="float32")
        with open(tmp, "rb") as fh:
            return fh.read()
    finally:
        try:
            os.unlink(tmp)
        except OSError:  # pragma: no cover
            pass
