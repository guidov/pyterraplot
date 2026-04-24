"""
xarray accessor: da.tp.*

Registered automatically when pyterraplot is imported.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import xarray as xr

from .serialize import serialize


@xr.register_dataarray_accessor("tp")
class TerraplotAccessor:
    """
    Cartopy-style plotting on a terraplot 3D globe, from xarray.

    Usage
    -----
    import pyterraplot                # registers .tp accessor
    import xarray as xr

    ds  = xr.open_dataset("ecmwf_s2s.nc")
    t2m = ds["2m_temperature"].isel(time=0)

    # Export for the browser
    t2m.tp.to_json("field.json")

    # Live-serve from Python (requires pyterraplot[serve])
    t2m.tp.serve(port=8765)

    # Get raw dict (pass directly to your own FastAPI route etc.)
    payload = t2m.tp.to_dict()
    """

    def __init__(self, da: xr.DataArray) -> None:
        self._da = da

    # ── Serialisation ──────────────────────────────────────────────────────────

    def to_dict(
        self,
        lon_dim: str | None = None,
        lat_dim: str | None = None,
        wrap_lon: bool = True,
    ) -> dict[str, Any]:
        """Return the terraplot JSON payload as a Python dict."""
        return serialize(self._da, lon_dim=lon_dim, lat_dim=lat_dim, wrap_lon=wrap_lon)

    def to_json(
        self,
        path: str | Path,
        lon_dim: str | None = None,
        lat_dim: str | None = None,
        wrap_lon: bool = True,
    ) -> Path:
        """Write JSON file consumable by terraplot FieldLayer."""
        p = Path(path)
        serialize(self._da, lon_dim=lon_dim, lat_dim=lat_dim, wrap_lon=wrap_lon, path=p)
        return p

    # ── Live server ───────────────────────────────────────────────────────────

    def serve(
        self,
        port: int = 8765,
        host: str = "127.0.0.1",
        lon_dim: str | None = None,
        lat_dim: str | None = None,
        wrap_lon: bool = True,
        open_browser: bool = False,
    ) -> None:
        """
        Start a local HTTP server that serves this field at GET /field.
        The browser fetches it and passes directly to terraplot FieldLayer.

        Requires: pip install pyterraplot[serve]
        """
        from .server import serve as _serve

        payload = self.to_dict(lon_dim=lon_dim, lat_dim=lat_dim, wrap_lon=wrap_lon)
        _serve(payload, host=host, port=port, open_browser=open_browser)

    # ── Multi-step / ensemble helpers ─────────────────────────────────────────

    def frames(
        self,
        dim: str,
        lon_dim: str | None = None,
        lat_dim: str | None = None,
        wrap_lon: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Serialise each slice along `dim` as a separate frame dict.
        Useful for forecast lead-time animation.

        Example
        -------
        # Returns list of dicts, one per lead time
        steps = ds["t2m"].tp.frames(dim="time")
        json.dump(steps, open("frames.json", "w"))
        """
        slices = [
            serialize(
                self._da.isel({dim: i}),
                lon_dim=lon_dim,
                lat_dim=lat_dim,
                wrap_lon=wrap_lon,
            )
            for i in range(self._da.sizes[dim])
        ]
        # Attach the coordinate value as metadata
        coord_vals = self._da[dim].values
        for i, s in enumerate(slices):
            s["frame"] = i
            s["coord_value"] = str(coord_vals[i])
        return slices

    def frames_to_json(
        self,
        path: str | Path,
        dim: str,
        **kwargs,
    ) -> Path:
        """Write all frames to a single JSON file."""
        p = Path(path)
        p.write_text(json.dumps(self.frames(dim=dim, **kwargs)))
        return p
