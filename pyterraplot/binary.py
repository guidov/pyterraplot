"""
Binary packing for terraplot field data.

Encoding pipeline:
  numpy float32 → little-endian binary → gzip → base64 ASCII

Browser decoding:
  base64 → gzip decompress (DecompressionStream) → DataView → Float32Array

Single-field format (magic TPLD):
  uint32  magic = 0x44_4C_50_54  "TPLD"
  uint32  version = 1
  uint32  nlon
  uint32  nlat
  float32[nlon]         lons
  float32[nlat]         lats
  float32[nlat * nlon]  field (row-major: field[j*nlon+i] → (lats[j], lons[i]))
  uint32                meta_len
  utf-8[meta_len]       JSON {"name", "units", "long_name"}

Multi-frame format (magic TPLF):
  uint32  magic = 0x46_4C_50_54  "TPLF"
  uint32  version = 1
  uint32  nlon
  uint32  nlat
  uint32  n_frames
  float32[nlon]          lons
  float32[nlat]          lats
  uint32                 coord_buf_len
  bytes[coord_buf_len]   coord values: each is uint16 len + utf-8 bytes
  float32[n_frames * nlat * nlon]  all fields (frame-major, then row-major)
  uint32                 meta_len
  utf-8[meta_len]        JSON {"name", "units", "long_name"}

Size reduction for typical S2S global grid (0.25°, 1440×721):
  JSON:   ~5.5 MB / field
  Binary: ~4.2 MB raw → ~1.5 MB gzipped base64
"""
from __future__ import annotations

import gzip
import json as _json
import struct
from typing import Any

import numpy as np

MAGIC_FIELD  = 0x44_4C_50_54   # "TPLD" as little-endian uint32
MAGIC_FRAMES = 0x46_4C_50_54   # "TPLF" as little-endian uint32
VERSION      = 1


def pack_field(payload: dict[str, Any], *, binary: bool = False) -> str | bytes:
    """
    Compress a single terraplot payload to gzip binary.

    Parameters
    ----------
    payload : dict with keys lons, lats, field, name, units, long_name
              (the output of pyterraplot.serialize.serialize())
    binary : when True, return raw gzip bytes instead of base64 ASCII —
              ~25% smaller over the wire and skips the browser atob() pass.

    Returns
    -------
    ASCII base64 string for inline embedding in HTML/JS, or raw gzip
    bytes when binary=True.
    """
    lons = np.asarray(payload["lons"], dtype="<f4")
    lats = np.asarray(payload["lats"], dtype="<f4")
    nlon, nlat = len(lons), len(lats)

    field_arr = _to_float32(payload["field"], nlat, nlon)

    meta = _json.dumps({
        "name":      payload.get("name", "") or "",
        "units":     payload.get("units", "") or "",
        "long_name": payload.get("long_name", "") or "",
    }).encode()

    raw = (
        struct.pack("<IIII", MAGIC_FIELD, VERSION, nlon, nlat)
        + lons.tobytes()
        + lats.tobytes()
        + field_arr.ravel().tobytes()
        + struct.pack("<I", len(meta))
        + meta
    )
    return _compress(raw, binary=binary)


def pack_frames(compact: dict[str, Any], *, binary: bool = False) -> str | bytes:
    """
    Compress a frames_compact payload to gzip binary.

    Parameters
    ----------
    compact : dict produced by pyterraplot accessor .frames_compact()
              { lons, lats, name, units, long_name,
                frames: [{ field, coord_value, frame }, ...] }
    binary : when True, return raw gzip bytes instead of base64 ASCII.

    Returns
    -------
    ASCII base64 string for inline embedding in HTML/JS, or raw gzip
    bytes when binary=True.
    """
    lons = np.asarray(compact["lons"], dtype="<f4")
    lats = np.asarray(compact["lats"], dtype="<f4")
    nlon, nlat = len(lons), len(lats)
    frames     = compact["frames"]
    n_frames   = len(frames)

    # Coord values: uint16 length-prefix + utf-8 bytes
    coord_parts: list[bytes] = []
    for f in frames:
        cv = str(f.get("coord_value", "")).encode()
        coord_parts.append(struct.pack("<H", len(cv)) + cv)
    coord_buf = b"".join(coord_parts)

    # Pack all fields into a contiguous float32 array
    all_fields = np.empty((n_frames, nlat, nlon), dtype="<f4")
    for k, f in enumerate(frames):
        all_fields[k] = _to_float32(f["field"], nlat, nlon)

    meta = _json.dumps({
        "name":      compact.get("name", "") or "",
        "units":     compact.get("units", "") or "",
        "long_name": compact.get("long_name", "") or "",
    }).encode()

    raw = (
        struct.pack("<IIIII", MAGIC_FRAMES, VERSION, nlon, nlat, n_frames)
        + lons.tobytes()
        + lats.tobytes()
        + struct.pack("<I", len(coord_buf))
        + coord_buf
        + all_fields.ravel().tobytes()
        + struct.pack("<I", len(meta))
        + meta
    )
    return _compress(raw, binary=binary)


# ── helpers ────────────────────────────────────────────────────────────────────

def _to_float32(field: Any, nlat: int, nlon: int) -> np.ndarray:
    """Convert a field (nested list-of-lists with possible None, or ndarray) to float32."""
    if isinstance(field, np.ndarray):
        return np.asarray(field, dtype="<f4").reshape(nlat, nlon)

    arr = np.empty((nlat, nlon), dtype="<f4")
    for j, row in enumerate(field):
        for i, v in enumerate(row):
            arr[j, i] = float("nan") if v is None else float(v)
    return arr


def _compress(raw: bytes, *, binary: bool = False) -> str | bytes:
    gz = gzip.compress(raw, compresslevel=6)
    if binary:
        return gz
    import base64
    return base64.b64encode(gz).decode("ascii")


def _compress_b64(raw: bytes) -> str:
    return _compress(raw)  # type: ignore[return-value]
