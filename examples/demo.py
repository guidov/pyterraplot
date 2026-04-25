"""
pyterraplot demo
================

Shows all accessor methods with a synthetic S2S-like dataset.
Run: python examples/demo.py
"""
import json
import numpy as np
import xarray as xr

import pyterraplot  # registers da.tp accessor

# ── 1. Synthetic 2D DataArray (single field, lat × lon) ───────────────────

nlat, nlon = 91, 180
lats = np.linspace(90, -90, nlat)
lons = np.linspace(-180, 180, nlon)
LON, LAT = np.meshgrid(lons, lats)

# Realistic-looking temperature anomaly pattern
t2m_data = (
    8  * np.cos(np.radians(LAT)) * np.sin(np.radians(2 * LON)) +
    5  * np.sin(np.radians(3 * LON)) * np.cos(np.radians(2 * LAT)) +
    3  * np.cos(np.radians(5 * LON)) * np.sin(np.radians(LAT))
).astype(np.float32)

# Punch some NaN holes (e.g. missing land mask)
rng = np.random.default_rng(42)
mask = rng.random((nlat, nlon)) < 0.03
t2m_data[mask] = np.nan

t2m = xr.DataArray(
    t2m_data,
    dims=["lat", "lon"],
    coords={"lat": lats, "lon": lons},
    name="t2m",
    attrs={"units": "K", "long_name": "2m temperature anomaly"},
)

# ── 2. to_dict() ─────────────────────────────────────────────────────────

payload = t2m.tp.to_dict()
print("── to_dict() ───────────────────────────────────────────────────────────")
print(f"  keys:        {list(payload.keys())}")
print(f"  lons:        {len(payload['lons'])} values  [{payload['lons'][0]:.1f} … {payload['lons'][-1]:.1f}]")
print(f"  lats:        {len(payload['lats'])} values  [{payload['lats'][0]:.1f} … {payload['lats'][-1]:.1f}]")
print(f"  field shape: {len(payload['field'])} × {len(payload['field'][0])}")
none_count = sum(1 for row in payload["field"] for v in row if v is None)
print(f"  NaN → None:  {none_count} cells")
print(f"  name:        {payload['name']!r}  units: {payload['units']!r}")
print()

# ── 3. to_json() ─────────────────────────────────────────────────────────

out_json = "/tmp/t2m.json"
t2m.tp.to_json(out_json)
size_kb = len(open(out_json).read()) / 1024
print("── to_json() ───────────────────────────────────────────────────────────")
print(f"  written to:  {out_json}  ({size_kb:.1f} kB)")
print()

# ── 4. Multi-frame S2S dataset (time × lat × lon) ────────────────────────

ntime = 8
times = np.arange(1, ntime + 1) * 7  # lead days: 7, 14, …, 56
phase = np.linspace(0, 2 * np.pi, ntime, endpoint=False)

frames_data = np.stack([
    (
        10 * np.cos(np.radians(LAT)) * np.sin(np.radians(2 * LON) + p) +
         4 * np.cos(np.radians(2 * LAT)) * np.cos(np.radians(3 * LON) + p * 0.7)
    ).astype(np.float32)
    for p in phase
], axis=0)

t2m_s2s = xr.DataArray(
    frames_data,
    dims=["time", "lat", "lon"],
    coords={"time": times, "lat": lats, "lon": lons},
    name="t2m",
    attrs={"units": "K", "long_name": "S2S 2m temperature anomaly"},
)

# ── 5. frames() — full format (lons/lats in every frame) ─────────────────

frames_full = t2m_s2s.tp.frames(dim="time")
full_size_kb = len(json.dumps(frames_full)) / 1024
print("── frames() ────────────────────────────────────────────────────────────")
print(f"  {len(frames_full)} frames, keys per frame: {list(frames_full[0].keys())}")
print(f"  JSON size:   {full_size_kb:.1f} kB")
print()

# ── 6. frames_compact() — compact format (lons/lats once) ────────────────

compact = t2m_s2s.tp.frames_compact(dim="time")
compact_size_kb = len(json.dumps(compact)) / 1024
saving_pct = (1 - compact_size_kb / full_size_kb) * 100
print("── frames_compact() ────────────────────────────────────────────────────")
print(f"  top-level keys:  {list(compact.keys())}")
print(f"  frame keys:      {list(compact['frames'][0].keys())}")
print(f"  JSON size:       {compact_size_kb:.1f} kB  ({saving_pct:.0f}% smaller than full frames)")
print(f"  coord_values:    {[f['coord_value'] for f in compact['frames']]}")
print()

# Write it out too
out_compact = "/tmp/t2m_compact.json"
t2m_s2s.tp.frames_compact_to_json(out_compact, dim="time")
print(f"  written to:  {out_compact}")
print()

# ── 7. to_html() — self-contained globe HTML ──────────────────────────────

out_html = "/tmp/t2m_globe.html"
t2m.tp.to_html(
    out_html,
    title="S2S Temperature Anomaly",
    cmap="RdYlBu_r",
    alpha=0.72,
)
html_size_kb = len(open(out_html).read()) / 1024
print("── to_html() ───────────────────────────────────────────────────────────")
print(f"  written to:  {out_html}  ({html_size_kb:.1f} kB)")
print(f"  open in browser:  xdg-open {out_html}")
print()

print("Done. JS contract summary:")
print("  globe.pcolormesh(payload.lons, payload.lats, payload.field, {{ cmap: 'RdYlBu_r' }})")
print("  globe.animate(compact, {{ type: 'pcolormesh', interval: 700 }})")
