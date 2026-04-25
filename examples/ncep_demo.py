"""
NCEP reanalysis + CMAP precipitation → pyterraplot globe export
================================================================

Downloads assumed already done:
  /home/guido/data/ncep/air.2m.mon.mean.nc
  /home/guido/data/ncep/precip.cmap.mon.mean.nc

Produces:
  /tmp/t2m_latest.html        — most recent 2m temperature
  /tmp/t2m_anomaly.html       — anomaly vs 1991-2020 climatology
  /tmp/precip_latest.html     — most recent monthly precipitation
  /tmp/precip_anomaly.html    — precip anomaly vs climatology
  /tmp/t2m_anom_compact.json  — compact frames for last 12 months (JS animate())
"""
import numpy as np
import xarray as xr
import pyterraplot  # noqa: F401 — registers .tp

DATA = "/home/guido/data/ncep"

# ── Load ──────────────────────────────────────────────────────────────────────

print("Loading datasets…")
t_ds = xr.open_dataset(f"{DATA}/air.2m.mon.mean.nc")
p_ds = xr.open_dataset(f"{DATA}/precip.cmap.mon.mean.nc")

t2m    = t_ds["air"]    # (time, lat, lon) in K, lon 0→360
precip = p_ds["precip"] # (time, lat, lon) in mm/day, lon 0→360

# ── Latest month ──────────────────────────────────────────────────────────────

t_latest = t2m.isel(time=-1)
p_latest = precip.isel(time=-1)

t_date = str(t_latest.time.values)[:7]
p_date = str(p_latest.time.values)[:7]
print(f"  Temperature latest:   {t_date}")
print(f"  Precipitation latest: {p_date}")

# ── 1991-2020 climatological mean (WMO standard) ─────────────────────────────

clim_slice = slice("1991-01-01", "2020-12-31")

t_clim  = t2m.sel(time=clim_slice).groupby("time.month").mean("time")
p_clim  = precip.sel(time=clim_slice).groupby("time.month").mean("time")

# Anomaly for latest month
t_month = int(t_latest.time.dt.month)
p_month = int(p_latest.time.dt.month)

t_anom  = t_latest - t_clim.sel(month=t_month)
p_anom  = p_latest - p_clim.sel(month=p_month)

# Drop the 'month' coord left over from groupby selection
t_anom  = t_anom.drop_vars("month", errors="ignore")
p_anom  = p_anom.drop_vars("month", errors="ignore")

# Give them nice metadata
t_latest.attrs.update({"long_name": f"2m Temperature {t_date}", "units": "K"})
t_anom.attrs.update({"long_name": f"2m Temp Anomaly vs 1991-2020 ({t_date})", "units": "K"})
p_latest.attrs.update({"long_name": f"Precipitation {p_date}", "units": "mm/day"})
p_anom.attrs.update({"long_name": f"Precip Anomaly vs 1991-2020 ({p_date})", "units": "mm/day"})
t_anom.name  = "t2m_anom"
p_anom.name  = "precip_anom"
t_latest.name = "t2m"
p_latest.name = "precip"

print()
print(f"  T anomaly  — min: {float(t_anom.min()):.2f} K   max: {float(t_anom.max()):.2f} K")
print(f"  P anomaly  — min: {float(p_anom.min()):.2f}     max: {float(p_anom.max()):.2f} mm/day")
print()

# ── Export HTML globes ────────────────────────────────────────────────────────

def sym_vlim(da, pct=98):
    """Symmetric colormap limit at the given percentile — keeps zero centred."""
    vals = da.values.ravel()
    vals = vals[~np.isnan(vals)]
    bound = float(np.percentile(np.abs(vals), pct))
    return -bound, bound

t_vlim  = sym_vlim(t_anom)
p_vlim  = sym_vlim(p_anom)
print(f"  T anomaly color range (p98 sym): {t_vlim[0]:.1f} … {t_vlim[1]:.1f} K")
print(f"  P anomaly color range (p98 sym): {p_vlim[0]:.1f} … {p_vlim[1]:.1f} mm/day")
print()

exports = [
    # (da, path, kind, cmap, title, vmin, vmax, levels)
    (t_latest, "/tmp/t2m_latest.html",         "pcolormesh", "viridis",  "2m Temperature",        None,       None,       12),
    (t_anom,   "/tmp/t2m_anomaly.html",         "pcolormesh", "RdYlBu_r","2m Temp Anomaly",        *t_vlim,    12),
    (t_anom,   "/tmp/t2m_anomaly_contourf.html","contourf",   "RdYlBu_r","2m Temp Anomaly contourf",*t_vlim,   14),
    (p_latest, "/tmp/precip_latest.html",       "pcolormesh", "YlGnBu",  "Precipitation",          0,          None,       12),
    (p_latest, "/tmp/precip_contourf.html",     "contourf",   "YlGnBu",  "Precipitation contourf", 0,          None,       12),
    (p_anom,   "/tmp/precip_anomaly.html",      "pcolormesh", "RdBu",    "Precip Anomaly",          *p_vlim,    12),
    (p_anom,   "/tmp/precip_anomaly_contourf.html","contourf","RdBu",    "Precip Anomaly contourf", *p_vlim,    14),
]

for da, path, kind, cmap, title, vmin, vmax, levels in exports:
    da.tp.to_html(path, kind=kind, title=title, cmap=cmap, alpha=0.82, vmin=vmin, vmax=vmax, levels=levels)
    kb = len(open(path).read()) / 1024
    print(f"  {path}  ({kb:.0f} kB)")

# ── Compact frames: last 12 months of temperature anomaly ─────────────────────

print()
print("Building last-12-months temperature anomaly animation…")

last12 = t2m.isel(time=slice(-12, None))
# compute anomaly for each month relative to 1991-2020 climatology
anom_frames = xr.concat(
    [(last12.isel(time=i) - t_clim.sel(month=int(last12.isel(time=i).time.dt.month)))
     .drop_vars("month", errors="ignore")
     for i in range(12)],
    dim="time",
).assign_coords(time=last12.time)
anom_frames.name = "t2m_anom"
anom_frames.attrs = {"units": "K", "long_name": "2m Temperature Anomaly"}

out_compact = "/tmp/t2m_anom_compact.json"
anom_frames.tp.frames_compact_to_json(out_compact, dim="time")
import os
kb = os.path.getsize(out_compact) / 1024
print(f"  {out_compact}  ({kb:.0f} kB)  — {12} frames")
print()
print("Done.")
print()
print("Open in browser:")
for _, path, *_ in exports:
    print(f"  xdg-open {path}")
