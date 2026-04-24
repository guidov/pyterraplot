# pyterraplot

**xarray accessor + server bridge for the [terraplot](https://github.com/guidov/terraplot) 3D globe visualization library.**

Serialize xarray DataArrays — including real S2S forecast data from ECMWF, CanSIPS, or CFS — directly to the JSON contract that terraplot's `FieldLayer` consumes in the browser. Handles CF-convention coordinate name resolution, 0→360 longitude wrapping, NaN masking, and multi-step frame export for animation.

```bash
pip install pyterraplot
pip install pyterraplot[serve]   # for .tp.serve() live server
pip install pyterraplot[all]     # + cf_xarray for automatic CF coord detection
```

---

## Quick start

```python
import pyterraplot          # registers .tp accessor on xr.DataArray
import xarray as xr

ds  = xr.open_dataset("ecmwf_s2s.nc")
t2m = ds["2m_temperature"].isel(time=0)

# Write JSON — fetch this in your JS app and pass to FieldLayer
t2m.tp.to_json("field.json")

# Or live-serve at http://localhost:8765/field
t2m.tp.serve(port=8765)
```

In your JS app:

```javascript
import { FieldLayer, Colormaps } from 'terraplot';

const { lons, lats, field } = await fetch('http://localhost:8765/field').then(r => r.json());
const layer = new FieldLayer(lons, lats, field, { cmap: Colormaps.RdYlBu_r, alpha: 0.65 });
globe.scene().add(layer.mesh);
```

---

## API

### `.tp.to_dict(lon_dim?, lat_dim?, wrap_lon?)` → `dict`

Returns the payload as a Python dict. Use this to build your own FastAPI/Flask routes.

```python
payload = t2m.tp.to_dict()
# { 'lons': [...], 'lats': [...], 'field': [[...], ...], 'name': ..., 'units': ..., 'long_name': ... }
```

### `.tp.to_json(path, lon_dim?, lat_dim?, wrap_lon?)` → `Path`

Write JSON to disk. The browser fetches it with a plain `fetch()` call.

### `.tp.serve(port?, host?, open_browser?)`

Start a local HTTP server. Requires `pyterraplot[serve]`.

- `GET /field` — returns the current payload
- `GET /health` — `{"status": "ok"}`
- `WS  /ws` — WebSocket; pushed on connect (future: push on update)

Blocks the calling thread. For Jupyter, run in a daemon thread:

```python
import threading
t = threading.Thread(target=t2m.tp.serve, kwargs={"port": 8765}, daemon=True)
t.start()
```

### `.tp.frames(dim, lon_dim?, lat_dim?, wrap_lon?)` → `list[dict]`

Serialise each slice along `dim` as a frame. Essential for S2S lead-time animation.

```python
# All 46 lead-time steps of an ECMWF ENS run
frames = ds["t2m"].tp.frames(dim="time")
# Each frame: { lons, lats, field, frame: 0..45, coord_value: "2026-04-25" }
```

### `.tp.frames_to_json(path, dim)` → `Path`

Write all frames to a single JSON file for static hosting.

---

## Input format

| Requirement | Detail |
|-------------|--------|
| DataArray must be **2D** | Reduce extra dims first: `.isel(time=0, number=0)` |
| Lat/lon dim names | Auto-detected from: `lat`, `latitude`, `y`, `rlat`, `nav_lat` / `lon`, `longitude`, `x`, `rlon`, `nav_lon`. Pass `lon_dim=`/`lat_dim=` to override |
| Longitude convention | `0→360` is automatically re-wrapped to `−180→180` (disable with `wrap_lon=False`) |
| NaN handling | NaN cells become `null` in JSON; terraplot renders them transparent |

## CF-convention support

Install `cf_xarray` for automatic axis detection from CF metadata:

```bash
pip install pyterraplot[cf]
```

```python
# Works even with non-standard dim names if CF attributes are present
ds["t2m"].tp.to_json("field.json")
```

---

## Real S2S data example

```python
import pyterraplot
import xarray as xr
import cfgrib   # pip install cfgrib

# ECMWF S2S GRIB2 file
ds = xr.open_dataset("ecmwf_s2s_2m_temperature.grib2", engine="cfgrib")

# Export all lead times for animation
ds["t2m"].tp.frames_to_json("frames.json", dim="time")
```

---

## Roadmap

| Version | Features |
|---------|----------|
| v0.1 | `to_json`, `to_dict`, `serve`, `frames`, CF coord detection |
| v0.2 | Gaussian reduced grid → regular lat/lon regrid, ensemble stats |
| v0.3 | Streaming WebSocket updates (push new field without page reload) |

---

## License

MIT © Guido Vettoretti
