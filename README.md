# pyterraplot

**xarray accessor + server bridge for the [terraplot](https://github.com/guidov/terraplot) 3D globe visualization library.**

Serialize xarray DataArrays — including real S2S forecast data from ECMWF, CanSIPS, or CFS — directly to the JSON contract that terraplot's `FieldLayer` consumes in the browser. Handles CF-convention coordinate name resolution, 0→360 longitude wrapping, NaN masking, and multi-step frame export for animation.

```bash
pip install pyterraplot
pip install pyterraplot[serve]   # for .tp.serve() live server
pip install pyterraplot[raster]  # for to_cog() GeoTIFF export
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

## Binary transport (recommended for anything bigger than a toy field)

JSON is fine for small grids, but a 0.25° global field serializes to ~5 MB of JSON per frame.
`pack_field` / `pack_frames` compress the same payload to gzip binary instead:

```python
from pyterraplot import pack_field, pack_frames

# Single field (gzip bytes by default — ~25% smaller than base64 on the wire)
blob = pack_field(t2m.tp.to_dict(), binary=True)

# Animation frames — same shape .tp.frames_compact() produces
blob = pack_frames(ds["t2m"].tp.frames_compact(dim="step"), binary=True)

# binary=False (default) returns base64 ASCII, for inline embedding in HTML/JS
b64 = pack_field(t2m.tp.to_dict())
```

Serve it from FastAPI:

```python
from fastapi import FastAPI
from fastapi.responses import Response
from pyterraplot import pack_frames

app = FastAPI()

@app.get("/api/field/t2m/frames-binary")
def frames_binary():
    blob = pack_frames(ds["t2m"].tp.frames_compact(dim="step"), binary=True)
    return Response(content=blob, media_type="application/gzip")
```

Consume it in the browser — `unpackField` / `unpackFrames` accept either a base64
string or the raw `ArrayBuffer` (preferred; skips the `atob` pass):

```javascript
import { unpackFrames, GeoMap } from 'terraplot';

const buf = await fetch('/api/field/t2m/frames-binary').then(r => r.arrayBuffer());
const data = await unpackFrames(buf);   // { lons, lats, frames: [{ field, coord_value }] }
map.animate(data, { type: 'pcolormesh', interval: 800 });
```

Typical payload for a 0.25° global grid (1440×721): ~5.5 MB/frame as JSON →
~1.5 MB/frame gzipped base64 → ~1.1 MB/frame raw gzip binary, and
`pack_frames` stores the grid coordinates once for all frames.

---

## GeoTIFF / COG export

Write the same 2D field as a Cloud-Optimized GeoTIFF for QGIS/ArcGIS, or serve it
back to terraplot's `unpackGeoTiff` (full round-trip). Requires
`pip install 'pyterraplot[raster]'`:

```python
from pyterraplot import to_cog

to_cog(t2m, "t2m_latest.tif")   # write to disk
blob = to_cog(t2m)              # or get the COG bytes to serve
```

```javascript
import { unpackGeoTiff, FieldLayer } from 'terraplot';

const f = await unpackGeoTiff('/api/field/t2m/cog');
const layer = new FieldLayer(f.lons, f.lats, f.field, { cmap: 'thermal' });
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

### `.tp.to_cog(path?)` → `Path | bytes`

Export the 2D field as a Cloud-Optimized GeoTIFF (to disk, or as bytes to serve).
Requires `pyterraplot[raster]`. See [GeoTIFF / COG export](#geotiff--cog-export).

### `.tp.to_html(path, ...)` → `Path`

Export a **self-contained HTML file** that renders the field on a 3D globe or a
2D projected map (with coastlines, colorbar, and optional `extent`) — no JS build
step, no server. Great for quick looks and sharing:

```python
t2m.tp.to_html("look.html", kind="pcolormesh", cmap="RdBu_r",
               center=(-95, 60), extent=(-141, -52, 41, 84))

# Static globe — disable the default auto-rotation (drag/zoom still work)
t2m.tp.to_html("still.html", spin=False)

# Neon-outline style: plain dark sphere + fluorescent green continents,
# no photographic texture underneath
t2m.tp.to_html("neon.html", earth_surface="none",
               coastlines=True, coastline_color="#39FF14")

# Thicker or thinner continental strokes (pixels)
t2m.tp.to_html("bold.html", earth_surface="none", coastlines=True,
               coastline_color="#00F0FF", coastline_width=4)
```

`earth_surface`: `'satellite'` (night-lights, default), `'shaded_relief'` /
`'stock'` (blue marble), `'none'` / `'outline'` (plain dark sphere), or an image
URL. `coastlines` defaults to True for 2D maps and False for the 3D globe;
`coastline_color`/`coastline_width` style the strokes (widths > 1 px render as
true thick lines — WebGL ignores width on hairlines).

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

### `.tp.frames_compact(dim, ...)` → `dict`

Multi-frame payload with `lons`/`lats` stored once and only the fields repeated —
this is the shape `pack_frames()` compresses and terraplot's `animate()` compact
branch consumes. Cuts payload size by ~60% vs `frames()`.

```python
compact = ds["t2m"].tp.frames_compact(dim="step")
# { lons, lats, frames: [{ field, coord_value, frame }, ...] }
```

Also `.tp.frames_to_json(path, dim)` for static hosting, and
`.tp.frames_to_html(path, dim, ...)` for a self-contained animated HTML export.

### Module-level: `pack_field`, `pack_frames`, `to_cog`

```python
from pyterraplot import pack_field, pack_frames, to_cog

pack_field(payload, binary=False)    # dict → gzip bytes (or base64 ASCII)
pack_frames(compact, binary=False)   # compact frames dict → gzip bytes / base64
to_cog(da, path=None)                # DataArray → COG (path or bytes)
```

`binary=True` serves raw gzip over the wire; the default base64 form is for
inline embedding in HTML/JS. See [Binary transport](#binary-transport-recommended-for-anything-bigger-than-a-toy-field).

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

# Export all lead times for animation — smallest static artifact
from pyterraplot import pack_frames
blob = pack_frames(ds["t2m"].tp.frames_compact(dim="step"))
open("frames.tplf", "w").write(blob)

# Or one self-contained HTML file with a globe + player
ds["t2m"].tp.frames_to_html("animation.html", dim="step")
```

---

## Roadmap

| Version | Features |
|---------|----------|
| v0.1 | `to_json`, `to_dict`, `serve`, `frames`, CF coord detection |
| v0.2 | Binary transport (`pack_field` / `pack_frames`), self-contained HTML export, `frames_compact` |
| v0.3 | `to_cog` GeoTIFF export, raw-gzip `binary=True` serving |
| next  | Streaming WebSocket updates (push new field without page reload), Gaussian reduced grid → regular regrid, ensemble stats |

---

## License

MIT © Guido Vettoretti
