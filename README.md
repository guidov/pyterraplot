# pyterraplot

**matplotlib + cartopy for xarray, rendered by [terraplot](https://github.com/guidov/terraplot) in the browser.**

Plot xarray DataArrays — including real S2S forecast data from ECMWF, CanSIPS,
or CFS — onto 52 map projections or an interactive 3D globe, with the cartopy
API you already know:

```python
ax = tp.Axes(projection=ccrs.NorthPolarStereo())
ax.contourf(t2m, levels=16, cmap="RdBu_r")
ax.add_feature(cfeature.LAND)
ax.gridlines(draw_labels=True)
ax.to_html("arctic.html")
```

The output is a **self-contained HTML file** — no JS build step, no server, no
plotting backend to install. Also serialises straight to the JSON contract
terraplot's `FieldLayer` consumes, handling CF-convention coordinate
resolution, 0→360 longitude wrapping, NaN masking, and multi-step frame export
for animation.

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

## Axes — matplotlib + cartopy in the browser

`to_html()` exports one field with one style. For everything else — layered
plots, real map projections, features, gridlines, vectors, multi-panel figures
— use the `Axes` / `subplots` API. Projections are instantiated cartopy-style:

```python
import pyterraplot as tp
import pyterraplot.crs as ccrs
import pyterraplot.feature as cfeature

ax = tp.Axes(projection=ccrs.Robinson(central_longitude=-100))
ax.contourf(air, levels=16, cmap="RdBu_r", vmin=-30, vmax=30)
ax.contour(air, levels=16, color="black", linewidth=1)   # outline the patches
ax.add_feature(cfeature.LAND)
ax.coastlines(resolution="50m")
ax.gridlines(draw_labels=True, xstep=60, ystep=30, linestyle="--")
ax.set_title("2 m air temperature")
ax.to_html("plot.html")
```

Layers render stacked in call order, and passing the same DataArray to several
primitives embeds the payload only once. In Jupyter, an Axes displays inline
(as its `_repr_html_`), so the last line is optional in a notebook. Every
primitive returns `self`, so calls chain.

`contourf` and `contour` with the same `levels` align **pixel-exactly**: like
matplotlib (one contour generator producing both lines and filled regions —
see contourpy's `mpl2014` algorithm), both derive their geometry from the same
marching-squares rings at the same band-edge thresholds, so isolines land on
the fill band boundaries by construction. This holds for the 3D globe and
every 2D projection, including NaN knockout regions (fills mask NaN cells,
isolines wrap around them).

### Projections — `pyterraplot.crs`

`ccrs.PlateCarree()`, `ccrs.Mercator()`, `ccrs.Robinson()` and friends carry
cartopy's parameter names and translate to the d3-geo projection terraplot
drives in the browser. 52 projections are available:

| Family | Projections |
|--------|-------------|
| Cylindrical | `PlateCarree`, `Mercator`, `TransverseMercator`, `Miller`, `LambertCylindrical` |
| Pseudocylindrical | `Robinson`, `Mollweide`, `Sinusoidal`, `EqualEarth`, `NaturalEarth`, `NaturalEarth2`, `EckertI`–`EckertVI` |
| Azimuthal | `Orthographic`, `Stereographic`, `NorthPolarStereo`, `SouthPolarStereo`, `LambertAzimuthalEqualArea`, `AzimuthalEquidistant`, `Gnomonic`, `NearsidePerspective` |
| Conic | `AlbersEqualArea`, `LambertConformal`, `EquidistantConic` — all take `standard_parallels=` |
| Other | `Aitoff`, `Hammer`, `WinkelTripel`, `Bonne`, `Polyconic`, `VanDerGrinten`, `Lagrange`, `Times` |
| Interrupted | `InterruptedGoodeHomolosine`, `InterruptedMollweide`, `InterruptedSinusoidal`, `InterruptedBoggs` |
| Special | `RotatedPole` (CORDEX-style native grids), `Globe3D` (the 3D globe), `Geodetic` (a transform, not a projection) |

```python
ccrs.LambertConformal(central_longitude=-96, central_latitude=39,
                      standard_parallels=(33, 45))
ccrs.NorthPolarStereo(central_longitude=-100)      # brings its own extent
ccrs.NearsidePerspective(satellite_height=35_785_831)
ccrs.RotatedPole(pole_longitude=-162, pole_latitude=39.25)
```

Projection classes are re-exported at the top level too, so `tp.Robinson()`
works if you prefer not to import the `crs` module. Ellipsoids and datums are
not modelled — terraplot renders on a sphere, so cartopy's
`globe=ccrs.Globe(ellipse=...)` has no counterpart.

Without a `projection=`, an Axes renders the interactive 3D globe, as before;
`projection=ccrs.Globe3D()` is the explicit spelling of that default. The
legacy `globe=False, projection="naturalEarth"` form still works.

### Primitives

| Method | Notes |
|--------|-------|
| `pcolormesh(da, cmap=, alpha=, vmin=, vmax=)` | smooth gradient fill (`imshow` is an alias) |
| `contourf(da, levels=, …)` | banded fill |
| `contour(da, levels=, color= / cmap=, linewidth=, zorder=)` | unfilled isolines; `linewidth > 1` renders as fat lines |
| `quiver(u, v, scale=, density=, cmap=)` | vector arrows |
| `barbs(u, v, length=, density=, flip=)` | wind barbs — pennant 50, full barb 10, half barb 5 |
| `streamplot(u, v, density=)` | streamlines, integrated in Python with RK4 |
| `plot(lons, lats, fmt, transform=)` | line; `fmt` takes matplotlib shorthand (`'r--o'`) |
| `scatter(lons, lats, c=, s=, cmap=)` | points, optionally coloured by value |
| `text(lon, lat, s)` / `annotate(text, xy)` | geographic labels |
| `marker(lat, lon, label=)` | labelled point marker |
| `tissot(rad_km=, lons=, lats=)` | Tissot's indicatrices — projection distortion made visible |
| `add_geometries(geoms, crs=)` | GeoJSON, or anything with `__geo_interface__` (shapely, geopandas) |
| `add_feature(cfeature.LAND, **style)` | cartopy-style map features |
| `coastlines(resolution=)` / `borders` / `states` / `rivers` / `lakes` / `land` / `ocean` / `cities` | shorthands for the same |
| `gridlines(draw_labels=, xlocs=, ylocs=, xstep=, ystep=, linestyle=, dms=)` | graticule with edge labels |
| `set_extent([lon0, lon1, lat0, lat1])` / `set_global()` / `set_center(lon, lat)` | the view |
| `stock_img()` / `background_img(url)` | blue-marble or custom basemap image |
| `set_title(text)` / `legend(loc=)` | labelling; `legend` picks up any layer given `label=` |
| `colorbar(cmap=, vmin=, vmax=, label=, orientation=, scale=, ticks=)` | override the auto-derived colorbar |
| `animate(da, dim="step", kind="contourf", interval=800)` | frame animation; static layers stay on top |
| `to_html(path)` / `savefig(path)` | write the self-contained page |

Features come from Natural Earth at three resolutions — `'110m'` (default),
`'50m'`, `'10m'` — via `coastlines(resolution="50m")` or
`cfeature.BORDERS.with_scale("50m")`. The `cfeature` constants are immutable;
`with_scale` and `with_style` return new objects.

`plot(..., transform=ccrs.Geodetic())` follows great circles between vertices;
`transform=ccrs.PlateCarree()` (the default) keeps segments straight in lon/lat.

Primitives that only exist for flat maps (`gridlines`, `plot`, `scatter`,
`text`, `barbs`, `streamplot`, `tissot`, `add_geometries`, `quiver`, `marker`)
raise `NotImplementedError` on a 3D-globe Axes rather than silently doing
nothing.

### Multi-panel figures — `subplots`

```python
fig, axes = tp.subplots(2, 2, figsize=(16, 9),
                        subplot_kw={"projection": ccrs.EqualEarth()})
for ax, (season, field) in zip(axes.flat, seasons.items()):
    ax.contourf(field, levels=16, cmap="RdBu_r", vmin=-35, vmax=35)
    ax.coastlines()
    ax.set_title(season)
fig.suptitle("Seasonal 2 m temperature")
fig.colorbar(label="2 m temperature [°C]", ticks=7)   # one shared scale
fig.savefig("seasons.html")
```

`subplots` follows matplotlib: `squeeze=True` by default, so a 1×1 grid returns
a bare Axes and a single row or column returns a 1D array. For per-panel
projections use `fig.add_subplot(nrows, ncols, index, projection=...)` — it
also accepts the three-digit `add_subplot(221)` shorthand, plus `rowspan=` and
`colspan=`. Panels may mix flat projections and 3D globes in one figure, and
they share one payload registry, so a DataArray drawn in several panels is
embedded once.

Each panel gets its own colorbar with `ax.colorbar(...)`; `fig.colorbar(...)`
puts a single one under the whole grid.

See `examples/cartopy_demo.py` for all of this end to end.

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
| v0.4 | `pyterraplot.crs` projection objects (52 projections), `pyterraplot.feature`, `gridlines`/`plot`/`scatter`/`text`/`barbs`/`streamplot`/`tissot`/`add_geometries`, `subplots()` multi-panel figures |
| next  | Streaming WebSocket updates (push new field without page reload), Gaussian reduced grid → regular regrid, ensemble stats |

---

## License

MIT © Guido Vettoretti

Exported HTML files embed the terraplot bundle, which includes d3-geo,
d3-geo-projection, d3-contour and d3-scale-chromatic (all ISC). Both licences
are permissive — the bundle may be redistributed inside a closed-source product
— and the required copyright notices are emitted as a banner at the top of the
inlined script in every generated page.
