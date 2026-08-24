# User guide

## Axes

An {class}`~pyterraplot.Axes` collects layer primitives and renders them stacked
in call order, the same mental model as matplotlib. Every primitive returns
`self`, so calls chain. Passing the same DataArray to several primitives embeds
the payload only once.

Without a `projection=`, an Axes renders the interactive 3D globe;
`projection=ccrs.Globe3D()` is the explicit spelling of that default.

`contourf` and `contour` with the same `levels` align pixel-exactly: both derive
their geometry from the same marching-squares rings at the same band-edge
thresholds, so isolines land on the fill band boundaries by construction.

## Projections

Projections follow cartopy's instantiation style and carry cartopy's parameter
names.

```python
import pyterraplot.crs as ccrs

ccrs.PlateCarree(central_longitude=180)
ccrs.LambertConformal(central_longitude=-96, standard_parallels=(33, 45))
ccrs.NorthPolarStereo(central_longitude=-100)      # brings its own extent
ccrs.NearsidePerspective(satellite_height=35_785_831)
ccrs.RotatedPole(pole_longitude=-162, pole_latitude=39.25)
```

| Family | Projections |
|--------|-------------|
| Cylindrical | `PlateCarree`, `Mercator`, `TransverseMercator`, `Miller`, `LambertCylindrical` |
| Pseudocylindrical | `Robinson`, `Mollweide`, `Sinusoidal`, `EqualEarth`, `NaturalEarth`, `NaturalEarth2`, `EckertI`–`EckertVI` |
| Azimuthal | `Orthographic`, `Stereographic`, `NorthPolarStereo`, `SouthPolarStereo`, `LambertAzimuthalEqualArea`, `AzimuthalEquidistant`, `Gnomonic`, `NearsidePerspective` |
| Conic | `AlbersEqualArea`, `LambertConformal`, `EquidistantConic` |
| Other | `Aitoff`, `Hammer`, `WinkelTripel`, `Bonne`, `Polyconic`, `VanDerGrinten`, `Lagrange`, `Times` |
| Interrupted | `InterruptedGoodeHomolosine`, `InterruptedMollweide`, `InterruptedSinusoidal`, `InterruptedBoggs` |
| Special | `RotatedPole`, `Globe3D`, `Geodetic` |

Projection classes are re-exported at the top level, so `tp.Robinson()` works
too. Ellipsoids and datums are not modelled — terraplot renders on a sphere, so
cartopy's `globe=ccrs.Globe(ellipse=...)` has no counterpart.

:::{note}
`RotatedPole` does not support `central_rotated_longitude`. It is a spin about
the rotated pole applied *after* the other two rotations, which d3-geo's
three-angle `rotate()` composition cannot express; passing a non-zero value
raises `NotImplementedError` rather than silently producing a wrong map.
:::

## Features

```python
import pyterraplot.feature as cfeature

ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN, facecolor="#0b1a30")
ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.8)
```

Geometry comes from Natural Earth at three resolutions — `110m` (default),
`50m`, `10m`. The `cfeature` constants are immutable; `with_scale` and
`with_style` return new objects. Filled areal features (`LAND`, `OCEAN`,
`LAKES`) render beneath the field layers, so data stays visible on top.

## Vectors

```python
ax.quiver(u, v, scale=26)                 # arrows
ax.barbs(u, v, length=9)                  # meteorological wind barbs
ax.streamplot(u, v, density=1.4)          # streamlines
```

Wind barbs follow the standard convention — the staff points upwind, with a
pennant per 50, a full barb per 10 and a half barb per 5, in the units of
`u`/`v`. Streamlines are integrated in Python with RK4 (see
{mod}`pyterraplot.geodesy`) and embedded as geometry.

## Geodesics

`plot(..., transform=ccrs.Geodetic())` follows great circles between vertices;
`transform=ccrs.PlateCarree()` (the default) keeps segments straight in lon/lat.

```python
ax.plot([-123.1, 139.7], [49.3, 35.7], transform=ccrs.Geodetic(),
        color="#38bdf8", linewidth=2, label="great circle")
ax.tissot(rad_km=500)      # distortion, made visible
```

## Multi-panel figures

```python
fig, axes = tp.subplots(2, 2, figsize=(16, 9),
                        subplot_kw={"projection": ccrs.EqualEarth()})
for ax, (season, field) in zip(axes.flat, seasons.items()):
    ax.contourf(field, levels=16, cmap="RdBu_r", vmin=-35, vmax=35)
    ax.coastlines()
    ax.set_title(season)
fig.suptitle("Seasonal 2 m temperature")
fig.colorbar(label="2 m temperature [°C]", ticks=7)
fig.savefig("seasons.html")
```

`subplots` follows matplotlib's squeeze rules. For per-panel projections use
`fig.add_subplot(nrows, ncols, index, projection=...)`, which also accepts the
three-digit `add_subplot(221)` shorthand plus `rowspan=`/`colspan=`. Panels may
mix flat projections and 3D globes, and share one payload registry.

:::{note}
Primitives that only exist for flat maps — `gridlines`, `plot`, `scatter`,
`text`, `barbs`, `streamplot`, `tissot`, `add_geometries`, `quiver`, `marker` —
raise `NotImplementedError` on a 3D-globe Axes rather than silently doing
nothing.
:::

## Serialisation

The `.tp` accessor also exports the raw payload for your own frontend.

```python
payload = t2m.tp.to_dict()
t2m.tp.to_json("field.json")
t2m.tp.to_cog("field.tif")                       # needs pyterraplot[raster]
blob = pack_field(payload, binary=True)          # gzip for the wire
compact = ds["t2m"].tp.frames_compact(dim="step")
```
