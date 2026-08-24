# pyterraplot

**matplotlib + cartopy for xarray, rendered by [terraplot](https://terraplot.readthedocs.io) in the browser.**

Plot xarray DataArrays onto 52 map projections or an interactive 3D globe, using
the cartopy API you already know. The output is a self-contained HTML file — no
JS build step, no server, no plotting backend to install.

```python
import pyterraplot as tp
import pyterraplot.crs as ccrs
import pyterraplot.feature as cfeature

ax = tp.Axes(projection=ccrs.Robinson(central_longitude=-100))
ax.contourf(t2m, levels=16, cmap="RdBu_r", vmin=-30, vmax=30)
ax.add_feature(cfeature.LAND)
ax.coastlines(resolution="50m")
ax.gridlines(draw_labels=True)
ax.set_title("2 m air temperature")
ax.to_html("plot.html")
```

## Installation

```bash
pip install pyterraplot
pip install pyterraplot[serve]   # .tp.serve() live server
pip install pyterraplot[raster]  # to_cog() GeoTIFF export
pip install pyterraplot[all]     # + cf_xarray for CF coord detection
```

Rendering requires the [terraplot](https://github.com/guidov/terraplot) JS
bundle. `to_html()` finds it via the `TERRAPLOT_BUNDLE` environment variable, or
a sibling `terraplot/dist/terraplot.js` checkout, or an explicit
`terraplot_bundle=` argument.

```{toctree}
:maxdepth: 2
:caption: Contents

guide
api
```

## Indices

- {ref}`genindex`
- {ref}`modindex`
