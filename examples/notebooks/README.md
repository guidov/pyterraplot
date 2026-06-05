# pyterraplot example notebooks

A hands-on tour with **full coverage of the public API**. Each notebook is standalone —
run them in any order. They build their own synthetic climate-style fields, so you need
no external data.

| # | Notebook | Covers |
|---|----------|--------|
| 00 | [`00_overview.ipynb`](00_overview.ipynb) | `import`, `.tp` accessor, inline `_repr_html_` globe |
| 01 | [`01_serialize_dict_json.ipynb`](01_serialize_dict_json.ipynb) | `serialize`, `.tp.to_dict` / `.to_json`, dim auto-detection, 0→360 lon wrap, NaN→null, CF detection |
| 02 | [`02_html_globe_maps.ipynb`](02_html_globe_maps.ipynb) | `.tp.to_html` — 3D globe, 2D projections, `contourf` vs `pcolormesh`, cmap/vmin/vmax/levels, `extent`, binary vs JSON |
| 03 | [`03_frames_animation.ipynb`](03_frames_animation.ipynb) | `.tp.frames`, `frames_compact`, `frames_to_json`, `frames_to_html` (play/scrub animation) |
| 04 | [`04_binary_packing.ipynb`](04_binary_packing.ipynb) | `pack_field`, `pack_frames`, the TPLD/TPLF binary format |
| 05 | [`05_dataset_quiver_compare.ipynb`](05_dataset_quiver_compare.ipynb) | `ds.tp.quiver_html` (vectors), `ds.tp.compare_html` (side-by-side) |
| 06 | [`06_geotiff_cog.ipynb`](06_geotiff_cog.ipynb) | `.tp.to_cog` to file and to bytes, round-trip read |
| 07 | [`07_live_server.ipynb`](07_live_server.ipynb) | `.tp.serve` in a background thread, `/field` + `/health` |
| 08 | [`08_cli.ipynb`](08_cli.ipynb) | `python -m pyterraplot` command-line rendering |

## Setup

```bash
pip install 'pyterraplot[all]'   # serve + cf + raster extras unlock notebooks 07, 01-bonus, 06
```

Optional extras per notebook:

- **07** (live server) needs `pyterraplot[serve]` (fastapi + uvicorn).
- **06** (GeoTIFF/COG) needs `pyterraplot[raster]` (rioxarray).
- **01** (CF auto-detection bonus cell) needs `pyterraplot[cf]` (cf_xarray) — degrades gracefully if absent.

### The terraplot JS bundle (notebooks 00, 02, 03, 05)

The HTML/globe exporters inline the terraplot JavaScript bundle. It is auto-detected from a
sibling checkout at `../terraplot/dist/terraplot.js`. Otherwise point an env var at it before
launching JupyterLab:

```bash
export TERRAPLOT_BUNDLE=/path/to/terraplot/dist/terraplot.js
jupyter lab
```

### Kernel

These notebooks are saved against a kernel named **`Python (pyterraplot)`**. Pick any kernel whose
Python has `pyterraplot` importable. If you don't have one, register the env that does:

```bash
python -m ipykernel install --user --name pyterraplot --display-name "Python (pyterraplot)"
```

## Generated files

Running the notebooks writes `*.html`, `*.json`, `*.tif`, and `*.nc` into this directory
(open the HTML files in a browser, or view them inline via the `IFrame` cells). They are
git-ignored — delete them anytime; re-running regenerates them.
