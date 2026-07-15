"""
pyterraplot CLI — render a NetCDF/Zarr/GRIB variable to a self-contained HTML.

Usage
-----
    python -m pyterraplot data.nc --var t2m --out t2m.html
    python -m pyterraplot data.nc --var precip --projection orthographic --cmap YlGnBu
    python -m pyterraplot data.nc --var t2m --animate time --interval 500 --out anim.html
    python -m pyterraplot data.nc --u u10 --v v10 --bg t2m --out winds.html
    python -m pyterraplot data.nc --compare model_t2m era5_t2m --out compare.html

Reads any format xarray supports — pass --engine if auto-detection fails.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _open(args):
    import xarray as xr
    kwargs = {}
    if args.engine:
        kwargs["engine"] = args.engine
    if args.path.endswith(".zarr") or args.path.endswith(".zarr/"):
        return xr.open_zarr(args.path, **kwargs)
    return xr.open_dataset(args.path, **kwargs)


def _maybe_select_2d(da, args):
    """Reduce extra dims via --isel key=value pairs, then auto-pick the first slice
    of any remaining non-lat/lon dimensions."""
    import re
    if args.isel:
        sel = {}
        for kv in args.isel:
            m = re.match(r"^([^=]+)=(\d+)$", kv)
            if not m:
                sys.exit(f"--isel must be 'dim=index', got {kv!r}")
            sel[m.group(1)] = int(m.group(2))
        da = da.isel(sel)

    keep = {"lat", "latitude", "y", "rlat", "lon", "longitude", "x", "rlon"}
    extra = [d for d in da.dims if d.lower() not in keep]
    for d in extra:
        if da.sizes[d] > 1 and not args.animate:
            print(f"  [info] auto-isel({d}=0); pass --isel {d}=N to choose another", file=sys.stderr)
        da = da.isel({d: 0})
    return da


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="pyterraplot", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", help="path to .nc / .zarr / .grib2 / .h5 etc.")
    p.add_argument("--var", help="single-variable name (use to_html / frames_to_html)")
    p.add_argument("--u", help="u-wind variable (use with --v for quiver)")
    p.add_argument("--v", help="v-wind variable")
    p.add_argument("--bg", help="background scalar variable for quiver overlay")
    p.add_argument("--compare", nargs=2, metavar=("A", "B"),
                   help="render two variables side-by-side")
    p.add_argument("--out", "-o", default=None, help="output HTML path (default <var>.html)")
    p.add_argument("--engine", help="xarray open engine (e.g. cfgrib, h5netcdf, zarr)")
    p.add_argument("--isel", nargs="*", default=[],
                   help="dim=index pairs to subset before plotting")
    p.add_argument("--animate", help="dim name to animate (e.g. time, lead_time)")
    p.add_argument("--kind", default="pcolormesh", choices=["pcolormesh", "contourf", "contour"])
    p.add_argument("--cmap", default="viridis")
    p.add_argument("--alpha", type=float, default=0.85)
    p.add_argument("--vmin", type=float, default=None)
    p.add_argument("--vmax", type=float, default=None)
    p.add_argument("--levels", type=int, default=12)
    p.add_argument("--projection", default=None,
                   help="2D projection (else 3D globe)")
    p.add_argument("--no-coastlines", action="store_true")
    p.add_argument("--center", nargs=2, type=float, default=(0.0, 0.0),
                   metavar=("LON", "LAT"))
    p.add_argument("--extent", nargs=4, type=float, default=None,
                   metavar=("LON0", "LON1", "LAT0", "LAT1"))
    p.add_argument("--interval", type=int, default=700, help="animation ms/frame")
    p.add_argument("--title", default="terraplot")
    p.add_argument("--symmetric", action="store_true",
                   help="(--compare only) force symmetric colour range")
    p.add_argument("--viewer", action="store_true",
                   help="Start the interactive visual NetCDF viewer server")
    p.add_argument("--port", type=int, default=8765,
                   help="Port to run the visual viewer server on (default 8765)")
    args = p.parse_args(argv)

    import pyterraplot  # noqa: F401  registers accessors

    if args.viewer:
        from pyterraplot.server import start_viewer
        start_viewer(args.path, port=args.port)
        return

    ds = _open(args)

    out = args.out or _default_out(args)

    if args.compare:
        ds.tp.compare_html(
            out,
            a=args.compare[0], b=args.compare[1],
            cmap=args.cmap, alpha=args.alpha,
            vmin=args.vmin, vmax=args.vmax, symmetric=args.symmetric,
            projection=args.projection or "equirectangular",
            coastlines=not args.no_coastlines,
            center=tuple(args.center),
            extent=tuple(args.extent) if args.extent else None,
            title=args.title,
        )
        print(f"wrote {out}")
        return

    if args.u and args.v:
        ds.tp.quiver_html(
            out, u=args.u, v=args.v, background=args.bg,
            cmap=args.cmap, alpha=args.alpha,
            vmin=args.vmin, vmax=args.vmax,
            projection=args.projection or "equirectangular",
            coastlines=not args.no_coastlines,
            center=tuple(args.center),
            extent=tuple(args.extent) if args.extent else None,
            title=args.title,
        )
        print(f"wrote {out}")
        return

    if not args.var:
        sys.exit("--var is required (or use --compare / --u --v)")
    da = ds[args.var]

    if args.animate:
        da = _maybe_select_2d(da, args)
        da.tp.frames_to_html(
            out,
            dim=args.animate, kind=args.kind, title=args.title,
            cmap=args.cmap, alpha=args.alpha,
            vmin=args.vmin, vmax=args.vmax, levels=args.levels,
            projection=args.projection,
            coastlines=not args.no_coastlines,
            center=tuple(args.center),
            extent=tuple(args.extent) if args.extent else None,
            interval=args.interval,
        )
    else:
        da = _maybe_select_2d(da, args)
        da.tp.to_html(
            out, kind=args.kind, title=args.title,
            cmap=args.cmap, alpha=args.alpha,
            vmin=args.vmin, vmax=args.vmax, levels=args.levels,
            projection=args.projection,
            coastlines=not args.no_coastlines,
            center=tuple(args.center),
            extent=tuple(args.extent) if args.extent else None,
        )

    print(f"wrote {out}")


def _default_out(args) -> str:
    if args.compare:
        return "compare.html"
    if args.u and args.v:
        return "winds.html"
    if args.var:
        return f"{args.var}.html"
    return "terraplot.html"


if __name__ == "__main__":
    main()
