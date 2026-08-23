"""
Tests for pyterraplot plotting features:
  - Binary encoding/decoding (binary.py)
  - HTML export with 2D projections (GeoMap)
  - Animation HTML export (frames_to_html)
  - Binary vs JSON size comparison
  - Accessor parameters: projection, extent, center, coastlines, binary
"""
from __future__ import annotations

import base64
import gzip
import struct
import math

import numpy as np
import pytest
import xarray as xr

import pyterraplot
from pyterraplot.binary import pack_field, pack_frames, MAGIC_FIELD, MAGIC_FRAMES


# ── Fixtures ───────────────────────────────────────────────────────────────────

def make_da(nlat=20, nlon=40, name="t2m", units="K", long_name="2m temperature",
            nan_fraction=0.0, lon_start=-180.0, lon_end=180.0, seed=0):
    """Synthetic 2D DataArray on a regular lat/lon grid."""
    lats = np.linspace(90, -90, nlat)
    lons = np.linspace(lon_start, lon_end, nlon)
    rng  = np.random.default_rng(seed)
    data = rng.standard_normal((nlat, nlon)).astype(np.float32) * 5 + 288
    if nan_fraction > 0:
        mask = rng.random((nlat, nlon)) < nan_fraction
        data[mask] = np.nan
    return xr.DataArray(
        data, dims=["lat", "lon"],
        coords={"lat": lats, "lon": lons},
        name=name,
        attrs={"units": units, "long_name": long_name},
    )


def make_time_da(ntime=6, nlat=18, nlon=36, seed=1):
    """Synthetic 3D DataArray for animation tests."""
    lats  = np.linspace(90, -90, nlat)
    lons  = np.linspace(-180, 180, nlon)
    times = np.arange(ntime)
    rng   = np.random.default_rng(seed)
    data  = rng.standard_normal((ntime, nlat, nlon)).astype(np.float32)
    return xr.DataArray(
        data, dims=["time", "lat", "lon"],
        coords={"time": times, "lat": lats, "lon": lons},
        name="t2m",
        attrs={"units": "K", "long_name": "S2S temperature"},
    )


# ── Binary helpers (Python-side decode for round-trip tests) ───────────────────

def _decode_field_binary(b64: str) -> dict:
    """Python round-trip decoder mirroring unpack.js unpackField."""
    raw = gzip.decompress(base64.b64decode(b64))
    off = 0

    magic, version, nlon, nlat = struct.unpack_from("<IIII", raw, off)
    off += 16

    lons  = np.frombuffer(raw, dtype="<f4", count=nlon, offset=off).copy(); off += nlon * 4
    lats  = np.frombuffer(raw, dtype="<f4", count=nlat, offset=off).copy(); off += nlat * 4
    field = np.frombuffer(raw, dtype="<f4", count=nlat * nlon, offset=off).copy()
    off += nlat * nlon * 4

    meta_len, = struct.unpack_from("<I", raw, off); off += 4
    import json
    meta = json.loads(raw[off: off + meta_len].decode())

    return dict(magic=magic, version=version, nlon=nlon, nlat=nlat,
                lons=lons, lats=lats, field=field.reshape(nlat, nlon), **meta)


def _decode_frames_binary(b64: str) -> dict:
    """Python round-trip decoder for pack_frames output."""
    raw = gzip.decompress(base64.b64decode(b64))
    off = 0

    magic, version, nlon, nlat, n_frames = struct.unpack_from("<IIIII", raw, off)
    off += 20

    lons = np.frombuffer(raw, dtype="<f4", count=nlon, offset=off).copy(); off += nlon * 4
    lats = np.frombuffer(raw, dtype="<f4", count=nlat, offset=off).copy(); off += nlat * 4

    coord_buf_len, = struct.unpack_from("<I", raw, off); off += 4
    coord_buf = raw[off: off + coord_buf_len]; off += coord_buf_len

    coord_values = []
    coff = 0
    for _ in range(n_frames):
        str_len, = struct.unpack_from("<H", coord_buf, coff); coff += 2
        coord_values.append(coord_buf[coff: coff + str_len].decode()); coff += str_len

    all_fields = np.frombuffer(raw, dtype="<f4", count=n_frames * nlat * nlon,
                               offset=off).copy()
    off += n_frames * nlat * nlon * 4

    meta_len, = struct.unpack_from("<I", raw, off); off += 4
    import json
    meta = json.loads(raw[off: off + meta_len].decode())

    fields = all_fields.reshape(n_frames, nlat, nlon)
    return dict(magic=magic, version=version, nlon=nlon, nlat=nlat,
                n_frames=n_frames, lons=lons, lats=lats,
                fields=fields, coord_values=coord_values, **meta)


# ── Tests: binary encoding ─────────────────────────────────────────────────────

class TestBinaryEncoding:

    def test_pack_field_returns_string(self):
        da = make_da()
        b64 = pack_field(da.tp.to_dict())
        assert isinstance(b64, str)

    def test_pack_field_is_valid_base64_gzip(self):
        b64 = pack_field(make_da().tp.to_dict())
        raw = gzip.decompress(base64.b64decode(b64))
        assert len(raw) > 0

    def test_pack_field_magic(self):
        b64 = pack_field(make_da().tp.to_dict())
        dec = _decode_field_binary(b64)
        assert dec["magic"] == MAGIC_FIELD

    def test_pack_field_dimensions_round_trip(self):
        da  = make_da(nlat=24, nlon=48)
        b64 = pack_field(da.tp.to_dict())
        dec = _decode_field_binary(b64)
        assert dec["nlon"] == 48
        assert dec["nlat"] == 24
        assert dec["field"].shape == (24, 48)

    def test_pack_field_values_preserved(self):
        da  = make_da(nlat=10, nlon=20)
        payload = da.tp.to_dict()
        b64 = pack_field(payload)
        dec = _decode_field_binary(b64)
        # float32 precision: max abs error < 0.001 K
        original = np.array(payload["field"], dtype="<f4")
        np.testing.assert_allclose(dec["field"], original, atol=1e-3)

    def test_pack_field_nan_preserved(self):
        da  = make_da(nan_fraction=0.15)
        b64 = pack_field(da.tp.to_dict())
        dec = _decode_field_binary(b64)
        nan_count = np.isnan(dec["field"]).sum()
        assert nan_count > 0

    def test_pack_field_metadata(self):
        da  = make_da(name="tp", units="mm/day", long_name="Total precip")
        b64 = pack_field(da.tp.to_dict())
        dec = _decode_field_binary(b64)
        assert dec["name"] == "tp"
        assert dec["units"] == "mm/day"
        assert dec["long_name"] == "Total precip"

    def test_pack_field_lons_ascending(self):
        da  = make_da(lon_start=0.0, lon_end=360.0)
        b64 = pack_field(da.tp.to_dict(wrap_lon=True))
        dec = _decode_field_binary(b64)
        # After wrap_lon, lons should be non-decreasing.
        # Strictly-ascending is not guaranteed when lon=0 and lon=360 both wrap to 0.
        assert np.all(np.diff(dec["lons"]) >= 0)
        assert dec["lons"][0] >= -180.0
        assert dec["lons"][-1] <= 180.0

    def test_pack_field_smaller_than_json(self):
        import json
        da  = make_da(nlat=36, nlon=72)
        payload = da.tp.to_dict()
        json_size   = len(json.dumps(payload).encode())
        binary_size = len(base64.b64decode(pack_field(payload)))  # compressed size
        # Compressed binary should be substantially smaller
        assert binary_size < json_size * 0.8

    def test_pack_frames_magic(self):
        da  = make_time_da(ntime=4)
        compact = da.tp.frames_compact(dim="time")
        b64 = pack_frames(compact)
        dec = _decode_frames_binary(b64)
        assert dec["magic"] == MAGIC_FRAMES

    def test_pack_frames_dimensions(self):
        da  = make_time_da(ntime=5, nlat=18, nlon=36)
        b64 = pack_frames(da.tp.frames_compact(dim="time"))
        dec = _decode_frames_binary(b64)
        assert dec["n_frames"] == 5
        assert dec["nlon"]     == 36
        assert dec["nlat"]     == 18
        assert dec["fields"].shape == (5, 18, 36)

    def test_pack_frames_values_preserved(self):
        da  = make_time_da(ntime=3, nlat=10, nlon=20)
        compact = da.tp.frames_compact(dim="time")
        b64 = pack_frames(compact)
        dec = _decode_frames_binary(b64)
        for k, f in enumerate(compact["frames"]):
            orig = np.array(f["field"], dtype="<f4")
            np.testing.assert_allclose(dec["fields"][k], orig, atol=1e-3)

    def test_pack_frames_coord_values(self):
        da  = make_time_da(ntime=4)
        compact = da.tp.frames_compact(dim="time")
        b64 = pack_frames(compact)
        dec = _decode_frames_binary(b64)
        expected = [str(f["coord_value"]) for f in compact["frames"]]
        assert dec["coord_values"] == expected

    def test_pack_frames_smaller_than_json_frames(self):
        import json
        da = make_time_da(ntime=6, nlat=36, nlon=72)
        compact = da.tp.frames_compact(dim="time")
        json_size   = len(json.dumps(compact).encode())
        binary_b64  = pack_frames(compact)
        binary_size = len(base64.b64decode(binary_b64))
        assert binary_size < json_size * 0.8

    def test_pack_frames_metadata(self):
        da  = make_time_da()
        b64 = pack_frames(da.tp.frames_compact(dim="time"))
        dec = _decode_frames_binary(b64)
        assert dec["name"] == "t2m"
        assert dec["units"] == "K"


# ── Tests: to_html projections ────────────────────────────────────────────────

class TestToHtmlProjections:

    PROJECTIONS = [
        "equirectangular", "platecarree",
        "mercator", "orthographic", "naturalEarth",
        "stereographic", "azimuthalEqualArea",
        "albers", "lambertConformal",
    ]

    @pytest.mark.parametrize("proj", PROJECTIONS)
    def test_projection_name_accepted(self, tmp_path, proj):
        da  = make_da()
        out = da.tp.to_html(tmp_path / f"{proj}.html", projection=proj,
                            title=proj, binary=False)
        content = out.read_text()
        assert "GeoMap" in content
        assert proj in content

    def test_globe_uses_geosphere(self, tmp_path):
        da  = make_da()
        out = da.tp.to_html(tmp_path / "globe.html", projection=None)
        content = out.read_text()
        assert "new GeoSphere" in content
        assert "new GeoMap" not in content.split("const map")[1]

    def test_2d_uses_geomap(self, tmp_path):
        da  = make_da()
        out = da.tp.to_html(tmp_path / "map.html", projection="mercator")
        assert "new GeoMap" in out.read_text()

    def test_binary_payload_embedded(self, tmp_path):
        da  = make_da()
        out = da.tp.to_html(tmp_path / "bin.html", projection="mercator", binary=True)
        content = out.read_text()
        assert "await unpackField(" in content
        assert "H4sI" in content  # gzip base64 starts with this

    def test_json_payload_when_binary_false(self, tmp_path):
        da  = make_da()
        out = da.tp.to_html(tmp_path / "json.html", projection="mercator", binary=False)
        content = out.read_text()
        assert '"lons"' in content
        assert "await unpackField" not in content

    def test_binary_html_smaller_than_json_html(self, tmp_path):
        da   = make_da(nlat=36, nlon=72)
        out_bin  = da.tp.to_html(tmp_path / "bin.html",  projection="mercator", binary=True)
        out_json = da.tp.to_html(tmp_path / "json.html", projection="mercator", binary=False)
        assert out_bin.stat().st_size < out_json.stat().st_size

    def test_coastlines_added_by_default(self, tmp_path):
        da  = make_da()
        out = da.tp.to_html(tmp_path / "cl.html", projection="mercator", binary=False)
        assert "addFeature('coastlines'" in out.read_text()

    def test_coastlines_suppressed(self, tmp_path):
        da  = make_da()
        out = da.tp.to_html(tmp_path / "nocl.html", projection="mercator",
                            coastlines=False, binary=False)
        assert "addFeature('coastlines'" not in out.read_text()

    def test_globe_coastlines_neon(self, tmp_path):
        da  = make_da()
        out = da.tp.to_html(tmp_path / "neon.html", binary=False,
                            earth_surface="none", coastlines=True, spin=False)
        html = out.read_text()
        assert "earthSurface: 'none'" in html
        assert "addFeature('coastlines', { color: '#39FF14'" in html
        assert "autoRotate: false" in html

    def test_center_embedded_in_js(self, tmp_path):
        da  = make_da()
        out = da.tp.to_html(tmp_path / "centered.html", projection="orthographic",
                            center=(20, 45), binary=False)
        content = out.read_text()
        assert "20" in content
        assert "45" in content

    def test_extent_embedded_in_js(self, tmp_path):
        da  = make_da()
        out = da.tp.to_html(tmp_path / "europe.html", projection="mercator",
                            extent=(-30, 45, 30, 75), binary=False)
        content = out.read_text()
        assert "extent" in content
        assert "-30" in content

    def test_contourf_kind(self, tmp_path):
        da  = make_da()
        out = da.tp.to_html(tmp_path / "cf.html", projection="mercator",
                            kind="contourf", levels=8, binary=False)
        content = out.read_text()
        assert "contourf" in content
        assert "levels: 8" in content

    def test_pcolormesh_kind(self, tmp_path):
        da  = make_da()
        out = da.tp.to_html(tmp_path / "pcm.html", projection="naturalEarth", binary=False)
        content = out.read_text()
        assert "pcolormesh" in content
        assert "levels: null" in content

    @pytest.mark.parametrize("cmap", [
        "viridis", "plasma", "RdBu_r", "RdYlBu_r", "Spectral_r",
        "Greys", "YlGnBu", "PuBuGn", "BrBG",
        "thermal", "haline", "ice", "balance", "speed", "topo",   # cmocean
        "thermal_r", "oxy_r",                                         # cmocean reversed
    ])
    def test_colormap_names_accepted(self, tmp_path, cmap):
        da  = make_da()
        out = da.tp.to_html(tmp_path / f"{cmap}.html", projection="mercator",
                            cmap=cmap, binary=False)
        assert cmap in out.read_text()

    def test_vmin_vmax_in_output(self, tmp_path):
        da  = make_da()
        out = da.tp.to_html(tmp_path / "range.html", projection="mercator",
                            vmin=280.0, vmax=300.0, binary=False)
        content = out.read_text()
        assert "280.0" in content
        assert "300.0" in content

    def test_label_uses_long_name_units(self, tmp_path):
        da  = make_da(long_name="Sea level pressure", units="hPa", name="slp")
        out = da.tp.to_html(tmp_path / "label.html", projection="mercator")
        content = out.read_text()
        assert "Sea level pressure" in content
        assert "[hPa]" in content

    def test_title_in_page(self, tmp_path):
        da  = make_da()
        out = da.tp.to_html(tmp_path / "titled.html", title="My Plot",
                            projection="mercator")
        assert "<title>My Plot</title>" in out.read_text()


# ── Tests: frames_to_html ─────────────────────────────────────────────────────

class TestFramesToHtml:

    def test_creates_file(self, tmp_path):
        da  = make_time_da(ntime=4)
        out = da.tp.frames_to_html(tmp_path / "anim.html", dim="time")
        assert out.exists()

    def test_uses_unpack_frames(self, tmp_path):
        da  = make_time_da(ntime=3)
        out = da.tp.frames_to_html(tmp_path / "anim.html", dim="time")
        content = out.read_text()
        assert "await unpackFrames(" in content
        assert "H4sI" in content

    def test_correct_frame_count_in_scrubber(self, tmp_path):
        ntime = 7
        da  = make_time_da(ntime=ntime)
        out = da.tp.frames_to_html(tmp_path / "anim.html", dim="time")
        # max attribute of scrubber should be ntime - 1
        assert f'max="{ntime - 1}"' in out.read_text()

    def test_play_pause_button_present(self, tmp_path):
        da  = make_time_da()
        out = da.tp.frames_to_html(tmp_path / "anim.html", dim="time")
        content = out.read_text()
        assert "play-btn" in content
        assert "scrubber" in content
        assert "frame-label" in content

    def test_animation_uses_geomap_with_projection(self, tmp_path):
        da  = make_time_da()
        out = da.tp.frames_to_html(tmp_path / "anim.html", dim="time",
                                   projection="mercator")
        assert "new GeoMap" in out.read_text()

    def test_animation_uses_geosphere_without_projection(self, tmp_path):
        da  = make_time_da()
        out = da.tp.frames_to_html(tmp_path / "anim.html", dim="time")
        assert "new GeoSphere" in out.read_text()

    def test_interval_embedded(self, tmp_path):
        da  = make_time_da()
        out = da.tp.frames_to_html(tmp_path / "anim.html", dim="time",
                                   interval=1200)
        assert "1200" in out.read_text()

    def test_colorbar_js_present(self, tmp_path):
        da  = make_time_da()
        out = da.tp.frames_to_html(tmp_path / "anim.html", dim="time")
        content = out.read_text()
        assert "drawColorbar" in content
        assert "cbar-ticks" in content

    def test_extent_in_animation(self, tmp_path):
        da  = make_time_da()
        out = da.tp.frames_to_html(tmp_path / "anim.html", dim="time",
                                   projection="mercator", extent=(-30, 45, 35, 72))
        assert "extent" in out.read_text()

    def test_animation_binary_smaller_than_json_would_be(self, tmp_path):
        """The binary blob should compress 6 frames significantly."""
        import json
        da      = make_time_da(ntime=6, nlat=36, nlon=72)
        compact = da.tp.frames_compact(dim="time")
        json_size = len(json.dumps(compact).encode())
        b64  = pack_frames(compact)
        binary_size = len(base64.b64decode(b64))
        assert binary_size < json_size * 0.85


# ── Tests: edge cases and regression ─────────────────────────────────────────

class TestEdgeCases:

    def test_all_nan_field_binary(self):
        """pack_field with all-NaN field should not raise."""
        da = make_da()
        da_nan = xr.DataArray(
            np.full((20, 40), np.nan),
            dims=["lat", "lon"],
            coords={"lat": da.lat.values, "lon": da.lon.values},
            attrs={"units": "K"},
        )
        b64 = pack_field(da_nan.tp.to_dict())
        dec = _decode_field_binary(b64)
        assert np.all(np.isnan(dec["field"]))

    def test_single_frame_animation(self, tmp_path):
        da = make_time_da(ntime=1)
        out = da.tp.frames_to_html(tmp_path / "single.html", dim="time")
        assert 'max="0"' in out.read_text()

    def test_descending_lats_preserved(self):
        """Lats stored descending in binary should come back in the same order."""
        da  = make_da()  # make_da uses descending lats (90 → -90)
        b64 = pack_field(da.tp.to_dict())
        dec = _decode_field_binary(b64)
        # lats should be descending (90 → -90)
        assert dec["lats"][0] > dec["lats"][-1]

    def test_ascending_lats_preserved(self):
        """Lats stored ascending in binary should come back ascending."""
        lats = np.linspace(-90, 90, 20)
        lons = np.linspace(-180, 180, 40)
        da = xr.DataArray(
            np.ones((20, 40)), dims=["lat", "lon"],
            coords={"lat": lats, "lon": lons},
        )
        b64 = pack_field(da.tp.to_dict())
        dec = _decode_field_binary(b64)
        assert dec["lats"][0] < dec["lats"][-1]

    def test_regional_grid_binary(self):
        """Sub-global grid (e.g., Europe) round-trips correctly."""
        lats = np.linspace(75, 25, 51)
        lons = np.linspace(-30, 45, 76)
        da = xr.DataArray(
            np.random.default_rng(9).standard_normal((51, 76)).astype(np.float32),
            dims=["lat", "lon"], coords={"lat": lats, "lon": lons},
            attrs={"units": "K"},
        )
        b64 = pack_field(da.tp.to_dict())
        dec = _decode_field_binary(b64)
        assert dec["nlon"] == 76
        assert dec["nlat"] == 51
        assert math.isclose(dec["lons"][0], -30.0, abs_tol=0.1)
        assert math.isclose(dec["lats"][0],  75.0, abs_tol=0.1)

    def test_large_grid_binary_compression_ratio(self):
        """Global 1° grid: binary should be at least 2× smaller than JSON."""
        import json
        nlat, nlon = 181, 360
        lats = np.linspace(90, -90, nlat)
        lons = np.linspace(-180, 179, nlon)
        # Smooth cosine field (compresses better than random)
        lon2d, lat2d = np.meshgrid(lons, lats)
        data = (5 * np.sin(np.radians(lat2d)) * np.cos(np.radians(lon2d))).astype(np.float32)
        da = xr.DataArray(data, dims=["lat", "lon"],
                          coords={"lat": lats, "lon": lons})
        payload = da.tp.to_dict()
        json_size   = len(json.dumps(payload).encode())
        binary_size = len(base64.b64decode(pack_field(payload)))
        ratio = json_size / binary_size
        assert ratio >= 2.0, f"Expected ≥2× compression, got {ratio:.2f}×"

    @pytest.mark.parametrize("kind", ["pcolormesh", "contourf"])
    def test_kind_accepted(self, tmp_path, kind):
        da  = make_da()
        out = da.tp.to_html(tmp_path / f"{kind}.html", kind=kind,
                            projection="equirectangular")
        assert out.exists()

    def test_invalid_kind_raises(self, tmp_path):
        da = make_da()
        with pytest.raises(ValueError, match="kind"):
            da.tp.to_html(tmp_path / "bad.html", kind="scatter")


# ── Colormap registry (cmocean + matplotlib/d3) ────────────────────────────────

class TestColormapRegistry:
    def test_cmocean_complete(self):
        """All 22 cmocean colormaps + reversals are registered."""
        from pyterraplot.colormaps import CMOCEAN, CMOCEAN_REVERSED, ALL_COLORMAPS
        assert len(CMOCEAN) == 22
        assert len(CMOCEAN_REVERSED) == 22
        for name in (*CMOCEAN, *CMOCEAN_REVERSED):
            assert name in ALL_COLORMAPS

    def test_cmocean_matches_reference_set(self):
        """The canonical cmocean names from matplotlib.org/cmocean."""
        from pyterraplot.colormaps import CMOCEAN
        reference = {
            "algae", "amp", "balance", "curl", "deep", "delta", "dense",
            "diff", "gray", "haline", "ice", "matter", "oxy", "phase",
            "rain", "solar", "speed", "tarn", "tempo", "thermal", "topo",
            "turbid",
        }
        assert set(CMOCEAN) == reference

    def test_options_html_includes_groups(self):
        from pyterraplot.colormaps import cmap_options_html
        html = cmap_options_html(selected="thermal")
        assert '<optgroup label="cmocean (oceanographic)">' in html
        assert '<option value="thermal" selected>thermal</option>' in html
        assert '<option value="viridis">viridis</option>' in html

    def test_is_valid_cmap(self):
        from pyterraplot.colormaps import is_valid_cmap
        assert is_valid_cmap("thermal")
        assert is_valid_cmap("haline_r")
        assert not is_valid_cmap("not_a_cmap")
