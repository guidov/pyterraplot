"""Tests for pyterraplot serialization and accessor."""
import json
import math

import numpy as np
import pytest
import xarray as xr

import pyterraplot  # registers .tp accessor


def make_da(nlat=10, nlon=20, lon_start=-180.0, lon_end=180.0, nan_fraction=0.0):
    """Helper: synthetic 2D DataArray."""
    lats = np.linspace(90, -90, nlat)
    lons = np.linspace(lon_start, lon_end, nlon)
    data = np.random.default_rng(0).standard_normal((nlat, nlon)).astype(np.float32)
    if nan_fraction > 0:
        mask = np.random.default_rng(1).random((nlat, nlon)) < nan_fraction
        data[mask] = np.nan
    da = xr.DataArray(
        data,
        dims=["lat", "lon"],
        coords={"lat": lats, "lon": lons},
        name="t2m",
        attrs={"units": "K", "long_name": "2m temperature"},
    )
    return da


# ── serialize ─────────────────────────────────────────────────────────────────

class TestSerialize:
    def test_basic_shape(self):
        da = make_da(nlat=10, nlon=20)
        p = da.tp.to_dict()
        assert len(p["lats"]) == 10
        assert len(p["lons"]) == 20
        assert len(p["field"]) == 10
        assert len(p["field"][0]) == 20

    def test_metadata(self):
        da = make_da()
        p = da.tp.to_dict()
        assert p["name"] == "t2m"
        assert p["units"] == "K"
        assert p["long_name"] == "2m temperature"

    def test_lon_wrapping_360(self):
        da = make_da(lon_start=0.0, lon_end=360.0)
        p = da.tp.to_dict(wrap_lon=True)
        lons = p["lons"]
        assert min(lons) >= -180.0
        assert max(lons) <= 180.0

    def test_lon_no_wrap(self):
        da = make_da(lon_start=0.0, lon_end=360.0)
        p = da.tp.to_dict(wrap_lon=False)
        assert max(p["lons"]) > 180.0

    def test_nan_serialized_as_none(self):
        da = make_da(nan_fraction=0.2)
        p = da.tp.to_dict()
        flat = [v for row in p["field"] for v in row]
        none_count = sum(1 for v in flat if v is None)
        assert none_count > 0

    def test_no_nan_in_clean_field(self):
        da = make_da(nan_fraction=0.0)
        p = da.tp.to_dict()
        flat = [v for row in p["field"] for v in row]
        assert all(v is not None for v in flat)

    def test_to_json_file(self, tmp_path):
        da = make_da()
        out = da.tp.to_json(tmp_path / "field.json")
        assert out.exists()
        loaded = json.loads(out.read_text())
        assert "lons" in loaded and "lats" in loaded and "field" in loaded

    def test_ndim_check(self):
        da = make_da().expand_dims("time")
        with pytest.raises(ValueError, match="2D"):
            da.tp.to_dict()

    def test_lat_transpose_independence(self):
        da = make_da(nlat=8, nlon=16)
        da_T = da.transpose("lon", "lat")
        p1 = da.tp.to_dict()
        p2 = da_T.tp.to_dict()
        assert p1["field"] == p2["field"]

    def test_explicit_dim_names(self):
        lats = np.linspace(90, -90, 8)
        lons = np.linspace(-180, 180, 16)
        data = np.ones((8, 16))
        da = xr.DataArray(data, dims=["y", "x"], coords={"y": lats, "x": lons})
        p = da.tp.to_dict(lat_dim="y", lon_dim="x")
        assert len(p["lats"]) == 8

    def test_unknown_dim_raises(self):
        da = xr.DataArray(np.ones((4, 8)), dims=["a", "b"])
        with pytest.raises(ValueError, match="Cannot identify"):
            da.tp.to_dict()


# ── frames ────────────────────────────────────────────────────────────────────

class TestFrames:
    def make_time_da(self, ntime=4, nlat=8, nlon=16):
        lats = np.linspace(90, -90, nlat)
        lons = np.linspace(-180, 180, nlon)
        times = np.arange(ntime)
        data = np.random.default_rng(2).standard_normal((ntime, nlat, nlon))
        return xr.DataArray(
            data,
            dims=["time", "lat", "lon"],
            coords={"time": times, "lat": lats, "lon": lons},
            name="t2m",
            attrs={"units": "K"},
        )

    def test_frames_length(self):
        da = self.make_time_da(ntime=4)
        frames = da.tp.frames(dim="time")
        assert len(frames) == 4

    def test_frames_have_lons_lats(self):
        da = self.make_time_da()
        frames = da.tp.frames(dim="time")
        for f in frames:
            assert "lons" in f and "lats" in f and "field" in f

    def test_frames_coord_value(self):
        da = self.make_time_da(ntime=3)
        frames = da.tp.frames(dim="time")
        assert frames[0]["frame"] == 0
        assert frames[1]["frame"] == 1

    def test_frames_to_json(self, tmp_path):
        da = self.make_time_da()
        out = da.tp.frames_to_json(tmp_path / "frames.json", dim="time")
        loaded = json.loads(out.read_text())
        assert isinstance(loaded, list)
        assert len(loaded) == 4


# ── frames_compact ────────────────────────────────────────────────────────────

class TestFramesCompact:
    def make_time_da(self, ntime=5, nlat=6, nlon=12):
        lats = np.linspace(90, -90, nlat)
        lons = np.linspace(-180, 180, nlon)
        data = np.random.default_rng(3).standard_normal((ntime, nlat, nlon))
        return xr.DataArray(
            data,
            dims=["time", "lat", "lon"],
            coords={"time": np.arange(ntime), "lat": lats, "lon": lons},
            name="tp",
            attrs={"units": "mm/day"},
        )

    def test_top_level_keys(self):
        da = self.make_time_da()
        compact = da.tp.frames_compact(dim="time")
        assert set(compact.keys()) >= {"lons", "lats", "frames", "name", "units"}

    def test_lons_lats_not_duplicated(self):
        da = self.make_time_da(ntime=5)
        compact = da.tp.frames_compact(dim="time")
        assert len(compact["frames"]) == 5
        for frame in compact["frames"]:
            assert "lons" not in frame
            assert "lats" not in frame
            assert "field" in frame

    def test_frame_index_matches(self):
        da = self.make_time_da(ntime=3)
        compact = da.tp.frames_compact(dim="time")
        for i, frame in enumerate(compact["frames"]):
            assert frame["frame"] == i

    def test_compact_smaller_than_full(self):
        da = self.make_time_da(ntime=6, nlat=20, nlon=40)
        full = da.tp.frames(dim="time")
        compact = da.tp.frames_compact(dim="time")
        full_size = len(json.dumps(full))
        compact_size = len(json.dumps(compact))
        assert compact_size < full_size

    def test_compact_to_json(self, tmp_path):
        da = self.make_time_da()
        out = da.tp.frames_compact_to_json(tmp_path / "compact.json", dim="time")
        loaded = json.loads(out.read_text())
        assert "lons" in loaded
        assert "frames" in loaded


# ── to_html ───────────────────────────────────────────────────────────────────

class TestToHtml:
    def test_creates_file(self, tmp_path):
        da = make_da()
        out = da.tp.to_html(tmp_path / "globe.html")
        assert out.exists()

    def test_html_structure(self, tmp_path):
        da = make_da()
        out = da.tp.to_html(tmp_path / "globe.html", title="Test Globe", cmap="plasma")
        content = out.read_text()
        assert "<!DOCTYPE html>" in content
        assert "Test Globe" in content
        assert "plasma" in content
        assert "pcolormesh" in content

    def test_json_payload_embedded(self, tmp_path):
        da = make_da(nlat=4, nlon=8)
        out = da.tp.to_html(tmp_path / "globe.html")
        content = out.read_text()
        assert '"lons"' in content
        assert '"lats"' in content
        assert '"field"' in content

    def test_label_uses_long_name_and_units(self, tmp_path):
        da = make_da()
        out = da.tp.to_html(tmp_path / "globe.html")
        content = out.read_text()
        assert "2m temperature" in content
        assert "[K]" in content
