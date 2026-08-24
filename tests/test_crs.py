import re
from pathlib import Path

import pytest

import pyterraplot as tp
import pyterraplot.crs as ccrs


class TestCRSBasics:
    def test_platecarree_defaults(self):
        crs = ccrs.PlateCarree()
        assert crs.tp_name == "equirectangular"
        assert crs.center == (0.0, 0.0)
        assert crs.to_js() == {"projection": "equirectangular", "center": [0.0, 0.0]}

    def test_central_longitude_becomes_center(self):
        crs = ccrs.Robinson(central_longitude=-100)
        assert crs.to_js()["center"] == [-100.0, 0.0]

    def test_orthographic_two_angles(self):
        crs = ccrs.Orthographic(central_longitude=-95, central_latitude=60)
        assert crs.to_js()["center"] == [-95.0, 60.0]

    def test_repr_roundtrips_params(self):
        crs = ccrs.Orthographic(central_longitude=-95, central_latitude=60)
        assert "central_longitude=-95.0" in repr(crs)
        assert "central_latitude=60.0" in repr(crs)

    def test_equality_and_hash(self):
        assert ccrs.Mollweide(10) == ccrs.Mollweide(10)
        assert ccrs.Mollweide(10) != ccrs.Mollweide(20)
        assert ccrs.Mollweide(10) != ccrs.Robinson(10)
        assert len({ccrs.Mollweide(10), ccrs.Mollweide(10)}) == 1

    def test_exported_at_top_level_too(self):
        assert tp.PlateCarree is ccrs.PlateCarree
        assert tp.NorthPolarStereo is ccrs.NorthPolarStereo


class TestPolarAndConic:
    def test_north_polar_stereo(self):
        crs = ccrs.NorthPolarStereo(central_longitude=-100)
        assert crs.tp_name == "stereographic"
        assert crs.center == (-100.0, 90.0)
        assert crs.default_extent == (-180.0, 180.0, 45.0, 90.0)

    def test_south_polar_stereo(self):
        assert ccrs.SouthPolarStereo().center == (0.0, -90.0)
        assert ccrs.SouthPolarStereo().default_extent[2] == -90.0

    def test_conic_emits_parallels(self):
        crs = ccrs.LambertConformal(standard_parallels=(33, 45))
        js = crs.to_js()
        assert js["projection"] == "conicconformal"
        assert js["parallels"] == [33.0, 45.0]

    def test_conic_rejects_wrong_parallel_count(self):
        with pytest.raises(ValueError, match="2-tuple"):
            ccrs.AlbersEqualArea(standard_parallels=(20, 30, 40))

    def test_lambert_conformal_cartopy_defaults(self):
        crs = ccrs.LambertConformal()
        assert crs.center == (-96.0, 39.0)
        assert crs.standard_parallels == (33.0, 45.0)

    def test_mercator_default_extent_clips_poles(self):
        assert ccrs.Mercator().default_extent == (-180.0, 180.0, -80.0, 84.0)


class TestSpecialCRS:
    def test_nearside_perspective_converts_height(self):
        js = ccrs.NearsidePerspective(satellite_height=6_378_137.0).to_js()
        assert js["distance"] == pytest.approx(2.0)

    def test_geodetic_is_transform_only(self):
        assert ccrs.Geodetic().is_transform_only
        with pytest.raises(TypeError, match="transform"):
            ccrs.Geodetic().to_js()

    def test_globe3d_is_globe(self):
        assert ccrs.Globe3D().is_globe
        with pytest.raises(TypeError, match="GeoSphere"):
            ccrs.Globe3D().to_js()

    def test_rotated_pole_rotation_triple(self):
        js = ccrs.RotatedPole(pole_longitude=-162, pole_latitude=39.25).to_js()
        # d3 rotate([-pole_lon, 90 - pole_lat, 0]) carries the geographic point
        # (pole_lon, pole_lat) onto the projection's north pole.
        assert js["rotate"] == [162.0, 50.75, 0.0]
        assert "center" not in js

    def test_rotated_pole_rejects_central_rotated_longitude(self):
        with pytest.raises(NotImplementedError, match="central_rotated_longitude"):
            ccrs.RotatedPole(pole_longitude=10, pole_latitude=50,
                             central_rotated_longitude=15)


class TestLookup:
    @pytest.mark.parametrize("name,expected", [
        ("PlateCarree", ccrs.PlateCarree),
        ("plate_carree", ccrs.PlateCarree),
        ("north_polar_stereo", ccrs.NorthPolarStereo),
        ("equirectangular", ccrs.PlateCarree),   # by terraplot key
        ("robinson", ccrs.Robinson),
    ])
    def test_get_by_name(self, name, expected):
        assert isinstance(ccrs.get(name), expected)

    def test_get_passes_kwargs(self):
        assert ccrs.get("Mollweide", central_longitude=45).central_longitude == 45.0

    def test_get_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown projection"):
            ccrs.get("teapot")

    def test_registry_covers_public_classes(self):
        for name in ccrs.__all__:
            obj = getattr(ccrs, name)
            if isinstance(obj, type) and issubclass(obj, ccrs.CRS) and obj is not ccrs.CRS:
                assert name in ccrs.PROJECTIONS


def _bundle_projection_keys() -> set[str] | None:
    """Registry keys from the sibling terraplot source, if it is checked out."""
    src = (Path(__file__).resolve().parent.parent.parent
           / "terraplot" / "src" / "GeoMap.js")
    if not src.exists():
        return None
    text = src.read_text()
    block = re.search(r"const PROJ_REGISTRY = \{(.*?)\n\};", text, re.DOTALL)
    if not block:
        return None
    return set(re.findall(r"^\s*([a-z0-9]+):", block.group(1), re.MULTILINE))


def test_every_crs_resolves_in_the_terraplot_registry():
    """Each CRS must name a projection the browser can actually build.

    A silent mismatch here falls back to equirectangular at render time, which
    looks like a working plot in the wrong projection — the worst failure mode.
    """
    keys = _bundle_projection_keys()
    if keys is None:
        pytest.skip("sibling terraplot checkout not available")

    missing = {}
    for name, cls in ccrs.PROJECTIONS.items():
        if cls.is_globe or cls.is_transform_only:
            continue
        if cls.tp_name not in keys:
            missing[name] = cls.tp_name
    assert not missing, f"projections absent from terraplot's registry: {missing}"
