"""Tests for the spherical geometry and streamline helpers."""
import numpy as np
import pytest

from pyterraplot import geodesy
from pyterraplot.geodesy import EARTH_RADIUS_M, geodesic_circle, great_circle, streamlines


def haversine_m(a, b):
    """Great-circle distance in metres between two [lon, lat] points."""
    lon1, lat1 = np.deg2rad(a)
    lon2, lat2 = np.deg2rad(b)
    h = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(h))


class TestGeodesicCircle:
    def test_ring_is_closed(self):
        ring = geodesic_circle(0, 0, 500_000, n_samples=40)
        assert len(ring) == 41
        assert ring[0] == ring[-1]

    @pytest.mark.parametrize("lon,lat", [(0, 0), (-95, 60), (120, -45), (0, 85)])
    def test_every_point_is_the_requested_distance_away(self, lon, lat):
        radius = 500_000.0
        ring = geodesic_circle(lon, lat, radius, n_samples=60)
        dists = [haversine_m([lon, lat], p) for p in ring]
        assert np.allclose(dists, radius, rtol=1e-6), \
            f"radii ranged {min(dists):.1f}–{max(dists):.1f} m"

    def test_longitudes_stay_in_range(self):
        # A circle straddling the dateline must not emit longitudes past ±180.
        ring = geodesic_circle(179, 0, 400_000, n_samples=60)
        lons = [p[0] for p in ring]
        assert min(lons) >= -180.0 and max(lons) <= 180.0

    def test_large_radius_near_pole(self):
        ring = geodesic_circle(0, 89, 1_000_000, n_samples=40)
        assert all(-90.0 <= p[1] <= 90.0 for p in ring)


class TestGreatCircle:
    def test_endpoints_preserved(self):
        line = great_circle((-100, 40), (10, 50), n=16)
        assert line[0] == pytest.approx([-100, 40], abs=1e-9)
        assert line[-1] == pytest.approx([10, 50], abs=1e-9)

    def test_midpoint_bulges_poleward(self):
        # The great circle between two mid-latitude points at the same latitude
        # arcs poleward of the constant-latitude line — that is the whole point.
        line = great_circle((-120, 45), (-60, 45), n=21)
        assert line[len(line) // 2][1] > 45.0

    def test_densification_count(self):
        assert len(great_circle((0, 0), (10, 0), n=9)) >= 9

    def test_identical_points(self):
        assert great_circle((5, 5), (5, 5)) == [[5, 5], [5, 5]]

    def test_all_points_on_the_sphere(self):
        line = great_circle((-170, 20), (170, -20), n=25)
        assert all(-180 <= p[0] <= 180 and -90 <= p[1] <= 90 for p in line)


class TestStreamlines:
    @staticmethod
    def zonal_field(nlat=37, nlon=72):
        lats = np.linspace(-90, 90, nlat)
        lons = np.linspace(-180, 175, nlon)
        u = np.ones((nlat, nlon)) * 10.0     # pure eastward flow
        v = np.zeros((nlat, nlon))
        return lons, lats, u, v

    def test_pure_zonal_flow_stays_on_its_latitude(self):
        lons, lats, u, v = self.zonal_field()
        lines = streamlines(lons, lats, u, v, density=0.5, max_steps=200)
        assert lines, "no streamlines traced"
        for line in lines:
            band = [p[1] for p in line]
            assert max(band) - min(band) < 1.0, \
                "a zonal streamline drifted off its parallel"

    def test_zero_field_returns_nothing(self):
        lons, lats, u, v = self.zonal_field()
        assert streamlines(lons, lats, np.zeros_like(u), v) == []

    def test_all_nan_field_returns_nothing(self):
        lons, lats, u, v = self.zonal_field()
        assert streamlines(lons, lats, np.full_like(u, np.nan),
                           np.full_like(v, np.nan)) == []

    def test_density_increases_coverage(self):
        lons, lats, u, v = self.zonal_field()
        sparse = streamlines(lons, lats, u, v, density=0.5)
        dense = streamlines(lons, lats, u, v, density=2.0)
        assert len(dense) > len(sparse)

    def test_descending_latitudes_handled(self):
        # Most climate files store latitude north-to-south; the result should
        # not depend on that storage order.
        lons, lats, u, v = self.zonal_field()
        asc = streamlines(lons, lats, u, v, density=0.5)
        desc = streamlines(lons, lats[::-1], u[::-1], v[::-1], density=0.5)
        assert len(asc) == len(desc)

    def test_points_stay_inside_the_grid(self):
        lats = np.linspace(-60, 60, 25)
        lons = np.linspace(-90, 90, 37)
        LON, LAT = np.meshgrid(lons, lats)
        u = -np.sin(np.deg2rad(LAT))
        v = np.cos(np.deg2rad(LON))
        for line in streamlines(lons, lats, u, v, density=1.0):
            for lon, lat in line:
                assert -90.5 <= lon <= 90.5
                assert -60.5 <= lat <= 60.5

    def test_dateline_wrap_is_split(self):
        lons, lats, u, v = self.zonal_field()
        lines = streamlines(lons, lats, u, v, density=0.5, max_steps=400)
        for line in lines:
            jumps = [abs(b[0] - a[0]) for a, b in zip(line, line[1:])]
            assert not jumps or max(jumps) < 180.0, \
                "a streamline segment jumped the dateline"

    def test_meridional_flow_runs_north_south(self):
        lats = np.linspace(-60, 60, 25)
        lons = np.linspace(-90, 90, 37)
        u = np.zeros((25, 37))
        v = np.ones((25, 37)) * 5.0
        lines = streamlines(lons, lats, u, v, density=0.8, max_steps=300)
        assert lines
        for line in lines:
            spread = max(p[0] for p in line) - min(p[0] for p in line)
            assert spread < 1.0, "a meridional streamline drifted in longitude"


def test_module_exported():
    assert geodesy.EARTH_RADIUS_M > 6e6
