"""Geodetic conversion invariants (WGS84 round trips, ENU frame)."""

from __future__ import annotations

import math

from mamut_routing_tools.geo import (
    LLA,
    bounds_center,
    ecef_from_lla,
    enu_distance,
    enu_from_lla,
    haversine_m,
    lla_from_ecef,
    lla_from_enu,
)


def test_lla_ecef_round_trip() -> None:
    for lla in (LLA(45.7578, 4.8351), LLA(-33.86, 151.21, 42.0), LLA(0.0, 0.0)):
        back = lla_from_ecef(ecef_from_lla(lla))
        assert math.isclose(back.lat, lla.lat, abs_tol=1e-9)
        assert math.isclose(back.lon, lla.lon, abs_tol=1e-9)
        assert math.isclose(back.alt, lla.alt, abs_tol=1e-6)


def test_enu_round_trip_and_distance() -> None:
    ref = LLA(45.0, 4.0)
    point = LLA(45.003, 4.004)
    enu = enu_from_lla(point, ref)
    back = lla_from_enu(enu, ref)
    assert math.isclose(back.lat, point.lat, abs_tol=1e-9)
    assert math.isclose(back.lon, point.lon, abs_tol=1e-9)
    # ENU Euclidean distance agrees with haversine to well under a percent
    # at city scale.
    origin = enu_from_lla(ref, ref)
    euclidean = enu_distance(origin, enu)
    great_circle = haversine_m(ref.lat, ref.lon, point.lat, point.lon)
    assert abs(euclidean - great_circle) / great_circle < 0.01


def test_bounds_center() -> None:
    center = bounds_center(44.0, 3.0, 46.0, 5.0)
    assert (center.lat, center.lon) == (45.0, 4.0)
