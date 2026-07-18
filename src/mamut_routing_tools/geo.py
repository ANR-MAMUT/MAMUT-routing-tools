"""Geodetic conversions matching OpenStreetMapX.jl exactly.

The Julia pipeline stores road nodes as ENU coordinates linearized around the
center of the OSM bounds, computes edge lengths as 3D Euclidean distances in
that frame, and converts back to LLA for polylines. The Python engine keeps
the same math so graph weights and rendered coordinates agree to floating
point noise rather than to a model difference.
"""

from __future__ import annotations

import math
from typing import NamedTuple

WGS84_A = 6378137.0
WGS84_B = 6356752.31424518
WGS84_E2 = 1.0 - (WGS84_B * WGS84_B) / (WGS84_A * WGS84_A)
WGS84_EP2 = (WGS84_A * WGS84_A) / (WGS84_B * WGS84_B) - 1.0


class LLA(NamedTuple):
    lat: float
    lon: float
    alt: float = 0.0


class ECEF(NamedTuple):
    x: float
    y: float
    z: float


class ENU(NamedTuple):
    east: float
    north: float
    up: float


def ecef_from_lla(lla: LLA) -> ECEF:
    lat_rad = math.radians(lla.lat)
    lon_rad = math.radians(lla.lon)
    sin_lat, cos_lat = math.sin(lat_rad), math.cos(lat_rad)
    sin_lon, cos_lon = math.sin(lon_rad), math.cos(lon_rad)
    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    return ECEF(
        (n + lla.alt) * cos_lat * cos_lon,
        (n + lla.alt) * cos_lat * sin_lon,
        (n * (1.0 - WGS84_E2) + lla.alt) * sin_lat,
    )


def lla_from_ecef(ecef: ECEF) -> LLA:
    x, y, z = ecef
    p = math.hypot(x, y)
    theta = math.atan2(z * WGS84_A, p * WGS84_B)
    lon = math.atan2(y, x)
    lat = math.atan2(
        z + WGS84_EP2 * WGS84_B * math.sin(theta) ** 3,
        p - WGS84_E2 * WGS84_A * math.cos(theta) ** 3,
    )
    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * math.sin(lat) ** 2)
    alt = p / math.cos(lat) - n
    return LLA(math.degrees(lat), math.degrees(lon), alt)


def enu_from_lla(lla: LLA, ref: LLA) -> ENU:
    ecef = ecef_from_lla(lla)
    ecef_ref = ecef_from_lla(ref)
    dx, dy, dz = ecef.x - ecef_ref.x, ecef.y - ecef_ref.y, ecef.z - ecef_ref.z
    lat_rad = math.radians(ref.lat)
    lon_rad = math.radians(ref.lon)
    sin_lat, cos_lat = math.sin(lat_rad), math.cos(lat_rad)
    sin_lon, cos_lon = math.sin(lon_rad), math.cos(lon_rad)
    east = -sin_lon * dx + cos_lon * dy
    north = -cos_lon * sin_lat * dx - sin_lon * sin_lat * dy + cos_lat * dz
    up = cos_lon * cos_lat * dx + sin_lon * cos_lat * dy + sin_lat * dz
    return ENU(east, north, up)


def lla_from_enu(enu: ENU, ref: LLA) -> LLA:
    lat_rad = math.radians(ref.lat)
    lon_rad = math.radians(ref.lon)
    sin_lat, cos_lat = math.sin(lat_rad), math.cos(lat_rad)
    sin_lon, cos_lon = math.sin(lon_rad), math.cos(lon_rad)
    east, north, up = enu
    dx = -sin_lon * east - cos_lon * sin_lat * north + cos_lon * cos_lat * up
    dy = cos_lon * east - sin_lon * sin_lat * north + sin_lon * cos_lat * up
    dz = cos_lat * north + sin_lat * up
    ecef_ref = ecef_from_lla(ref)
    return lla_from_ecef(ECEF(ecef_ref.x + dx, ecef_ref.y + dy, ecef_ref.z + dz))


def enu_distance(a: ENU, b: ENU) -> float:
    return math.sqrt((b.east - a.east) ** 2 + (b.north - a.north) ** 2 + (b.up - a.up) ** 2)


def bounds_center(min_lat: float, min_lon: float, max_lat: float, max_lon: float) -> LLA:
    lon_mid = (min_lon + max_lon) / 2.0
    lat_mid = (min_lat + max_lat) / 2.0
    if min_lon > max_lon:
        lon_mid = lon_mid - 180.0 if lon_mid > 0 else lon_mid + 180.0
    return LLA(lat_mid, lon_mid)


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2.0) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2.0) ** 2
    return 2.0 * r * math.asin(math.sqrt(a))
