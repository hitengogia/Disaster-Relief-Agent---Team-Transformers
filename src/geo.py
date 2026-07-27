"""
geo.py
------
Lightweight, offline city -> (lat, lon) lookup for map visualization.

This is deliberately NOT a geocoding API call: the whole point of the live
map / simulation feature is to render instantly, every time, with zero
network dependency and zero added latency or cost. All locations used in
data/resources.csv and data/needs.csv are covered exactly; a broader list
of major Indian cities is included so manually-typed locations in the
"Submit a Need" tab also resolve, with a substring-match fallback for
close variants (e.g. "bangalore" vs "bengaluru").

Returns None for genuinely unknown locations -- callers must handle that
gracefully (e.g. skip the map, but still show the need in the list).
"""

import hashlib
import math

CITY_COORDS = {
    "delhi": (28.6139, 77.2090),
    "mumbai": (19.0760, 72.8777),
    "chennai": (13.0827, 80.2707),
    "bengaluru": (12.9716, 77.5946),
    "bangalore": (12.9716, 77.5946),
    "kolkata": (22.5726, 88.3639),
    "hyderabad": (17.3850, 78.4867),
    "pune": (18.5204, 73.8567),
    "ahmedabad": (23.0225, 72.5714),
    "jaipur": (26.9124, 75.7873),
    "lucknow": (26.8467, 80.9462),
    "chandigarh": (30.7333, 76.7794),
    "bhopal": (23.2599, 77.4126),
    "patna": (25.5941, 85.1376),
    "surat": (21.1702, 72.8311),
    "nagpur": (21.1458, 79.0882),
    "kochi": (9.9312, 76.2673),
    "cochin": (9.9312, 76.2673),
    "guwahati": (26.1445, 91.7362),
    "bhubaneswar": (20.2961, 85.8245),
    "ranchi": (23.3441, 85.3096),
    "shimla": (31.1048, 77.1734),
    "dehradun": (30.3165, 78.0322),
    "amritsar": (31.6340, 74.8723),
    "kanpur": (26.4499, 80.3319),
    "indore": (22.7196, 75.8577),
    "srinagar": (34.0837, 74.7973),
    "guntur":( 16.3067, 80.4365),
    "vijayawada": (16.5062, 80.6480),
    "varanasi": (25.3176, 82.9739),
    "coimbatore": (11.0168, 76.9558),
    "visakhapatnam": (17.6868, 83.2185),
}


def get_coords(location: str):
    """
    Return (lat, lon) for a location string, or None if it can't be
    resolved. Case-insensitive exact match first, then a substring
    fallback so close variants still resolve (e.g. a typo or a locality
    name that contains a known city name).
    """
    if not location or not isinstance(location, str):
        return None

    key = location.strip().lower()
    if key in CITY_COORDS:
        return CITY_COORDS[key]

    for city, coords in CITY_COORDS.items():
        if city in key or key in city:
            return coords

    return None


def jitter_coords(lat: float, lon: float, seed, spread: float = 0.22):
    """
    Deterministically offset a (lat, lon) point by a small amount based on
    `seed` (e.g. a resource/need id). Two calls with the same seed always
    return the same offset point.

    Why this exists: every match in this app pairs a need with a resource
    in the SAME city (matching.py requires same_location), so a need's
    coordinates and its matched resource's coordinates are identical by
    construction. Drawing an arc/line between two identical points renders
    as nothing -- zero length. Jittering spreads multiple entries in one
    city into a small visible cluster, which both (a) makes overlapping
    dots for a city with several resources distinguishable, and (b) gives
    need->resource arcs actual, visible length.

    `spread` is in degrees (~0.22 deg is roughly 20-25km at these
    latitudes) -- enough to separate points at map zoom ~4-6 without
    moving them into a different city's territory.
    """
    h = int(hashlib.md5(str(seed).encode()).hexdigest(), 16)
    angle_deg = h % 360
    frac = 0.35 + ((h // 360) % 100) / 100 * 0.65  # vary distance, avoid all points on one ring
    angle = math.radians(angle_deg)

    dlat = spread * frac * math.cos(angle)
    # longitude degrees compress toward the poles; correct so the offset
    # looks like a consistent physical distance in both directions
    dlon = spread * frac * math.sin(angle) / max(math.cos(math.radians(lat)), 0.1)

    return (lat + dlat, lon + dlon)
