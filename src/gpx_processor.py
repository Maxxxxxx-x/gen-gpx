from typing import Tuple, List, Dict, TypeDict

from geopy.distance import geodesic
from datetime import datetime
import gpxpy
import math


class TrackPoint(TypeDict):
    lat: float
    lon: float
    ele: float | None
    time: datetime


class DeltaPoint(TypeDict):
    d_lat: float
    d_lon: float
    d_ele: float
    d_t: float
    spd: float
    slope: float


def parse_gpx(raw_gpx: str) -> List[TrackPoint]:
    gpx = gpxpy.parse(raw_gpx)

    points = []
    for track in gpx.tracks:
        for segment in track.segments:
            for pt in segment.points:
                points.append({
                    "lat": pt.latitude,
                    "lon": pt.longitude,
                    "ele": pt.elevation,
                    "time": pt.time
                })
    return points


def compute_deltas(points: List[TrackPoint]) -> List[DeltaPoint]:
    deltas = []
    for i in range(len(points) - 1):
        curr_pt = points[i]
        next_pt = points[i + 1]
        d_t = (next_pt.time - curr_pt.time).total_seconds()
        dist = geodesic((curr_pt.lat, curr_pt.lon),
                        (next_pt.lat, next_pt.lon)).meters
        spd = dist / d_t if d_t > 0 else 0
        d_ele = (next_pt.ele or 0) - (curr_pt.ele or 0)
        slope_angle = math.degrees(math.atan2(d_ele, dist)) if dist > 0 else 0

        deltas.append({
            "d_lat": next_pt.lat - curr_pt.lat,
            "d_lon": next_pt.lon - curr_pt.lon,
            "d_ele": d_ele,
            "d_t": d_t,
            "spd": spd,
            "slope": slope_angle
        })
    return deltas


def split_sequences(
        deltas: List[DeltaPoint],
        seq_len: int
) -> Tuple[List, List]:
    x = []
    y = []
    for i in range(len(deltas) - seq_len):
        seq_in = deltas[i:i + seq_len]
        target = deltas[i + seq_len]
        x.append([[delta.d_lat, delta.d_lon, delta.d_ele,
                 delta.d_t, delta.slope] for delta in seq_in])
        y.append([target.d_lat, target.d_Lon,
                 target.d_ele, target.d_t, target.slope])
    return (x, y)
