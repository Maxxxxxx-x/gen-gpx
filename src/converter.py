from .gpx_processor import TrackPoint, DeltaPoint
from datetime import timedelta
from typing import List
import gpxpy.gpx


def deltas_to_gpx(
        deltas: List[DeltaPoint],
        start_point: TrackPoint
) -> gpxpy.gpx.GPX:
    gpx = gpxpy.gpx.GPX()
    track = gpxpy.gpx.GPXTrack()
    gpx.tracks.append(track)
    segment = gpxpy.gpx.GPXTrackSegment()
    track.segments.append(segment)

    curr_lat = start_point["lat"]
    curr_lon = start_point["lon"]
    curr_ele = start_point["ele"] or 0
    curr_time = start_point["time"]

    for delta in deltas:
        curr_lat += delta["d_lat"]
        curr_lon += delta["d_lon"]
        curr_ele += delta["d_ele"]
        curr_time += timedelta(seconds=delta["d_t"])
        pt = gpxpy.gpx.GPXTrackPoint(
            curr_lat,
            curr_lon,
            elevation=curr_ele,
            time=curr_time
        )
        segment.points.append(pt)
    return gpx
