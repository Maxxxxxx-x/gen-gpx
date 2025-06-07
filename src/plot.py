import argparse
import os
import folium
import gpxpy
import random


def parseGPX(gpxFile) -> tuple[list[float], list[float]]:
    print(f'Processing {gpxFile}...')
    latitudes = []
    longitudes = []
    with open(gpxFile, 'r') as f:
        gpx = gpxpy.parse(f)
        for track in gpx.tracks:
            for segment in track.segments:
                for point in segment.points:
                    latitudes.append(point.latitude)
                    longitudes.append(point.longitude)
    return (latitudes, longitudes)


def main():
    parser = argparse.ArgumentParser(
        description='Visualize gpx.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'gpxs',
        type=str,
        help='Folder containing gpx files.'
    )
    args = parser.parse_args()
    gpxsPath = os.path.abspath(args.gpxs)
    gpxFiles = []
    for gpxFile in os.listdir(gpxsPath):
        if gpxFile.endswith('.gpx'):
            gpxFiles.append(os.path.join(gpxsPath, gpxFile))

    map = folium.Map(location=[23.83462548786052,
                     121.01649906097934], zoom_start=7)

    for gpxFile in gpxFiles:
        latitudes, longitudes = parseGPX(gpxFile)
        color = "%06x" % random.randint(0, 0xDDDDDD)
        folium.PolyLine(
            list(
                zip(latitudes, longitudes)
            ),
            color=f'#{color}',
            weight=2.5,
            opacity=1,
            popup=os.path.basename(gpxFile)
        ).add_to(map)
        folium.Marker(
            [latitudes[0], longitudes[0]],
            popup=f'{os.path.basename(gpxFile)} start', icon=folium.Icon(color='green')
        ).add_to(map)
        folium.Marker(
            [latitudes[-1], longitudes[-1]],
            popup=f'{os.path.basename(gpxFile)} end', icon=folium.Icon(color='red')
        ).add_to(map)

    map.save('map.html')


if __name__ == '__main__':
    main()
