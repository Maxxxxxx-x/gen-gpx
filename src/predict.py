from generator import generate_gpx_route
from gpx_processor import TrackPoint
from datetime import datetime
import argparse


def main():
    parser = argparse.ArgumentParser(description='Generate GPX route using trained model')
    parser.add_argument('--start-lat', type=float, required=True, help='Starting latitude')
    parser.add_argument('--start-lon', type=float, required=True, help='Starting longitude')
    parser.add_argument('--start-ele', type=float, default=100.0, help='Starting elevation')
    parser.add_argument('--num-points', type=int, default=100, help='Number of points to generate')
    parser.add_argument('--model-path', type=str, help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='generated_route.gpx', help='Output GPX file')

    args = parser.parse_args()

    # 創建起始點
    start_point: TrackPoint = {
        "lat": args.start_lat,
        "lon": args.start_lon,
        "ele": args.start_ele,
        "time": datetime.now()
    }

    try:
        # 生成 GPX
        gpx_xml = generate_gpx_route(
            start_point=start_point,
            num_points=args.num_points,
            model_path=args.model_path
        )

        # 儲存到檔案
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(gpx_xml)

        print(f"Generated GPX route saved to: {args.output}")

    except Exception as e:
        print(f"Error generating GPX: {e}")


if __name__ == "__main__":
    main()
