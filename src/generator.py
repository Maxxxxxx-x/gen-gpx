from gpx_processor import TrackPoint
from trainer import load_model, predict, get_device
from converter import deltas_to_gpx
from config import get_config
import os


def generate_gpx_route(
    start_point: TrackPoint,
    num_points: int = 100,
    model_path: str | None = None
) -> str:
    """生成 GPX 路線"""
    config = get_config()
    device = get_device()

    # 載入模型
    if model_path is None:
        model_path = os.path.join(config.checkpoint_dir, "best.pt")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = load_model(
        model_path,
        config.input_size,
        config.hidden_size,
        config.num_layers,
        device
    )

    # 初始化序列（可以用一些預設值或從訓練資料中取樣）
    initial_sequence = [[0.0001, 0.0001, 1.0, 30.0, 1.5, 0.1] for _ in range(config.sequence_length)]

    generated_deltas = []
    current_sequence = initial_sequence.copy()

    # 生成指定數量的點
    for _ in range(num_points):
        # 預測下一個增量
        next_delta_values = predict(model, current_sequence, device)

        # 創建 DeltaPoint
        next_delta = {
            "d_lat": next_delta_values[0],
            "d_lon": next_delta_values[1],
            "d_ele": next_delta_values[2],
            "d_t": next_delta_values[3],
            "spd": next_delta_values[4],
            "slope": next_delta_values[5]
        }

        generated_deltas.append(next_delta)

        # 更新序列（移除第一個，添加新預測的）
        current_sequence = current_sequence[1:] + [next_delta_values]

    # 轉換為 GPX
    gpx = deltas_to_gpx(generated_deltas, start_point)
    return gpx.to_xml()
