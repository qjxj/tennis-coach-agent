"""
最小链路入口：视频 → 姿态 → 指标 → 标注视频 + JSON。

用法：
    python -m tennis_coach.run_minimal <video> [out_dir]
    python tennis_coach/run_minimal.py third_party/tennis-tracking/VideoInput/video_input3.mp4

产出（默认 data/runs/<视频名>/）：
    pose.json      逐帧关键点
    metrics.json   击球事件 + 指标 + 定义
    annotated.mp4  骨架与击球标注
    impact_*.jpg   每个击球点的关键帧
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tennis_coach.pose.extract import PoseExtractConfig, extract_pose, save_pose
from tennis_coach.pose.metrics import compute_metrics
from tennis_coach.pose.visualize import draw_video, make_contact_sheet


def video_quality(pose: dict) -> dict:
    """量化"主体在画面里有多大" —— 这是指标可不可信的第一前提。

    体高（像素）是最直接的可用性指标：低于 150px 时 MediaPipe 关键点
    抖动 1~2px，归一化后就会变成十几的假腕速，击球次数和指标全部失真。
    """
    ok = [f for f in pose["frames"] if f["ok"]]
    if not ok:
        return {"body_height_px": 0.0, "shoulder_width_px": 0.0, "usable_ratio": 0.0}

    h, w = pose["meta"]["height"], pose["meta"]["width"]
    return {
        "body_height_px": round(float(np.median([f["scale"] for f in ok])) * h, 1),
        "shoulder_width_px": round(
            float(np.median([f.get("shoulder_width", 0.0) for f in ok])) * w, 1),
        "usable_ratio": round(len(ok) / len(pose["frames"]), 3),
    }


def run(video_path: str, out_dir: str = None, model: str = "third_party/models/pose_landmarker_lite.task", wrist: str = None) -> dict:
    video = Path(video_path)
    if not video.exists():
        raise FileNotFoundError(f"视频不存在: {video}")

    out_root = Path(out_dir) if out_dir else Path("data/runs") / video.stem
    out_root.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    pose = extract_pose(str(video), PoseExtractConfig(model_path=model))
    pose_path = out_root / "pose.json"
    save_pose(pose, str(pose_path))
    print(f"[1/4] 姿态提取完成: {pose_path} "
          f"({sum(1 for f in pose['frames'] if f['ok'])}/{len(pose['frames'])} 帧可用, "
          f"{time.time() - t0:.1f}s)")

    metrics = compute_metrics(pose, wrist)
    metrics_path = out_root / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[2/4] 指标计算完成: {metrics_path}（检出 {len(metrics['events'])} 次击球）")

    video_out = out_root / "annotated.mp4"
    keyframes = draw_video(str(video), pose, metrics, str(video_out), str(out_root / "keyframes"))
    print(f"[3/4] 可视化完成: {video_out}（关键帧 {len(keyframes)} 张）")

    # 关键帧拼图：一眼核对"识别出来的每一拍是不是真的击球"
    sheet_path = make_contact_sheet(keyframes, str(out_root / "contact_sheet.jpg"))
    print(f"[4/4] 关键帧拼图: {sheet_path or '（无击球帧，未生成）'}")

    quality = video_quality(pose)
    print(f"\n=== 画质 ===\n体高 {quality['body_height_px']} px | "
          f"肩宽 {quality['shoulder_width_px']} px | "
          f"有效帧 {quality['usable_ratio']:.1%}")

    print("\n=== 指标摘要 ===")
    for k, v in metrics["summary"].items():
        # 只有 1 次击球时 std / iqr / cv 会是 None，不能直接格式化
        cv = "n/a" if v["cv"] is None else f"{v['cv']:.3f}"
        std = "n/a" if v["std"] is None else f"{v['std']:>7.3f}"
        print(f"{k:22} mean={v['mean']:>8}  std={std}  cv={cv}")

    return {"pose": str(pose_path), "metrics": str(metrics_path),
            "video": str(video_out), "keyframes": keyframes,
            "contact_sheet": sheet_path, "quality": quality,
            "events": metrics["events"], "summary": metrics["summary"],
            "fps": pose["meta"]["fps"], "total_frames": pose["meta"]["total_frames"]}


if __name__ == "__main__":
    video = sys.argv[1] if len(sys.argv) > 1 else "third_party/tennis-tracking/VideoInput/video_input3.mp4"
    out = sys.argv[2] if len(sys.argv) > 2 else None
    run(video, out)
