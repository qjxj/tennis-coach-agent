"""
姿态提取：视频 → 逐帧关键点（MediaPipe Tasks API）。

为什么用 Tasks API：mediapipe 1.x 已移除旧的 mp.solutions.pose，
统一改为 PoseLandmarker + .task 模型文件。

输出结构（每帧）：
{
  "frame": 12,
  "t": 0.4,
  "landmarks": {"left_wrist": [x, y, v], ...},   # 归一化坐标 0~1 + 可见度
  "scale": 0.183,        # 体高（归一化），用于把像素/归一化距离换算成"身体比例"
  "shoulder_width": 0.09,  # 肩宽（归一化），仅用于判断视角与画面尺度
  "hip_center": [0.52, 0.61],
  "ok": true             # 该帧是否有可用姿态
}
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

# MediaPipe 33 个关键点里我们真正要用的
KEY_JOINTS = {
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    23: "left_hip",
    24: "right_hip",
    25: "left_knee",
    26: "right_knee",
    27: "left_ankle",
    28: "right_ankle",
}


@dataclass
class PoseExtractConfig:
    model_path: str = "third_party/models/pose_landmarker_lite.task"
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    frame_stride: int = 1          # 抽帧步长，1 = 每帧都处理


def _to_dict(landmarks) -> Dict[str, List[float]]:
    """把 MediaPipe 结果转成 {关节名: [x, y, visibility]}，坐标归一化到 0~1。"""
    out: Dict[str, List[float]] = {}
    for idx, name in KEY_JOINTS.items():
        lm = landmarks[idx]
        out[name] = [float(lm.x), float(lm.y), float(getattr(lm, "visibility", 1.0))]
    return out


def _shoulder_width(pts: Dict[str, List[float]]) -> float:
    ls, rs = pts["left_shoulder"], pts["right_shoulder"]
    return float(np.hypot(ls[0] - rs[0], ls[1] - rs[1]))


def _body_scale(pts: Dict[str, List[float]]) -> float:
    """归一化尺度：体高（肩中点 → 较低脚踝）。

    为什么不用肩宽：侧面拍摄时两肩在 2D 上几乎重合，肩宽会塌缩到接近 0
    （实测同一段视频里 0.054 → 0.0009，变异 0.57），所有除以它的量
    ——腕速、击球点偏移——都会爆表，出现"腕速 194""contact_offset +21"
    这种物理上不存在的数值，进而凭空造出假击球峰。
    体高在同一段视频里变异只有 0.05，是稳定尺度。

    脚踝不可见时退化为 1.9×躯干长（成人体高约等于躯干长的 1.9 倍）。
    """
    sx = (pts["left_shoulder"][0] + pts["right_shoulder"][0]) / 2
    sy = (pts["left_shoulder"][1] + pts["right_shoulder"][1]) / 2
    hx = (pts["left_hip"][0] + pts["right_hip"][0]) / 2
    hy = (pts["left_hip"][1] + pts["right_hip"][1]) / 2
    torso = float(np.hypot(sx - hx, sy - hy))
    ankle_y = max(pts["left_ankle"][1], pts["right_ankle"][1])
    body = float(ankle_y - sy)
    return body if body > 1.5 * torso else 1.9 * torso


def _hip_center(pts: Dict[str, List[float]]) -> List[float]:
    lh, rh = pts["left_hip"], pts["right_hip"]
    return [(lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2]


def extract_pose(video_path: str, cfg: PoseExtractConfig = None) -> dict:
    """逐帧提取姿态，返回 {meta: {...}, frames: [...]}。"""
    cfg = cfg or PoseExtractConfig()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=cfg.model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,                                  # 只分析画面里最主要的那个人
        min_pose_detection_confidence=cfg.min_detection_confidence,
        min_tracking_confidence=cfg.min_tracking_confidence,
    )

    frames: List[dict] = []
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % cfg.frame_stride == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms = int(idx / fps * 1000)
                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                if result.pose_landmarks:
                    pts = _to_dict(result.pose_landmarks[0])
                    scale = _body_scale(pts)
                    # 尺度接近 0 说明姿态基本没检到，后续所有归一化都不可信
                    ok = scale > 0.02
                    frames.append({
                        "frame": idx,
                        "t": round(idx / fps, 4),
                        "landmarks": pts,
                        "scale": scale if ok else 0.0,
                        "shoulder_width": _shoulder_width(pts),
                        "hip_center": _hip_center(pts),
                        "ok": ok,
                    })
                else:
                    frames.append({"frame": idx, "t": round(idx / fps, 4),
                                   "landmarks": {}, "scale": 0.0,
                                   "shoulder_width": 0.0,
                                   "hip_center": [0.5, 0.5], "ok": False})
            idx += 1

    cap.release()

    return {
        "meta": {
            "video": str(video_path),
            "fps": fps,
            "width": width,
            "height": height,
            "total_frames": total,
            "processed_frames": len(frames),
            "model": Path(cfg.model_path).name,
        },
        "frames": frames,
    }


def save_pose(pose: dict, out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(pose, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    import sys

    src = sys.argv[1] if len(sys.argv) > 1 else "third_party/tennis-tracking/VideoInput/video_input3.mp4"
    out = sys.argv[2] if len(sys.argv) > 2 else "data/pose/sample_pose.json"
    result = extract_pose(src)
    save_pose(result, out)
    ok = sum(1 for f in result["frames"] if f["ok"])
    print(f"frames={len(result['frames'])} ok={ok} -> {out}")
