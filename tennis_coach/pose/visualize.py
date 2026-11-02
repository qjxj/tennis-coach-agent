"""
可视化：骨架叠加 + 击球事件标注 + 关键帧导出。

用法（通常由 run_minimal.py 调用）：
    draw_video(video_path, pose, metrics, out_video, out_dir)
"""

from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

# 只画我们关心的关节连线
CONNECTIONS = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]

OK_COLOR = (80, 220, 120)      # 骨架
EVENT_COLOR = (60, 160, 255)   # 击球帧


def _draw_skeleton(frame, landmarks: Dict[str, List[float]], color) -> None:
    h, w = frame.shape[:2]
    for a, b in CONNECTIONS:
        if a not in landmarks or b not in landmarks:
            continue
        pa = (int(landmarks[a][0] * w), int(landmarks[a][1] * h))
        pb = (int(landmarks[b][0] * w), int(landmarks[b][1] * h))
        cv2.line(frame, pa, pb, color, 3)
    for name in ("left_wrist", "right_wrist", "left_hip", "right_hip"):
        if name in landmarks:
            p = (int(landmarks[name][0] * w), int(landmarks[name][1] * h))
            cv2.circle(frame, p, 6, color, -1)


def draw_video(video_path: str,
               pose: dict,
               metrics: dict,
               out_video: str,
               out_dir: str,
               score_threshold: float = 0.6) -> List[str]:
    """生成标注视频，并把每个击球事件的关键帧存成 jpg。"""
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    Path(out_video).parent.mkdir(parents=True, exist_ok=True)

    events = {e["frame"]: e for e in metrics.get("events", [])}
    pose_by_frame = {f["frame"]: f for f in pose["frames"]}

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    keyframes: List[str] = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        pf = pose_by_frame.get(idx)
        if pf and pf["ok"] and pf["scale"] > 1e-6:
            ev = events.get(idx)
            color = EVENT_COLOR if ev else OK_COLOR
            _draw_skeleton(frame, pf["landmarks"], color)

            if ev:
                cv2.putText(frame, f"IMPACT #{list(events).index(idx) + 1}",
                            (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)
                lines = [
                    f"contact offset {ev['contact_offset']:+.2f}",
                    f"backswing h {ev['backswing_height']:+.2f}",
                    f"followthrough {ev['followthrough_length']:.2f}",
                    f"elbow {ev['elbow_angle_deg']:.0f} deg",
                ]
                for j, text in enumerate(lines):
                    cv2.putText(frame, text, (20, 90 + j * 34),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                kf = out_dir_p / f"impact_{idx:04d}.jpg"
                cv2.imwrite(str(kf), frame)
                keyframes.append(str(kf))

        writer.write(frame)
        idx += 1

    cap.release()
    writer.release()
    return keyframes


def make_contact_sheet(keyframes: List[str], out_path: str,
                       cols: int = 4, thumb_w: int = 400) -> str:
    """把每拍关键帧拼成一张图，方便人工核对"这些是不是真的击球"。

    标注/调参场景用：一眼看出哪几帧是误检。
    """
    imgs = []
    for p in keyframes:
        img = cv2.imread(str(p))
        if img is None:
            continue
        scale = thumb_w / img.shape[1]
        img = cv2.resize(img, (thumb_w, int(img.shape[0] * scale)))
        label = Path(p).stem
        cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        imgs.append(img)

    if not imgs:
        return ""

    rows = []
    for i in range(0, len(imgs), cols):
        row = imgs[i:i + cols]
        while len(row) < cols:                       # 最后一行补齐，避免 vconcat 维度不一致
            row.append(np.zeros_like(imgs[0]))
        rows.append(cv2.hconcat(row))

    sheet = cv2.vconcat(rows)
    cv2.imwrite(str(out_path), sheet)
    return str(out_path)
