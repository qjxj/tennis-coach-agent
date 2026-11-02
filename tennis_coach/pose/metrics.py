"""
动作指标：从关键点序列里切出击球事件，并计算 5 个可解释指标。

设计原则：
1. 所有距离都除以"体高"做归一化 —— 这样与拍摄距离、分辨率无关，
   且侧面拍摄时不会像肩宽那样塌缩到 0
2. 事件检测用"腕速峰值"，而不是固定帧窗口 —— 应对不同节奏
3. 每个指标都写清定义，方便后续与教练评分做对比

坐标约定：
- 归一化坐标 (0~1)，x 向右、y 向下
- "前方"默认是画面 +x 方向；若球员面朝左侧，接触点偏移的符号会相反
- 因此文档里标注：建议侧面拍摄、球员大致面朝画面右侧
"""

from typing import Dict, List, Optional

import numpy as np
from scipy.signal import find_peaks

# 事件检测参数
# 思路：先估一个"站定不动时的腕速基线"，只有明显高于基线的峰才算挥拍。
# 用中位/分位数而不是全局最大值，避免被个别异常帧带偏。
PEAK_BASE_PERCENTILE = 25      # 静默基线 = 正速度的 25 分位
PEAK_HEIGHT_RATIO = 2.5        # 峰值至少是基线的 2.5 倍
PEAK_PROMINENCE_RATIO = 1.5    # 峰的显著度至少是基线的 1.5 倍（区分"起伏"和"真峰"）
MIN_GAP_SEC = 0.90             # 两次击球最小间隔：一次完整挥拍约 0.5~0.7s
SMOOTH_WINDOW = 5              # 腕速平滑窗口（帧）：3 帧太短，一个挥拍会被切出多个峰
CONTACT_ELBOW_MIN = 45.0       # 击球瞬间肘角下限：低于此值说明手臂几乎折叠，不是击球
CONTACT_ELBOW_MAX = 180.0
SHOULDER_MIN_RATIO = 0.20      # 肩线宽度 / 体高 低于此值：侧面视角，肩角不可用


def _wrist_name(pose: Dict, side: Optional[str]) -> str:
    """选择主手：显式指定 side，否则取平均可见度更高的那一侧。"""
    if side in ("left", "right"):
        return f"{side}_wrist"

    vis = {"left": 0.0, "right": 0.0}
    for f in pose["frames"]:
        if not f["ok"]:
            continue
        for k in vis:
            vis[k] += f["landmarks"][f"{k}_wrist"][2]
    return "left_wrist" if vis["left"] >= vis["right"] else "right_wrist"


def _angle(a, b, c) -> float:
    """b 点处的夹角（度）。a-b-c 三点，坐标均为 [x, y]。"""
    a, b, c = map(np.asarray, (a, b, c))
    ba, bc = a - b, c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-8
    cos = float(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def _moving_average(values: List[float], window: int) -> np.ndarray:
    if len(values) < window or window <= 1:
        return np.asarray(values, dtype=float)
    kernel = np.ones(window) / window
    return np.convolve(np.asarray(values, dtype=float), kernel, mode="same")


def detect_events(pose: Dict, wrist: str) -> List[dict]:
    """用腕速峰值找击球点。返回 [{frame, t, speed}]。"""
    frames = pose["frames"]
    fps = pose["meta"]["fps"]

    speeds: List[float] = []
    prev = None
    for f in frames:
        if not f["ok"] or f["scale"] <= 1e-6:
            speeds.append(0.0)
            prev = None
            continue
        w = f["landmarks"][wrist]
        if prev is None:
            speeds.append(0.0)
            prev = (w[0], w[1], f["t"])
            continue
        dist = float(np.hypot(w[0] - prev[0], w[1] - prev[1])) / f["scale"]
        dt = max(f["t"] - prev[2], 1e-3)
        speeds.append(dist / dt)
        prev = (w[0], w[1], f["t"])

    smooth = _moving_average(speeds, SMOOTH_WINDOW)

    positive = smooth[smooth > 0]
    if len(positive) == 0:
        return []
    base = float(np.percentile(positive, PEAK_BASE_PERCENTILE))
    if base <= 1e-6:
        return []

    peaks, _ = find_peaks(
        smooth,
        height=max(float(np.percentile(positive, 75)), base * PEAK_HEIGHT_RATIO),
        prominence=base * PEAK_PROMINENCE_RATIO,
        distance=max(1, int(MIN_GAP_SEC * fps)),
    )
    return [{"frame": int(i), "t": frames[int(i)]["t"], "speed": float(smooth[i])}
            for i in peaks]


def compute_metrics(pose: Dict, wrist: Optional[str] = None) -> dict:
    """切事件 + 算指标，返回 {events: [...], summary: {...}, definitions: {...}}。"""
    frames = pose["frames"]
    fps = pose["meta"]["fps"]
    wrist = _wrist_name(pose, wrist)
    events = detect_events(pose, wrist)

    back_swing_frames = int(0.8 * fps)
    follow_frames = int(0.4 * fps)

    out_events = []
    for ev in events:
        i = ev["frame"]
        f = frames[i]
        if not f["ok"]:
            continue
        scale = f["scale"] or 1.0
        hip = f["hip_center"]
        w = f["landmarks"][wrist]

        # 1) 击球点相对身体的前后偏移（正=身体前方）
        contact_offset = (w[0] - hip[0]) / scale

        # 2) 引拍高度：窗口内手腕最高点（y 最小）相对肩线
        lo = max(0, i - back_swing_frames)
        window = [fr for fr in frames[lo:i + 1] if fr["ok"]]
        if window:
            top_y = min(fr["landmarks"][wrist][1] for fr in window)
            shoulder_y = (f["landmarks"]["left_shoulder"][1] +
                          f["landmarks"]["right_shoulder"][1]) / 2
            backswing_height = (shoulder_y - top_y) / scale
        else:
            backswing_height = 0.0

        # 3) 随挥长度：击球后手腕轨迹累计长度
        hi = min(len(frames), i + follow_frames + 1)
        path = 0.0
        for k in range(i, hi - 1):
            a, b = frames[k], frames[k + 1]
            if not (a["ok"] and b["ok"]) or a["scale"] <= 1e-6:
                continue
            pa, pb = a["landmarks"][wrist], b["landmarks"][wrist]
            path += float(np.hypot(pb[0] - pa[0], pb[1] - pa[1])) / a["scale"]
        followthrough_length = path

        # 4) 击球瞬间肘角
        side = wrist.split("_")[0]
        elbow_angle = _angle(
            f["landmarks"][f"{side}_shoulder"][:2],
            f["landmarks"][f"{side}_elbow"][:2],
            w[:2],
        )

        # 5) 肩线倾角（躯干旋转的 2D 代理指标）
        # 侧面视角下两肩在 2D 上几乎重合，atan2 的结果完全由关键点噪声决定，
        # 因此只有当肩线在画面里足够展开时才输出，否则给 None。
        # 倾角取相对水平的绝对值：左右肩谁在前取决于机位在哪一侧，
        # 符号没有物理意义（否则会出现 ±180° 的假跳变）。
        ls, rs = f["landmarks"]["left_shoulder"], f["landmarks"]["right_shoulder"]
        shoulder_w_2d = float(np.hypot(ls[0] - rs[0], ls[1] - rs[1]))
        if shoulder_w_2d >= SHOULDER_MIN_RATIO * scale:
            angle = abs(float(np.degrees(np.arctan2(rs[1] - ls[1], rs[0] - ls[0]))))
            shoulder_angle = round(min(angle, 180.0 - angle), 1)
        else:
            shoulder_angle = None

        out_events.append({
            "frame": i,
            "t": round(ev["t"], 3),
            "wrist_speed": round(ev["speed"], 3),
            "contact_offset": round(contact_offset, 3),
            "backswing_height": round(backswing_height, 3),
            "followthrough_length": round(followthrough_length, 3),
            "elbow_angle_deg": round(elbow_angle, 1),
            "shoulder_angle_deg": shoulder_angle,
        })

    # 姿势合理性过滤：击球瞬间肘部不可能完全折叠，
    # 这类峰值多半是引拍或非击球的手臂动作，直接丢弃。
    plausible, rejected = [], 0
    for e in out_events:
        if CONTACT_ELBOW_MIN <= e["elbow_angle_deg"] <= CONTACT_ELBOW_MAX:
            plausible.append(e)
        else:
            rejected += 1
    out_events = plausible

    summary = {}
    for key in ("wrist_speed", "contact_offset", "backswing_height",
                "followthrough_length", "elbow_angle_deg", "shoulder_angle_deg"):
        vals = np.array([e[key] for e in out_events if e.get(key) is not None], dtype=float)
        if len(vals) == 0:
            continue
        single = len(vals) < 2
        mean = float(vals.mean())
        std = float(vals.std())
        q75, q25 = np.percentile(vals, [75, 25])
        summary[key] = {
            "mean": round(mean, 3),
            # 只有 1 次击球时，离散度没有意义，必须返回 None 而不是 0
            "std": None if single else round(std, 3),
            "iqr": None if single else round(float(q75 - q25), 3),
            # CV 在均值接近 0 时会爆炸（曾出现 0.97 的假"不稳定"），
            # 因此仅在 |mean| 足够大时才报告。
            "cv": (None if single or abs(mean) < 0.3
                   else round(std / abs(mean), 3)),
        }

    return {
        "meta": {**pose["meta"], "wrist": wrist, "events": len(out_events)},
        "events": out_events,
        "summary": summary,
        "definitions": {
            "contact_offset": "击球点相对髋部中心的前后偏移 / 体高；正=身体前方。侧面拍摄、面朝画面右侧时有效",
            "backswing_height": "引拍最高点相对肩线的高度 / 体高；正=高于肩",
            "followthrough_length": "击球后 0.4s 内手腕轨迹长度 / 体高",
            "elbow_angle_deg": "击球瞬间肩-肘-腕夹角（度）",
            "shoulder_angle_deg": "击球瞬间肩线相对水平线角度（度），躯干旋转的 2D 代理",
            "cv": "变异系数 std/|mean|，衡量多次击球的稳定性",
        },
    }


if __name__ == "__main__":
    import json
    import sys

    src = sys.argv[1] if len(sys.argv) > 1 else "data/pose/sample_pose.json"
    pose = json.loads(open(src, encoding="utf-8").read())
    result = compute_metrics(pose)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print("events:", [e["t"] for e in result["events"]])
