"""
阶段 1：把姿态分析管线包成一个 agent 可调用的工具。

三个设计要点：
1. 参数用 Annotated 写说明，并用 Literal 锁死可选值
2. docstring 写清「适用 + 不适用」，让模型自己判断该不该调
3. 返回结构化 ToolResult，而不是一段自然语言
4. 工具内部吞掉异常，返回错误结果（工具失败不该打崩整条链路）

只回传「摘要 + 事件 + 证据路径」，不回传逐帧关键点：
93 帧 × 33 点 ≈ 9000 个数字，塞进上下文会直接爆掉。
"""

import sys
import time
from pathlib import Path
from typing import Annotated, Literal

from langchain_core.tools import tool

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from tennis_coach.core.schema import build_err, build_ok          # noqa: E402
from tennis_coach.core.store import record_analysis               # noqa: E402
from tennis_coach.run_minimal import run as run_pipeline          # noqa: E402

MODEL_FILES = {
    "lite": "third_party/models/pose_landmarker_lite.task",
    "full": "third_party/models/pose_landmarker_full.task",
}

# 体高像素下限：低于它，关键点抖动会被放大成假峰，指标不可信
MIN_BODY_HEIGHT_PX = 150


def _confidence(pose_ratio: float, event_count: int) -> float:
    """置信度 = 可用帧比例 × 事件充分性。

    可用帧比例低说明遮挡/出画；事件太少说明没拍到完整挥拍。
    两者任一不达标，confidence 会明显下降，供编排器触发复核或拒答。
    """
    if event_count == 0:
        return 0.05
    event_factor = min(1.0, event_count / 3.0)
    return max(0.0, min(1.0, pose_ratio * event_factor))


@tool
def analyze_stroke(
    video_path: Annotated[str, "击球视频的相对路径（相对项目根目录），如 data/uploads/forehand.mp4"],
    wrist: Annotated[Literal["auto", "left", "right"], "主手：auto 按可见度自动判断，或强制指定"] = "auto",
    model_size: Annotated[Literal["lite", "full"], "姿态模型：lite 快但精度低，full 慢但更准"] = "lite",
) -> str:
    """分析一段网球击球视频，返回击球次数与 5 项动作指标。

    指标均为「体高归一化」的无量纲值，与拍摄距离和分辨率无关：
      contact_offset        击球点相对髋部的前后偏移（正=身体前方）
      backswing_height      引拍最高点相对肩线的高度
      followthrough_length  击球后 0.4s 内手腕轨迹长度
      elbow_angle_deg       击球瞬间肘关节角度
      shoulder_angle_deg    肩线角度（躯干旋转的 2D 代理）

    适用：单人、固定机位、侧面拍摄、全身入镜的击球视频。
    不适用：多人同框、镜头切换的转播视频、慢动作教学剪辑、严重遮挡、
            只有上半身的画面、主体过小（体高不足画面 1/3）的远景。
    """
    started = time.time()

    if Path(video_path).is_absolute():
        return build_err("只接受相对路径，避免越权访问工作目录之外的文件", started).to_json()

    video = (PROJECT_ROOT / video_path).resolve()
    if not video.is_relative_to(PROJECT_ROOT):
        return build_err(f"路径越界：{video_path}", started).to_json()
    if not video.exists():
        return build_err(f"视频不存在：{video_path}", started).to_json()

    model_path = MODEL_FILES.get(model_size, MODEL_FILES["lite"])

    try:
        out = run_pipeline(str(video), model=model_path,
                           wrist=None if wrist == "auto" else wrist)
    except Exception as e:
        # 工具失败 → 返回错误结果，让模型看到原因后自行决定下一步
        return build_err(f"分析失败：{type(e).__name__}: {e}", started).to_json()

    events = out["events"]
    summary = out["summary"]
    quality = out.get("quality") or {}
    usable_ratio = float(quality.get("usable_ratio") or 0.0)
    body_px = float(quality.get("body_height_px") or 0.0)

    evidence = {
        "metrics_path": out["metrics"],
        "annotated_video": out["video"],
        "keyframes": out["keyframes"],
        "contact_sheet": out.get("contact_sheet", ""),
    }

    # 失败要显式：几乎检不到人体时，不能伪装成"没击球"，那是管线问题而非视频问题
    if usable_ratio < 0.10:
        return build_err(
            f"姿态检测几乎全部失败（有效帧比例 {usable_ratio:.2f}），"
            f"疑似模型文件不兼容或视频内容异常；请改用 model_size='lite' 或检查视频",
            started, evidence,
        ).to_json()

    # 主体太小：单帧关键点抖动 1~2px 就会被放大成"击球峰"，
    # 表现为次数忽多忽少。此时应当拒答，而不是把噪声当动作。
    if body_px and body_px < MIN_BODY_HEIGHT_PX:
        return build_err(
            f"画面里主体太小（体高仅 {body_px:.0f}px，至少需要 {MIN_BODY_HEIGHT_PX}px）。"
            f"这个尺度下关键点抖动会被放大成假击球峰，击球次数和动作指标都不可信。"
            f"请靠近拍摄：让人物高度至少占画面 1/3，侧面全身入镜",
            started, evidence,
        ).to_json()

    # 跨会话记忆：落一条记录，供 get_progress 做纵向对比。
    # 写库失败不能影响本次分析结果，因此单独吞掉异常。
    try:
        record_analysis(video=video_path, event_count=len(events),
                        summary=summary, quality=quality)
    except Exception:
        pass

    return build_ok(
        data={
            "event_count": len(events),
            "events": events,
            "summary": summary,
            "video_meta": {
                "fps": out.get("fps"),
                "usable_frame_ratio": round(usable_ratio, 3),
                "body_height_px": body_px,
                "shoulder_width_px": quality.get("shoulder_width_px"),
            },
        },
        evidence=evidence,
        confidence=_confidence(usable_ratio, len(events)),
        unit="body_ratio",
        started=started,
    ).to_json()


if __name__ == "__main__":
    import json

    sample = "third_party/tennis-tracking/VideoInput/video_input3.mp4"
    raw = analyze_stroke.invoke({"video_path": sample})
    result = json.loads(raw)
    print("ok:", result["ok"], "| confidence:", result["confidence"],
          "| latency:", result["latency_ms"], "ms")
    print("events:", result["data"]["event_count"])
    print("summary keys:", list(result["data"]["summary"].keys()))
    print("evidence:", list(result["evidence"].keys()))
