"""
阶段 5：可靠性门禁（硬约束，不依赖 Prompt）。

原则：模型可以"想给建议"，但代码可以"不让它给"。

三个判定：
    refuse —— 数据不可用，直接拒答（确定性文案，不经模型）
    warn   —— 可用但需标注（交给模型，同时附带警示）
    ok     —— 正常
"""

import json
from typing import Any, Dict, List

from tennis_coach.core.context import _find_anomalies


# 阈值集中在一处，便于评测时调整
CONFIDENCE_REFUSE = 0.60     # 低于此值：拒答
USABLE_RATIO_REFUSE = 0.50   # 有效帧比例低于此值：拒答
MIN_EVENTS = 3               # 少于这么多次击球，无法评估"稳定性"
MIN_BODY_HEIGHT_PX = 150     # 主体体高低于此值：关键点噪声会被放大成假击球
ANOMALY_WARN = 2             # 异常击球达到此数量：警示


def _load(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(str(raw))
    except (json.JSONDecodeError, TypeError):
        return {}
    # 工具可能返回裸字符串（合法 JSON 但不是对象），同样视为不可用
    return parsed if isinstance(parsed, dict) else {}


def evaluate(raw_result: Any) -> Dict[str, Any]:
    """根据工具返回值给出可用性判定。"""
    obj = _load(raw_result)
    if not obj:
        return {"status": "refuse", "reasons": ["无法解析工具返回值"],
                "message": "分析结果格式异常，本次无法给出结论，请重试。"}

    if not obj.get("ok", False):
        return {"status": "refuse", "reasons": [obj.get("error") or "工具执行失败"],
                "message": f"分析未能完成：{obj.get('error') or '未知错误'}"}

    data = obj.get("data") or {}
    confidence = float(obj.get("confidence") or 0.0)
    meta = data.get("video_meta") or {}
    usable = meta.get("usable_frame_ratio")
    usable = float(usable) if usable is not None else None
    body_px = meta.get("body_height_px")
    body_px = float(body_px) if body_px is not None else None
    event_count = int(data.get("event_count") or 0)
    anomalies: List[Dict[str, Any]] = _find_anomalies(
        data.get("events") or [], data.get("summary") or {})

    too_small = body_px is not None and 0 < body_px < MIN_BODY_HEIGHT_PX

    reasons = []
    if too_small:
        # 放在第一条：主体太小是根因，其它指标都不可信
        reasons.append(f"画面主体太小（体高 {body_px:.0f}px，低于 {MIN_BODY_HEIGHT_PX}px），"
                       "关键点抖动会被放大成假击球")
    if event_count == 0:
        reasons.append("未检出任何击球事件")
    elif event_count < MIN_EVENTS:
        # 与"视频质量差"区分开：样本不足时就该说样本不足
        reasons.append(f"只识别到 {event_count} 次击球，"
                       f"少于评估稳定性所需的 {MIN_EVENTS} 次")
    if confidence < CONFIDENCE_REFUSE and event_count >= MIN_EVENTS:
        reasons.append(f"置信度 {confidence:.2f} 低于阈值 {CONFIDENCE_REFUSE}")
    if usable is not None and usable < USABLE_RATIO_REFUSE:
        reasons.append(f"有效帧比例 {usable:.2f} 低于阈值 {USABLE_RATIO_REFUSE}")

    if reasons:
        # 三种失败原因，给的建议不一样
        if too_small:
            advice = "请靠近拍摄：人物高度至少占画面 1/3，侧面全身入镜，再连续打 5~8 次。"
        elif event_count > 0 and event_count < MIN_EVENTS and usable is not None \
                and usable >= USABLE_RATIO_REFUSE:
            advice = f"建议连续多打几次再录（至少 {MIN_EVENTS} 次，最好 5~8 次），稳定性才有意义。"
        else:
            advice = "建议重拍：固定机位、侧面拍摄、全身入镜、光线充足、单一主体。"
        return {
            "status": "refuse",
            "reasons": reasons,
            "message": (
                "本次分析不通过质量门禁，暂不给出技术建议。\n"
                f"原因：{'；'.join(reasons)}\n" + advice
            ),
        }

    anomaly_count = len(anomalies)
    if anomaly_count >= ANOMALY_WARN:
        return {
            "status": "warn",
            "reasons": [f"{anomaly_count} 次击球的指标偏离均值超过 2σ"],
            "message": (f"数据可用，但检测到 {anomaly_count} 次异常击球，"
                        "以下结论仅针对稳定击球，异常样本请回看关键帧。"),
        }

    return {"status": "ok", "reasons": [], "message": ""}
