"""
阶段 3：上下文工程。

两个手法：
1. 工具返回压缩 —— 只把"摘要 + 异常 + 证据路径"给模型，逐次击球的原始数值留在磁盘
2. 历史压缩   —— 较早的工具结果折叠成一行摘要，避免多轮对话线性膨胀
"""

import json
from typing import Any, Dict, List


def estimate_tokens(text: str) -> int:
    """粗略估算 token：中文约 1.5 字/token。用于前后对比，不做精确计费。"""
    return int(len(text) / 1.5)


def _find_anomalies(events: List[Dict[str, Any]], summary: Dict[str, Any],
                    limit: int = 3) -> List[Dict[str, Any]]:
    """找出偏离均值超过 2 倍标准差的击球，只回传这几条。"""
    anomalies = []
    for ev in events:
        reasons = []
        for metric, stat in summary.items():
            mean, std = stat.get("mean"), stat.get("std") or 0
            value = ev.get(metric)
            if value is None or not std:
                continue
            if abs(value - mean) > 2 * std:
                reasons.append(f"{metric}={value:.3g}（均值{mean:.3g}±{std:.3g}）")
        if reasons:
            anomalies.append({"frame": ev.get("frame"), "reasons": reasons})
    return anomalies[:limit]


def compact_tool_result(raw: str, keep_keyframes: int = 3) -> str:
    """把工具的完整返回压成模型真正需要的部分。

    保留：ok / confidence / unit / event_count / summary / anomalies / 证据路径
    丢弃：逐次击球的全部原始指标（已在 summary 与 anomalies 中体现）
    """
    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw

    data = obj.get("data") or {}
    if "summary" not in data:
        return raw

    evidence = obj.get("evidence") or {}
    compact = {
        "ok": obj.get("ok"),
        "confidence": obj.get("confidence"),
        "unit": obj.get("unit"),
        "event_count": data.get("event_count"),
        "summary": data.get("summary"),
        "anomalies": _find_anomalies(data.get("events") or [], data.get("summary") or {}),
        "usable_frame_ratio": (data.get("video_meta") or {}).get("usable_frame_ratio"),
        "evidence": {
            "keyframes": (evidence.get("keyframes") or [])[:keep_keyframes],
            "contact_sheet": evidence.get("contact_sheet"),
            "metrics_path": evidence.get("metrics_path"),
            "annotated_video": evidence.get("annotated_video"),
        },
    }
    return json.dumps(compact, ensure_ascii=False)


def prune_history(messages: list, keep_recent_tools: int = 1,
                  head: int = 120) -> list:
    """把较早的 ToolMessage 折叠成一行，保留最近 keep_recent_tools 条原文。"""
    tool_positions = [i for i, m in enumerate(messages)
                      if getattr(m, "type", "") == "tool"]
    stale = set(tool_positions[:-keep_recent_tools] if keep_recent_tools else tool_positions)

    pruned = []
    for i, m in enumerate(messages):
        if i in stale and len(str(m.content)) > head:
            m = m.model_copy(update={
                "content": str(m.content)[:head] + " ...[历史结果已压缩，完整数据见文件]"
            })
        pruned.append(m)
    return pruned
