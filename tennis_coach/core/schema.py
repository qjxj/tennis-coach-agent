"""
工具统一返回结构。

为什么必须有它：
    如果工具只返回一段自然语言，编排器无法判断"这一步可不可信"，
    只能把文本原样塞给模型，既浪费上下文又无法做校验。

四个字段各解决一个问题：
    ok          → 要不要重试 / 换工具 / 降级
    data        → 结构化结果，可被代码校验
    evidence    → 结论回溯：哪段视频、哪一帧、哪张图
    confidence  → 触发复核（低置信走 crop/重分析）
"""

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ToolResult:
    ok: bool
    data: Dict[str, Any]
    evidence: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    latency_ms: int = 0
    error: Optional[str] = None
    unit: str = "normalized"

    def to_json(self) -> str:
        """工具返回值统一序列化成 JSON 字符串，便于模型阅读与代码解析。"""
        return json.dumps(self.__dict__, ensure_ascii=False)


def build_ok(data: Dict[str, Any],
             evidence: Optional[Dict[str, Any]] = None,
             confidence: float = 1.0,
             unit: str = "normalized",
             started: float = 0.0) -> ToolResult:
    return ToolResult(
        ok=True,
        data=data,
        evidence=evidence or {},
        confidence=round(float(confidence), 3),
        latency_ms=int((time.time() - started) * 1000) if started else 0,
        unit=unit,
    )


def build_err(message: str, started: float = 0.0,
              evidence: Optional[Dict[str, Any]] = None) -> ToolResult:
    return ToolResult(
        ok=False,
        data={},
        evidence=evidence or {},
        confidence=0.0,
        latency_ms=int((time.time() - started) * 1000) if started else 0,
        error=message,
    )
