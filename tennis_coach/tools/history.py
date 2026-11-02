"""
阶段 8：跨会话记忆工具。

为什么它必须是"工具"，而不是自动塞进上下文：
    历史记录可能有几十条，每轮都塞进去只会污染上下文；
    让模型自己判断"这个问题需不需要历史"，需要时才取。
    这是上下文工程里"按需检索"和"全量注入"的分界线。

它读取的是 store.record_analysis 在每次 analyze_stroke 之后落库的数据，
因此不需要用户做任何额外操作。
"""

import sys
import time
from pathlib import Path
from typing import Annotated

from langchain_core.tools import tool

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from tennis_coach.core.schema import build_err, build_ok     # noqa: E402
from tennis_coach.core.store import DB_PATH, build_progress  # noqa: E402


@tool
def get_progress(
    days: Annotated[int, "回看最近多少天，1~365，默认 30"] = 30,
) -> str:
    """查询用户的历史击球记录，做纵向对比（首次 vs 最近一次）。

    每个指标返回：
      mean         [首次均值, 最近均值]
      mean_delta   均值变化量（正负不代表好坏，要结合指标含义解释）
      cv           [首次变异系数, 最近变异系数]，越小越稳定
      stability    improved / worsened / stable —— 按 cv 变化判定
    另有 sessions（总记录数）、usable_sessions（有 ≥2 次击球、可算稳定性的记录数）、
    history（最近 20 次的日期、击球数、主体尺寸）。

    若 setup_mismatch=true，说明历史记录的拍摄尺度差异大（机位/距离不同），
    此时趋势不可直接比较，必须向用户说明。
    `note` 字段是给用户看的解释，应原样转述其中要点。

    适用：用户问"我有没有进步""比上次怎么样""最近状态如何"。
    不适用：单次动作细节（那属于 analyze_stroke）。
    insufficient=true 表示可对比的记录不足 2 次，必须如实说明，禁止编造历史成绩。
    """
    started = time.time()
    days = max(1, min(int(days), 365))

    try:
        data = build_progress(days)
    except Exception as e:
        return build_err(f"读取历史记录失败：{type(e).__name__}: {e}", started).to_json()

    evidence = {"db_path": str(DB_PATH)}

    if data.get("insufficient"):
        data["note"] = ("可对比的历史记录不足 2 次（每次至少要有 2 次击球才算数），"
                        "现在还看不出趋势。请先上传击球视频做分析，之后再看进步。")
        return build_ok(data=data, evidence=evidence, confidence=0.2,
                        started=started).to_json()

    # 样本越少，趋势越不可信
    confidence = min(0.95, 0.3 + 0.1 * data["sessions"])
    return build_ok(data=data, evidence=evidence, confidence=confidence,
                    started=started).to_json()


if __name__ == "__main__":
    import json

    raw = get_progress.invoke({"days": 30})
    obj = json.loads(raw)
    print("ok:", obj["ok"], "| confidence:", obj["confidence"])
    print(json.dumps(obj["data"], ensure_ascii=False, indent=2)[:1200])
