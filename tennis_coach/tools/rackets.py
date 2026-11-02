"""球拍数据 + 硬约束过滤工具（阶段 4 的第二个子智能体用）。"""

import time
from typing import Annotated, Optional

from langchain_core.tools import tool

from tennis_coach.core.schema import build_ok, build_err

# 简化示例数据：真实项目应 20~50 款，字段以下单实测为准
RACKETS = [
    {"name": "轻量入门 A", "weight_g": 270, "head_in2": 105, "balance": "head_light",
     "stiffness_ra": 68, "price_cny": 499, "level": "beginner", "arm_friendly": False},
    {"name": "舒适控制 B", "weight_g": 295, "head_in2": 100, "balance": "even",
     "stiffness_ra": 60, "price_cny": 899, "level": "beginner", "arm_friendly": True},
    {"name": "均衡进阶 C", "weight_g": 300, "head_in2": 100, "balance": "even",
     "stiffness_ra": 64, "price_cny": 1299, "level": "intermediate", "arm_friendly": False},
    {"name": "旋转友好 D", "weight_g": 305, "head_in2": 100, "balance": "head_light",
     "stiffness_ra": 66, "price_cny": 1499, "level": "intermediate", "arm_friendly": False},
    {"name": "软拍护肘 E", "weight_g": 290, "head_in2": 98, "balance": "even",
     "stiffness_ra": 56, "price_cny": 1199, "level": "intermediate", "arm_friendly": True},
    {"name": "重拍力量 F", "weight_g": 320, "head_in2": 98, "balance": "head_heavy",
     "stiffness_ra": 67, "price_cny": 1699, "level": "advanced", "arm_friendly": False},
    {"name": "职业控制 G", "weight_g": 315, "head_in2": 97, "balance": "head_light",
     "stiffness_ra": 62, "price_cny": 1899, "level": "advanced", "arm_friendly": True},
    {"name": "高性价比 H", "weight_g": 285, "head_in2": 102, "balance": "head_light",
     "stiffness_ra": 69, "price_cny": 699, "level": "beginner", "arm_friendly": False},
]


@tool
def filter_rackets(
    max_price_cny: Annotated[int, "预算上限，单位人民币"],
    level: Annotated[str, "球员水平：beginner / intermediate / advanced"],
    max_weight_g: Annotated[Optional[int], "球拍重量上限（克）；不填表示不限"] = None,
    arm_issue: Annotated[bool, "是否有肘部不适；True 时只保留护肘友好的球拍"] = False,
    prefer: Annotated[str, "偏好：control 控制 / power 力量 / spin 旋转"] = "control",
) -> str:
    """按硬约束过滤球拍并按偏好打分。

    硬约束：预算、水平匹配、重量上限、（若有肘部问题）护肘友好。
    返回 Top3 与每项的分数拆解，便于解释推荐理由。
    """
    started = time.time()

    if level not in {"beginner", "intermediate", "advanced"}:
        return build_err(f"非法 level：{level}", started).to_json()
    if prefer not in {"control", "power", "spin"}:
        return build_err(f"非法 prefer：{prefer}", started).to_json()

    # 1) 硬约束过滤
    passed = [
        r for r in RACKETS
        if r["price_cny"] <= max_price_cny
        and r["level"] == level
        and (max_weight_g is None or r["weight_g"] <= max_weight_g)
        and (not arm_issue or r["arm_friendly"])
    ]

    if not passed:
        return build_ok(
            data={"matched": 0, "top": []},
            evidence={"rule": "hard_constraints", "checked": len(RACKETS)},
            confidence=1.0, started=started,
        ).to_json()

    # 2) 软偏好打分（0~1，权重可解释）
    def score(r):
        s_control = 1 - (r["stiffness_ra"] - 55) / 20
        s_power = (r["weight_g"] - 270) / 60
        s_spin = (r["head_in2"] - 95) / 15
        table = {"control": s_control, "power": s_power, "spin": s_spin}
        return {
            "total": round(max(0.0, min(1.0, table[prefer])), 3),
            "control": round(max(0.0, min(1.0, s_control)), 3),
            "power": round(max(0.0, min(1.0, s_power)), 3),
            "spin": round(max(0.0, min(1.0, s_spin)), 3),
        }

    ranked = sorted(
        ({"name": r["name"], "price_cny": r["price_cny"], "weight_g": r["weight_g"],
          "stiffness_ra": r["stiffness_ra"], "scores": score(r)} for r in passed),
        key=lambda x: x["scores"]["total"], reverse=True,
    )[:3]

    return build_ok(
        data={"matched": len(passed), "top": ranked},
        evidence={"filters": {"max_price_cny": max_price_cny, "level": level,
                              "max_weight_g": max_weight_g, "arm_issue": arm_issue},
                  "prefer": prefer},
        confidence=1.0, started=started,
    ).to_json()
