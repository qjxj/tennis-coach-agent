"""
阶段 9：训练计划工具。

为什么"短板判定"必须写在代码里：
    让模型自己看 summary 决定练什么，它会挑自己解释得顺手的、
    或者挑听起来好听的指标。代码按变异系数排名，模型只能在这个
    结论之上组织语言 —— 判断权归代码，表达权归模型。

为什么按 CV 排名而不是按绝对值：
    不同指标的绝对量纲不同（横向偏移是体高倍数，肘角是度），
    没法直接比大小；但"自己跟自己的历史比稳不稳"是可比的。
"""

import sys
import time
from pathlib import Path
from typing import Annotated, Any, Dict, List

from langchain_core.tools import tool

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from tennis_coach.core.schema import build_err, build_ok      # noqa: E402
from tennis_coach.core.store import fetch_analyses            # noqa: E402

# 指标 → 症状与训练项。这段映射是经验设定，未经教练标注校准。
DRILLS: Dict[str, Dict[str, Any]] = {
    "contact_offset": {
        "label": "击球点稳定性",
        "symptom": "击球点忽前忽后：有时被球挤到，有时探身够着打",
        "drills": [
            "标志盘标出理想击球区（身体前方约一臂），只在该区域击球，20 球 × 3 组",
            "慢速定点多球，每次击球后停 1 秒，确认拍面落在击球区内",
        ],
    },
    "backswing_height": {
        "label": "引拍高度",
        "symptom": "引拍忽高忽低，上旋和平击混着打，出球弧线不统一",
        "drills": [
            "对墙连续 30 次，引拍固定在肩线高度，球不落地",
            "影子挥拍 3 组 × 15 次，引拍到位后停顿 1 秒再向前",
        ],
    },
    "followthrough_length": {
        "label": "随挥完整度",
        "symptom": "随挥收得太早，力量没送出去，球质发虚",
        "drills": [
            "击球后强制收拍到对侧肩上，20 球 × 3 组",
            "七成力打深球，要求落点在底线前 2 米内",
        ],
    },
    "elbow_angle_deg": {
        "label": "击球时肘角",
        "symptom": "击球瞬间手臂要么锁死伸直，要么过度折叠",
        "drills": [
            "放松握拍，击球瞬间保持肘部微屈（约 130°），20 球 × 3 组",
            "腋下夹毛巾挥拍，毛巾不掉，练手臂与躯干的配合",
        ],
    },
    "shoulder_angle_deg": {
        "label": "转体幅度",
        "symptom": "转体不足，靠手臂发力；或转体过度，动作散",
        "drills": [
            "双手持拍转体练习，肩膀转到位再出手，15 次 × 3 组",
            "对墙练习，击球后胸口转向来球方向",
        ],
    },
    "wrist_speed": {
        "label": "挥拍速度节奏",
        "symptom": "挥速忽快忽慢，发力时机不一致",
        "drills": [
            "空挥 3 组 × 20 次，跟 60bpm 节拍器保持同一节奏",
            "多球快慢交替：一快一慢各 10 球，体会加速点",
        ],
    },
}

# 所有指标都低于这个 CV 时，认为已经足够稳定，改为维持性计划
CV_CONCERN = 0.15


def _rank(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    """按 CV 从大到小排名。CV 不可用（均值≈0 或样本不足）的指标跳过。"""
    ranked = []
    for key, meta in DRILLS.items():
        stat = summary.get(key) or {}
        cv = stat.get("cv")
        if cv is None:
            continue
        ranked.append({"metric": key, "label": meta["label"], "cv": cv})
    ranked.sort(key=lambda x: x["cv"], reverse=True)
    return ranked


@tool
def make_training_plan(
    weeks: Annotated[int, "计划周期（周），1~12，默认 4"] = 4,
) -> str:
    """根据历史分析记录，找出最不稳定的指标并生成分周训练计划。

    流程：读最近一次有效分析 → 按指标变异系数（CV）排名 → 取最不稳的 1~2 项
    → 映射到训练项 → 排成分周计划。

    返回字段：
      ranking   全部指标按 CV 从大到小，cv 越大越不稳
      focus     本次要练的 1~2 项，含 symptom（症状）与 drills（训练动作）
      schedule  分周安排，main 是当周主练项
      review    周期结束后的复测动作

    适用：用户问"我该怎么练""给个训练计划""先练什么"。
    不适用：用户还没有任何历史记录（此时会返回错误，需先分析一段视频）。
    注意：CV 是"相对自己波动"的稳定性指标，不是绝对水平；训练项映射为经验设定。
    """
    started = time.time()
    weeks = max(1, min(int(weeks), 12))

    try:
        rows = fetch_analyses(365)
    except Exception as e:
        return build_err(f"读取历史记录失败：{type(e).__name__}: {e}", started).to_json()

    # 只有 ≥2 次击球的记录才谈得上"稳定"
    usable = [r for r in rows if (r["event_count"] or 0) >= 2]
    if not usable:
        return build_err(
            "还没有可用于制定计划的分析记录（需要每次至少识别出 2 次击球）。"
            "请先上传一段击球视频做分析，再让我排计划",
            started,
        ).to_json()

    latest = usable[-1]
    ranking = _rank(latest["summary"])
    if not ranking:
        return build_err(
            "最近一次分析里所有指标都缺少有效的离散度数据（可能是击球次数太少），"
            "无法判断短板。建议重录一段至少 5 次连续击球的视频",
            started,
        ).to_json()

    weak = [r for r in ranking if r["cv"] >= CV_CONCERN][:2]
    focus = []
    for item in weak:
        meta = DRILLS[item["metric"]]
        focus.append({**item, "symptom": meta["symptom"], "drills": meta["drills"]})

    if focus:
        schedule = []
        for w in range(1, weeks + 1):
            main = focus[(w - 1) % len(focus)]
            schedule.append({"week": w, "main": main["label"],
                             "metric": main["metric"], "sessions": 3})
        plan_note = "按短板优先级轮换主练项，每周 3 次，每次 40~60 分钟。"
    else:
        schedule = [{"week": w, "main": "维持性综合训练", "sessions": 2}
                    for w in range(1, weeks + 1)]
        plan_note = ("各指标波动都低于 %.2f，整体已经比较稳定，"
                     "不建议再针对单项加练，保持每周 2 次即可。" % CV_CONCERN)

    data = {
        "weeks": weeks,
        "based_on": {
            "date": latest["date"],
            "video": Path(latest["video"]).name,
            "event_count": latest["event_count"],
            "sessions": len(usable),
        },
        "ranking": ranking,
        "focus": focus,
        "schedule": schedule,
        "review": {
            "after_weeks": weeks,
            "action": "用同样的机位和距离重录一段，再用 get_progress 对比 CV 变化",
            "metric": [f["metric"] for f in focus],
        },
        "note": plan_note + " 短板按 CV 排名得出；指标与训练项的对应关系为经验设定，"
                            "未经教练标注校准，请结合自身情况取舍。",
    }

    confidence = min(0.9, 0.4 + 0.1 * len(usable))
    return build_ok(data=data, evidence={"db_path": str(PROJECT_ROOT / "data/tennis.db")},
                    confidence=confidence, started=started).to_json()


if __name__ == "__main__":
    import json

    obj = json.loads(make_training_plan.invoke({"weeks": 4}))
    print("ok:", obj["ok"], "| error:", obj["error"])
    if obj["ok"]:
        print(json.dumps(obj["data"], ensure_ascii=False, indent=2)[:1500])
