"""
跨会话记忆：每次视频分析落一条记录，供"纵向对比"查询。

为什么是 SQLite 而不是向量库：
    这里要的是"按时间取数 + 聚合"，一条 SQL 就够；
    语义检索解决的是"找相似内容"，不是这个场景的问题。

为什么不放在 LangGraph 的会话状态里：
    checkpoint 按 thread_id 隔离，天然做不到"跨会话"。
    用户真正关心的是"我练了一个月有没有进步"，这件事必须落到会话之外。

为什么用 ContextVar 存 thread_id：
    工具签名里加一个 session_id 参数，模型未必老实填；
    由调用方在进入编排前注入，工具侧只读不写，才是可靠的做法。
"""

import json
import sqlite3
from contextlib import closing
from contextvars import ContextVar
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "data" / "tennis.db"

# 由 runner / 服务端写入，工具侧只读。默认值用于脚本直跑。
_THREAD_ID: ContextVar[str] = ContextVar("thread_id", default="local")
# 用户归属：同样由调用方注入。没有它，两个人的记录会被算成同一个人的两次训练，
# "有没有进步"就会拿别人的数据做对比。
_USER_ID: ContextVar[str] = ContextVar("user_id", default="local")

# stability 判定的死区：CV 变化小于这个幅度视为没变
CV_EPS = 0.05

METRIC_KEYS = ("wrist_speed", "contact_offset", "backswing_height",
               "followthrough_length", "elbow_angle_deg", "shoulder_angle_deg")


def set_thread_id(thread_id: str) -> None:
    _THREAD_ID.set(thread_id)


def get_thread_id() -> str:
    return _THREAD_ID.get()


def set_user_id(user_id: str) -> None:
    _USER_ID.set(user_id or "anonymous")


def get_user_id() -> str:
    return _USER_ID.get()


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with closing(_connect()) as conn, conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS analyses (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at     TEXT NOT NULL,   -- ISO8601，本地时间
                user_id        TEXT,            -- 归属用户；NULL 表示历史遗留数据
                thread_id      TEXT,
                video          TEXT,
                event_count    INTEGER,
                body_height_px REAL,
                usable_ratio   REAL,
                summary_json   TEXT             -- {指标: {mean, std, iqr, cv}}
            )
        """)
        # 老库没有 user_id 列，补上。历史行的 user_id 保持 NULL，
        # 它们不归属任何用户，因此不会混进任何人的趋势。
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(analyses)")}
        if "user_id" not in cols:
            conn.execute("ALTER TABLE analyses ADD COLUMN user_id TEXT")


def record_analysis(video: str,
                    event_count: int,
                    summary: Dict[str, Any],
                    quality: Optional[Dict[str, Any]] = None) -> int:
    """写入一次分析结果，返回记录 id。"""
    init_db()
    quality = quality or {}
    with closing(_connect()) as conn, conn:
        cur = conn.execute(
            "INSERT INTO analyses (created_at, user_id, thread_id, video, event_count,"
            " body_height_px, usable_ratio, summary_json) VALUES (?,?,?,?,?,?,?,?)",
            (
                datetime.now().isoformat(timespec="seconds"),
                get_user_id(),
                get_thread_id(),
                video,
                int(event_count),
                quality.get("body_height_px"),
                quality.get("usable_ratio"),
                json.dumps(summary, ensure_ascii=False),
            ),
        )
        return int(cur.lastrowid)


def fetch_analyses(days: int, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """取该用户最近 days 天的记录，按时间升序。

    必须按 user_id 过滤：否则多人的记录会被当成同一个人的训练序列，
    "有没有进步"就会拿别人的数据做对比。
    """
    init_db()
    user_id = user_id or get_user_id()
    cutoff = (datetime.now() - timedelta(days=days)).isoformat(timespec="seconds")
    with closing(_connect()) as conn, conn:
        rows = conn.execute(
            "SELECT * FROM analyses WHERE user_id = ? AND created_at >= ? "
            "ORDER BY created_at ASC",
            (user_id, cutoff),
        ).fetchall()

    out = []
    for r in rows:
        try:
            summary = json.loads(r["summary_json"] or "{}")
        except json.JSONDecodeError:
            summary = {}
        out.append({
            "id": r["id"],
            "date": r["created_at"],
            "user_id": r["user_id"],
            "thread_id": r["thread_id"],
            "video": r["video"],
            "event_count": r["event_count"],
            "body_height_px": r["body_height_px"],
            "summary": summary,
        })
    return out


def _stability(cv_delta: Optional[float]) -> Optional[str]:
    """CV 越小越稳，所以下降 = 变好。"""
    if cv_delta is None:
        return None
    if cv_delta < -CV_EPS:
        return "improved"
    if cv_delta > CV_EPS:
        return "worsened"
    return "stable"


def build_progress(days: int = 30) -> Dict[str, Any]:
    """把最近 days 天的记录整理成"首次 vs 最近"的对比。"""
    rows = fetch_analyses(days)

    # 只有 ≥2 次击球的记录才有 std/cv，也才谈得上"稳定性"
    usable = [r for r in rows if (r["event_count"] or 0) >= 2]
    history = [{"date": r["date"], "event_count": r["event_count"],
                "body_height_px": r["body_height_px"]} for r in rows[-20:]]

    result: Dict[str, Any] = {
        "sessions": len(rows),
        "usable_sessions": len(usable),
        "window_days": days,
        "insufficient": len(usable) < 2,
        "history": history,
    }
    if result["insufficient"]:
        return result

    # 只返回文件名，不返回可用路径：曾观察到子智能体拿到完整路径后
    # 会顺手重新分析历史视频，导致同一次训练被重复记账、趋势失真。
    first, latest = usable[0], usable[-1]
    result["first"] = {"date": first["date"], "video": Path(first["video"]).name,
                       "event_count": first["event_count"]}
    result["latest"] = {"date": latest["date"], "video": Path(latest["video"]).name,
                        "event_count": latest["event_count"]}

    # 尺度不一致时跨记录比较没有意义：同样是"击球点偏移"，
    # 222px 的近景和 203px 的远景量出来的噪声水平完全不同。
    heights = [r["body_height_px"] for r in usable if r["body_height_px"]]
    if heights and max(heights) / min(heights) > 1.4:
        result["setup_mismatch"] = True
        result["note"] = (f"历史记录的拍摄尺度差异较大（主体高度 "
                          f"{min(heights):.0f}~{max(heights):.0f}px），"
                          "不同机位/距离的结果不宜直接比较，趋势仅供参考。")

    changes: Dict[str, Any] = {}
    for key in METRIC_KEYS:
        # 只取有该指标的记录（侧面视角下 shoulder_angle 可能是 None）
        series = [(r["date"], r["summary"][key]) for r in usable
                  if isinstance(r["summary"].get(key), dict)]
        if len(series) < 2:
            continue

        mean_first = series[0][1].get("mean")
        mean_last = series[-1][1].get("mean")
        mean_delta = (None if mean_first is None or mean_last is None
                      else round(mean_last - mean_first, 3))

        cvs = [(d, s.get("cv")) for d, s in series if s.get("cv") is not None]
        cv_first = cvs[0][1] if len(cvs) >= 2 else None
        cv_last = cvs[-1][1] if len(cvs) >= 2 else None
        cv_delta = (None if cv_first is None
                    else round(cv_last - cv_first, 3))

        changes[key] = {
            "samples": len(series),
            "mean": [mean_first, mean_last],
            "mean_delta": mean_delta,
            "cv": [cv_first, cv_last],
            "cv_delta": cv_delta,
            "stability": _stability(cv_delta),
        }

    result["changes"] = changes
    return result
