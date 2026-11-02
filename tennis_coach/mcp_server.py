"""
把网球工具暴露成 MCP 服务（stdio 传输）。

为什么单独做这一层：
    现在的工具是进程内的 LangChain StructuredTool，只能被本项目的 agent 用。
    包成 MCP 之后，Claude Desktop / Cursor / 任何 MCP 客户端都能直接调用，
    工具才从"某个项目的内部函数"变成"可复用能力"。

为什么复用原函数而不是重写：
    工具层已经返回统一结构（ok/data/evidence/confidence），MCP 只换传输层，
    重写只会让两份逻辑慢慢分叉。这里只做参数透传和 stdout 隔离。

启动：
    python -m tennis_coach.mcp_server

接进 Claude Desktop 的配置示例：
    {
      "mcpServers": {
        "tennis-coach": {
          "command": "/home/qjx/.conda/envs/tennis/bin/python",
          "args": ["-m", "tennis_coach.mcp_server"],
          "cwd": "/home/qjx/paper_agent"
        }
      }
    }
"""

import contextlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# MCP SDK 1.x 用 FastMCP，2.x 改名为 MCPServer。
# 装 langchain-mcp-adapters 会把 mcp 从 2.x 降到 1.x，所以两边都得兼容。
try:                                                             # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _ServerFactory      # noqa: E402
except ImportError:                                              # mcp 2.x
    from mcp.server.mcpserver import MCPServer as _ServerFactory  # noqa: E402

from tennis_coach.tools.analyze_stroke import analyze_stroke    # noqa: E402
from tennis_coach.tools.history import get_progress             # noqa: E402
from tennis_coach.tools.knowledge import search_technique      # noqa: E402
from tennis_coach.tools.rackets import filter_rackets           # noqa: E402
from tennis_coach.tools.training import make_training_plan     # noqa: E402


@contextlib.contextmanager
def _stdout_to_stderr():
    """stdio 传输下 stdout 是 JSON-RPC 通道，任何多余输出都会破坏协议。

    视频分析管线内部会用 print 打进度，所以每次调用工具期间把 stdout
    重定向到 stderr，调用结束再恢复。
    """
    with contextlib.redirect_stdout(sys.stderr):
        yield


def _unwrap(raw: str) -> dict:
    """工具返回的是 JSON 字符串，转成 dict 交给 MCP 序列化。"""
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {"ok": False, "error": "工具返回了无法解析的内容", "raw": str(raw)[:500]}


_INSTRUCTIONS = (
    "网球击球分析与技术知识服务。所有工具返回统一结构 "
    "{ok, data, evidence, confidence, error}。"
    "ok=false 时请把 error 原样转述给用户，不要自行补全细节；"
    "检索类工具返回空结果时，应明确说明未收录，禁止凭印象编造。"
)

try:
    mcp = _ServerFactory("tennis-coach", instructions=_INSTRUCTIONS)      # 1.x
except TypeError:
    mcp = _ServerFactory(name="tennis-coach", instructions=_INSTRUCTIONS)  # 2.x


@mcp.tool(name="search_technique")
def _search_technique(query: str, top_k: int = 3) -> dict:
    """检索网球知识库：技术动作（正手/反手/发球/步伐）、体能伤病、战术、
    以及比赛规则（计分、抢七、发球判罚、换边、场地边界、器材规格）。

    Args:
        query: 中文自然语言问题，例如「正手击球点应该在哪」
        top_k: 返回条数，1~5

    Returns:
        hits 为空表示知识库没有收录，应如实说明，不要编造。
    """
    with _stdout_to_stderr():
        return _unwrap(search_technique.invoke({"query": query, "top_k": top_k}))


@mcp.tool(name="analyze_stroke")
def _analyze_stroke(video_path: str, wrist: str = "auto",
                        model_size: str = "lite") -> dict:
    """分析一段网球击球视频，返回击球次数与动作指标（均为体高归一化的无量纲值）。

    单次分析耗时约 20~40 秒，请确保客户端超时设置足够长。

    适用：单人、固定机位、侧面拍摄、全身入镜、主体高度占画面 1/3 以上。
    不适用：多人同框、镜头切换的转播或教学剪辑、严重遮挡、远景。

    Args:
        video_path: 相对项目根目录的路径，如 data/uploads/forehand.mp4
        wrist: auto / left / right
        model_size: lite（快）或 full（更慢，实测在这类视频上更抖）
    """
    with _stdout_to_stderr():
        return _unwrap(analyze_stroke.invoke({
            "video_path": video_path,
            "wrist": wrist,
            "model_size": model_size,
        }))


@mcp.tool(name="get_progress")
def _get_progress(days: int = 30) -> dict:
    """查询历史训练记录的纵向对比（首次 vs 最近）：各指标的均值与变异系数变化。

    insufficient=true 表示可对比记录不足 2 次，此时应如实说明，不要编造趋势。

    Args:
        days: 回看最近多少天，1~365
    """
    with _stdout_to_stderr():
        return _unwrap(get_progress.invoke({"days": days}))


@mcp.tool(name="make_training_plan")
def _make_training_plan(weeks: int = 4) -> dict:
    """根据历史分析记录，按变异系数找出最不稳定的指标，生成分周训练计划。

    需要至少一条有效历史记录（每次至少识别出 2 次击球），否则返回 ok=false。

    Args:
        weeks: 计划周期（周），1~12
    """
    with _stdout_to_stderr():
        return _unwrap(make_training_plan.invoke({"weeks": weeks}))


@mcp.tool(name="filter_rackets")
def _filter_rackets(max_price_cny: int, level: str,
                    max_weight_g: int | None = None,
                    arm_issue: bool = False,
                    prefer: str = "control") -> dict:
    """按预算、水平、重量与护肘需求筛选网球拍，返回命中条数与 Top3 得分拆解。

    Args:
        max_price_cny: 预算上限（元）
        level: beginner / intermediate / advanced
        max_weight_g: 重量上限（克），不填表示不限
        arm_issue: 是否有肘部不适；True 时只保留护肘友好的球拍
        prefer: control / power / spin
    """
    with _stdout_to_stderr():
        return _unwrap(filter_rackets.invoke({
            "max_price_cny": max_price_cny, "level": level,
            "max_weight_g": max_weight_g, "arm_issue": arm_issue,
            "prefer": prefer,
        }))


if __name__ == "__main__":
    mcp.run(transport="stdio")
