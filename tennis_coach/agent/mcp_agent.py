"""
让整条编排走 MCP 协议：工具不再是进程内导入，而是通过 stdio 客户端加载。

和直接导入的区别：
    直接导入  ── 工具是 Python 对象，和 agent 同一个进程、同一个内存空间
    MCP 路径  ── agent 通过 JSON-RPC 调用一个独立进程，工具可被替换、可跨语言

为什么编排图不用改：
    子智能体的工具集是从 SUBAGENTS 注册表里取的，configure() 换掉工具，
    图结构、门禁、评测断言全部复用。

一个必须处理的细节：
    MCP 工具返回的是 content blocks（[{"type":"text","text":"{...}"}]），
    而门禁和上下文压缩都按 JSON 字符串解析。所以这里要包一层归一化，
    否则 evaluate() 会解析失败、把所有结果都判成 refuse。

用法：
    python -m tennis_coach.agent.mcp_agent
"""

import asyncio
import json
import sys
import time
from pathlib import Path

from langchain_core.tools import StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from tennis_coach.agent import orchestrator        # noqa: E402

SERVER_CONFIG = {
    "tennis-coach": {
        "command": sys.executable,
        "args": ["-m", "tennis_coach.mcp_server"],
        "transport": "stdio",
        "cwd": str(PROJECT_ROOT),
    }
}

STROKE_TOOL_NAMES = ("analyze_stroke", "get_progress",
                     "make_training_plan", "search_technique")
GEAR_TOOL_NAMES = ("filter_rackets",)


def _as_text(result) -> str:
    """把 MCP 的 content blocks 归一化成 JSON 字符串。

    门禁（reliability.evaluate）和上下文压缩都假设工具返回 JSON 文本，
    不归一化的话它们会拿到 "[{'type': 'text', ...}]" 这种 Python repr。
    """
    if isinstance(result, str):
        return result
    if isinstance(result, list):
        parts = []
        for block in result:
            if isinstance(block, dict):
                parts.append(block.get("text") or json.dumps(block, ensure_ascii=False))
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return str(result)


def _wrap(tool) -> StructuredTool:
    """MCP 工具 → 本地结构化工具，只做结果归一化。

    适配器给出的工具是**纯异步**的（func=None，只有 coroutine），
    直接 invoke 会抛 NotImplementedError。编排图是同步的，
    所以在包装层用 asyncio.run 把异步调用转成同步。
    注意：调用点不能处在已运行的事件循环里（本项目在 worker 线程里跑编排，满足）。
    """
    def _call(**kwargs):
        return _as_text(asyncio.run(tool.ainvoke(kwargs)))

    return StructuredTool(
        name=tool.name,
        description=tool.description,
        args_schema=getattr(tool, "args_schema", None),
        func=_call,
    )


def load_tools() -> dict:
    """启动 MCP 服务、拉取工具清单并包装。只在启动时执行一次。"""
    tools = asyncio.run(MultiServerMCPClient(SERVER_CONFIG).get_tools())
    return {t.name: _wrap(t) for t in tools}


def configure() -> dict:
    """把编排层的工具集换成 MCP 工具。"""
    by_name = load_tools()
    missing = [n for n in STROKE_TOOL_NAMES + GEAR_TOOL_NAMES if n not in by_name]
    if missing:
        raise RuntimeError(f"MCP 服务缺少工具: {missing}")

    orchestrator.configure(
        stroke_tools=[by_name[n] for n in STROKE_TOOL_NAMES],
        gear_tools=[by_name[n] for n in GEAR_TOOL_NAMES],
    )
    return by_name


def main():
    started = time.time()
    by_name = configure()
    print(f"[MCP] 已通过协议加载 {len(by_name)} 个工具，耗时 {time.time() - started:.1f}s")
    print(f"[MCP] 工具: {sorted(by_name)}")

    queries = [
        "抢七要打到几分才算赢？",
        "预算1000元，初学者，手肘不太舒服，帮我选个拍子",
    ]
    for q in queries:
        t0 = time.time()
        result = orchestrator.ask_traced(q)
        print("-" * 60)
        print(f"用户：{q}")
        print(f"调度={result['dispatches']} 工具={[t['tool'] for t in result['tools']]} "
              f"耗时={time.time() - t0:.1f}s")
        print("回答：", result["answer"][:220].replace("\n", " "))


if __name__ == "__main__":
    main()
