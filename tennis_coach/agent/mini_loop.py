"""
阶段 2：最小 Agent 循环（不依赖 deepagents，手写 30 行核心）。

四步循环：
    ① 把 messages + tools(schema) 发给模型
    ② 模型回 tool_calls（JSON，不执行代码）
    ③ ToolNode 解析 arguments，真正调用 analyze_stroke
    ④ 结果作为 ToolMessage 回填，再发一次，直到没有 tool_calls

这个文件就是 deepagents 的"内核"：一个带条件边的图。

运行：
    python -m tennis_coach.agent.mini_loop
    python -m tennis_coach.agent.mini_loop "帮我看看这段正手"
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from tennis_coach.tools.analyze_stroke import analyze_stroke   # noqa: E402
from tennis_coach.core.context import (                        # noqa: E402
    compact_tool_result, estimate_tokens, prune_history,
)

load_dotenv(find_dotenv())

# ---------- 模型 ----------
model = ChatOpenAI(
    model=os.getenv("LLM_QWEN_MAX"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0,
)

TOOLS = [analyze_stroke]

SYSTEM_PROMPT = """你是网球技术分析助手。

规则：
1. 只要用户提到分析击球视频，就调用 analyze_stroke，不要自己猜测动作。
2. 调用前先确认视频路径；用户没给路径就问，不要编。
3. 拿到工具结果后：
   - confidence < 0.6 时，必须说明"视频质量不足，结论仅供参考"，不要给确定性判断
   - event_count == 0 时，直接说明没检出击球，建议重拍
   - 否则基于 summary 里的数值给 1~3 条具体建议，每条都要引用数字
4. 不要编造 summary 里没有的指标。
"""


# ---------- 节点 ----------
def model_node(state: MessagesState):
    """② 模型这一轮只决定：说话 or 下单。"""
    # 阶段 3：较早的工具结果折叠，避免多轮对话线性膨胀
    history = prune_history(state["messages"], keep_recent_tools=1)
    messages = [SystemMessage(SYSTEM_PROMPT), *history]
    response = model.bind_tools(TOOLS).invoke(messages)
    return {"messages": [response]}


_tool_node = ToolNode(TOOLS)


def tools_node(state: MessagesState):
    """③ 执行工具，并把结果压缩后再回填。"""
    result = _tool_node.invoke(state)
    for msg in result.get("messages", []):
        before = len(str(msg.content))
        msg.content = compact_tool_result(str(msg.content))
        print(f"  [上下文] 工具结果 {before} → {len(msg.content)} 字符"
              f"（约 {estimate_tokens(str(msg.content))} token）")
    return result


# ---------- 图 ----------
graph = StateGraph(MessagesState)
graph.add_node("model", model_node)
graph.add_node("tools", tools_node)                # ③ 解析 arguments 并执行
graph.add_edge(START, "model")
graph.add_conditional_edges("model", tools_condition)   # 有 tool_calls → tools，否则结束
graph.add_edge("tools", "model")                   # ④ 回填后回到模型
app = graph.compile()


def ask(query: str, show_trace: bool = True) -> str:
    final = ""
    step = 0
    for chunk in app.stream({"messages": [HumanMessage(query)]},
                            config={"recursion_limit": 8}):
        for _node, state in chunk.items():
            messages = state.get("messages") if isinstance(state, dict) else None
            if not isinstance(messages, list) or not messages:
                continue
            last = messages[-1]
            step += 1

            if isinstance(last, ToolMessage):
                if show_trace:
                    preview = str(last.content)[:160].replace("\n", " ")
                    print(f"  [{step}] 工具返回 <- {last.name}: {preview}...")
            elif getattr(last, "tool_calls", None):
                if show_trace:
                    for tc in last.tool_calls:
                        print(f"  [{step}] 模型下单 -> {tc['name']}({tc['args']})")
            elif last.content:
                final = last.content
                if show_trace:
                    print(f"  [{step}] 模型最终回答完成")

    return final


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else (
        "帮我分析 third_party/tennis-tracking/VideoInput/video_input3.mp4 这段正手"
    )
    print(f"\n用户：{query}\n{'-' * 60}")
    answer = ask(query)
    print(f"{'-' * 60}\n最终回答：\n{answer}")
