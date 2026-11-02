"""
阶段 7 的事件流：把主图的 stream 翻译成前端能消费的事件。

事件：
    {"type":"dispatch", "subagent":"装备顾问", "description":"..."}
    {"type":"subagent_done", "chars": 754}
    {"type":"final", "content": "..."}
    {"type":"error", "message": "...", "error_id": "ab12cd34"}
"""

import logging
import uuid
from typing import Iterator

from langchain_core.messages import HumanMessage

from tennis_coach.agent.orchestrator import extract_citation, main_agent
from tennis_coach.core.store import set_thread_id, set_user_id

logger = logging.getLogger(__name__)


def stream_run(query: str, thread_id: str, user_id: str = "local") -> Iterator[dict]:
    """逐事件产出一次任务的结果。调用方负责推送。"""
    # 工具层通过 ContextVar 读取当前会话 id，用于给落库记录标注来源。
    # 放在编排入口注入，而不是让模型在工具参数里填。
    set_thread_id(thread_id)
    set_user_id(user_id)
    citation = None
    try:
        for chunk in main_agent.stream(
            {"messages": [HumanMessage(query)]},
            config={"recursion_limit": 10, "configurable": {"thread_id": thread_id}},
        ):
            for _node, state in chunk.items():
                msgs = state.get("messages") if isinstance(state, dict) else None
                if not isinstance(msgs, list) or not msgs:
                    continue
                last = msgs[-1]

                if getattr(last, "tool_calls", None):
                    for tc in last.tool_calls:
                        yield {"type": "dispatch",
                               "subagent": tc["args"].get("subagent_type"),
                               "description": tc["args"].get("description", "")[:120]}
                elif getattr(last, "type", "") == "tool":
                    if getattr(last, "name", "") == "task":
                        citation = extract_citation(str(last.content)) or citation
                    yield {"type": "subagent_done", "chars": len(str(last.content))}
                elif last.content:
                    text = str(last.content)
                    if citation and "（依据：" not in text:
                        text = f"{text}\n\n（{citation}）"
                    yield {"type": "final", "content": text}
    except Exception as e:
        error_id = uuid.uuid4().hex[:8]
        logger.exception("任务执行异常 trace=%s thread_id=%s", error_id, thread_id)
        yield {"type": "error",
               "message": f"任务失败（错误码 {error_id}）：{type(e).__name__}",
               "error_id": error_id}
