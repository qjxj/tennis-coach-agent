"""
阶段 4：多智能体编排（星型拓扑，与 deepagents 的 subagents 同构）。

结构：
    主智能体（路由）
      ├─ task(subagent_type="技术分析师") → 子图 A：analyze_stroke
      └─ task(subagent_type="装备顾问")   → 子图 B：filter_rackets

学到什么：
    1. 子智能体 = 独立编译的子图，通过一个普通工具去 invoke
    2. 每个子图有自己的 system_prompt 和工具，上下文互不污染
    3. 子图结果回传给主智能体时会被"折损"，需要控制长度
"""

import json
import os
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv, find_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from tennis_coach.tools.analyze_stroke import analyze_stroke   # noqa: E402
from tennis_coach.tools.history import get_progress            # noqa: E402
from tennis_coach.tools.training import make_training_plan     # noqa: E402
from tennis_coach.tools.knowledge import search_technique      # noqa: E402
from tennis_coach.tools.rackets import filter_rackets          # noqa: E402
from tennis_coach.core.reliability import evaluate             # noqa: E402

load_dotenv(find_dotenv())

# 模型可以按角色分配，但默认全部走主模型。
# 实测过"装备顾问换 flash"：15 条用例仍能通过，速度快约 12%，
# 但约 1/6 的情况会把"没有匹配"说成"工具返回异常"，然后凭印象编造市场行情——
# 正是本项目要防的失败模式。因此回退，LLM_MODEL_GEAR 保留作为将来再试的入口。
_MAIN_MODEL_NAME = os.getenv("LLM_MODEL_MAIN") or os.getenv("LLM_QWEN_MAX")
_GEAR_MODEL_NAME = os.getenv("LLM_MODEL_GEAR") or _MAIN_MODEL_NAME


def _make_llm(name: str) -> ChatOpenAI:
    return ChatOpenAI(
        model=name,
        base_url=os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0,
    )


model = _make_llm(_MAIN_MODEL_NAME)          # 主调度器 + 技术分析师
gear_model = _make_llm(_GEAR_MODEL_NAME)     # 装备顾问

# ---------- 子智能体 ----------
STROKE_PROMPT = """你是网球技术分析师。你掌握四个工具：
  analyze_stroke     —— 分析一段击球视频，返回击球次数与动作指标
  get_progress       —— 查询历史记录，做纵向对比（首次 vs 最近）
  make_training_plan —— 按短板生成分周训练计划
  search_technique   —— 检索技术知识库（动作要领、常见错误、伤病预防）

规则：
- 用户给出视频路径时调用 analyze_stroke，然后基于返回值给结论。
- 用户问"有没有进步""比上次怎么样""最近状态如何"时调用 get_progress。
- 做纵向对比时只用 get_progress 的返回值，禁止为了对比去重新分析历史视频
  （会把同一次训练重复记账，污染趋势）。
- 用户问"该怎么练""给个训练计划""先练什么"时调用 make_training_plan。
  练什么必须由工具按 CV 排名决定，禁止自己凭印象指定训练重点。
- 用户问技术原理、动作要领、为什么会出错、伤病预防、比赛规则时调用 search_technique。
  讲解必须基于检索结果，并在结尾标注依据（source 字段）；
  hits 为空时明说"知识库没收录"，不要用通用印象补细节。
- get_progress 返回 insufficient=true 时，如实说"历史记录不足，还无法对比"，
  禁止编造过去的成绩或趋势。
- confidence < 0.6 必须说明"数据质量不足，仅供参考"。
- 不得编造返回值里没有的指标。"""

GEAR_PROMPT = """你是网球装备顾问。你掌握 filter_rackets 工具。
必须先调用工具，禁止凭印象推荐型号。
回答时说明：命中了哪几条硬约束、Top3 各自得分、为什么第 1 名更适合。
如果 matched = 0，直接说明预算/条件下没有匹配，并给出放宽建议。"""


# ---------- 工具级轨迹（评测用） ----------
# 只记录子智能体内部真正调用了哪个工具。主智能体的 task 调用由 ask_traced 单独收集。
# 用模块级列表而不是 ContextVar：LangGraph 可能在线程池里执行节点，
# ContextVar 不保证跨线程传播。该开关只在评测（串行）里打开，
# 服务端并发请求不会记录，因此不存在互相串扰。
_TOOL_TRACE: list = []
_RECORDING = False

def extract_citation(text: str) -> str | None:
    """从 task 的返回值里抠出"依据：xxx"。

    task 的返回是主图唯一能看到的子图产物。LangGraph 会把工具放到别的线程执行，
    ContextVar / 线程局部都传不回来，所以直接从文本里取最可靠。
    """
    marker = "（依据："
    idx = str(text).rfind(marker)
    if idx < 0:
        return None
    return str(text)[idx:].splitlines()[0].rstrip("）").strip()


def begin_trace() -> None:
    global _RECORDING
    _TOOL_TRACE.clear()
    _RECORDING = True


def end_trace() -> list:
    global _RECORDING
    _RECORDING = False
    return list(_TOOL_TRACE)


# 子智能体最多调用几轮工具。实测见过装备顾问把同一个工具连着调 4 次，
# 既慢又没意义；超过上限就不再绑工具，强制它收尾作答。
MAX_TOOL_ROUNDS = 3


def build_subagent(tools, system_prompt: str, name: str = "", llm=None,
                   max_rounds: int = MAX_TOOL_ROUNDS):
    """一个子智能体 = 一张独立的小图（model → tools → model）。

    max_rounds：该智能体最多调几轮工具。轻量模型常常拿到结果后还会
    反复重调（实测 flash 每次都会打满上限），所以按角色收紧。
    """
    llm = llm or model
    def model_node(state: MessagesState):
        msgs = [SystemMessage(system_prompt), *state["messages"]]
        used = sum(1 for m in state["messages"] if getattr(m, "type", "") == "tool")
        if used >= max_rounds:
            return {"messages": [llm.invoke(msgs)]}

        reply = llm.bind_tools(tools).invoke(msgs)
        if _RECORDING:
            for tc in getattr(reply, "tool_calls", None) or []:
                _TOOL_TRACE.append({"subagent": name,
                                    "tool": tc.get("name"),
                                    "args": tc.get("args", {})})
        return {"messages": [reply]}

    g = StateGraph(MessagesState)
    g.add_node("model", model_node)
    g.add_node("tools", ToolNode(tools))
    g.add_edge(START, "model")
    g.add_conditional_edges("model", tools_condition)
    g.add_edge("tools", "model")
    return g.compile()


DEFAULT_STROKE_TOOLS = [analyze_stroke, get_progress, make_training_plan, search_technique]
DEFAULT_GEAR_TOOLS = [filter_rackets]

# 子智能体注册表。做成可替换的，是为了同一套编排既能用进程内工具，
# 也能用 MCP 客户端加载来的远程工具——图结构本身不用改。
SUBAGENTS: dict = {}


def configure(stroke_tools=None, gear_tools=None) -> dict:
    """重建两个子智能体。传入 MCP 工具即可让整条链路走协议。"""
    SUBAGENTS.clear()
    SUBAGENTS.update({
        "技术分析师": build_subagent(stroke_tools or DEFAULT_STROKE_TOOLS,
                                STROKE_PROMPT, name="技术分析师", llm=model),
        "装备顾问": build_subagent(gear_tools or DEFAULT_GEAR_TOOLS,
                               GEAR_PROMPT, name="装备顾问",
                               llm=gear_model, max_rounds=1),
    })
    return SUBAGENTS


configure()


# ---------- 主智能体 ----------
@tool
def task(
    subagent_type: str,
    description: str,
) -> str:
    """把子任务交给专科助手执行，并返回它的最终答复。

    subagent_type：技术分析师（分析视频 / 对比进步 / 出计划 / 技术问答）/ 装备顾问（球拍推荐）
    description：给该助手的具体任务描述，要包含必要参数（如视频路径、预算、水平）
    """
    agent = SUBAGENTS.get(subagent_type)
    if agent is None:
        return f"未知子智能体：{subagent_type}，可选：{list(SUBAGENTS)}"

    result = agent.invoke({"messages": [HumanMessage(description)]},
                          config={"recursion_limit": 8})
    answer = str(result["messages"][-1].content)

    # 阶段 5：代码层门禁。只对视频分析生效——按工具名判断，而不是按子智能体名。
    tool_msgs = [m for m in result["messages"] if getattr(m, "type", "") == "tool"]
    stroke_msgs = [m for m in tool_msgs if getattr(m, "name", "") == "analyze_stroke"]
    if stroke_msgs:
        decision = evaluate(stroke_msgs[-1].content)
        if decision["status"] == "refuse":
            return f"[质量门禁] {decision['message']}"
        if decision["status"] == "warn":
            answer = f"[提示] {decision['message']}\n\n{answer}"

    answer = _ensure_citation(answer, tool_msgs)

    # 信息折损：子智能体可能产出很长文本，回传给主智能体前截断
    return answer if len(answer) <= 1200 else answer[:1200] + " ...[子智能体输出已截断]"


def _citation_from(tool_msgs: list) -> str | None:
    """从检索工具返回里取出第一条结果的 source。"""
    note_msgs = [m for m in tool_msgs if getattr(m, "name", "") == "search_technique"]
    if not note_msgs:
        return None
    try:
        payload = json.loads(str(note_msgs[-1].content))
        hits = (payload.get("data") or {}).get("hits") or []
        return hits[0].get("source") if hits else None
    except (json.JSONDecodeError, AttributeError, IndexError):
        return None


def _ensure_citation(answer: str, tool_msgs: list) -> str:
    """知识类回答必须标注依据。

    Prompt 里已经要求了，但实测模型经常忘（问"抢七几分算赢"就没写来源）。
    和门禁同一个思路：模型可以不说，代码补上。
    """
    if "（依据：" in answer:
        return answer
    source = _citation_from(tool_msgs)
    return f"{answer}\n\n（依据：{source}）" if source else answer


MAIN_PROMPT = """你是网球助手的总调度。

团队成员：
  1. 技术分析师 —— 分析视频、问进步、要训练计划、问技术要领与原理时调用
  2. 装备顾问   —— 用户问球拍选购、预算、水平、护肘时调用

硬性规则：
- 只要问题涉及球拍选购，必须调度装备顾问，即使参数齐全也不要自己回答。
- 只要用户给出视频并要求分析动作，必须调度技术分析师。
- 用户问"我有没有进步""和之前比怎么样""最近状态如何"这类需要历史的问题，
  也必须调度技术分析师。这类问题不需要视频路径，直接调度，不要追问视频。
- 用户问"该怎么练""给个训练计划""先练什么"时，同样直接调度技术分析师。
- 用户问网球技术原理、动作要领、常见错误、伤病预防、比赛规则时，必须调度技术分析师。
  即使你认为自己知道答案，也要先让技术分析师检索知识库——你的记忆可能过时或与知识库不一致。
- 一个问题里同时包含"分析视频"和"选球拍"时，必须分别调度两次（两次 task 调用）。
- 某个子任务被质量门禁拒绝，只代表这一项做不了；问题里的其它请求仍必须继续完成，
  不能因为一次拒答就当作整个任务结束。
- 同一位助手在一次回答里只调度一次，禁止重复调用。
- description 里要带齐参数（视频路径 / 预算 / 水平 / 是否有肘部问题）。
- 用户没给关键参数（如预算）就先追问，不要编。
- 与网球无关的请求直接说明无法处理，不要调度。
"""


# 多轮对话只回传最近若干条，避免上下文随轮次线性膨胀。
# 截断点必须避开 ToolMessage：它的 tool_call_id 依赖前面那条 AIMessage，
# 只留半截会让部分模型报 400。
HISTORY_WINDOW = 12


def _recent(messages):
    recent = messages[-HISTORY_WINDOW:]
    while recent and getattr(recent[0], "type", "") == "tool":
        recent = recent[1:]
    return recent


def main_node(state: MessagesState):
    msgs = [SystemMessage(MAIN_PROMPT), *_recent(state["messages"])]
    return {"messages": [model.bind_tools([task]).invoke(msgs)]}


g = StateGraph(MessagesState)
g.add_node("model", main_node)
g.add_node("tools", ToolNode([task]))
g.add_edge(START, "model")
g.add_conditional_edges("model", tools_condition)
g.add_edge("tools", "model")
main_agent = g.compile(checkpointer=MemorySaver())


def ask_traced(query, thread_id: str = None) -> dict:
    """执行一次或多次对话，返回回答 + 调度轨迹 + 子智能体工具轨迹（评测用）。

    query 传字符串 = 单轮；传列表 = 多轮（共用同一个 thread_id，用于测记忆）。

    每次调用默认新建 thread_id：挂了 checkpointer 之后，用例之间共用会话会互相污染记忆。
    """
    turns = [query] if isinstance(query, str) else list(query)
    config = {"recursion_limit": 10,
              "configurable": {"thread_id": thread_id or f"eval-{uuid.uuid4().hex[:12]}"}}

    final, dispatches, refusals = "", [], []
    citation = None
    begin_trace()
    try:
        for text in turns:
            final = ""                      # 只保留最后一轮的回答
            for chunk in main_agent.stream({"messages": [HumanMessage(text)]}, config=config):
                for _node, state in chunk.items():
                    msgs = state.get("messages") if isinstance(state, dict) else None
                    if not isinstance(msgs, list) or not msgs:
                        continue
                    last = msgs[-1]
                    if getattr(last, "tool_calls", None):
                        for tc in last.tool_calls:
                            dispatches.append(tc["args"].get("subagent_type"))
                    elif getattr(last, "type", "") == "tool":
                        content = str(last.content)
                        if "[质量门禁]" in content:
                            refusals.append(content)
                        if getattr(last, "name", "") == "task":
                            citation = extract_citation(content) or citation
                    elif last.content:
                        final = str(last.content)
    finally:
        tools = end_trace()

    # 主智能体会重写一遍子智能体的答复，经常把"依据"那行丢掉，所以在最外层再补一次
    if citation and final and "（依据：" not in final:
        final = f"{final}\n\n（{citation}）"

    return {"answer": final, "dispatches": dispatches, "refusals": refusals,
            "tools": tools, "turns": turns}


def ask(query: str) -> str:
    result = ask_traced(query)
    for name in result["dispatches"]:
        print(f"  [调度] {name}")
    return result["answer"]


if __name__ == "__main__":
    queries = [
        "预算 1000 元，我是初学者，手肘有点不舒服，想要控制好的球拍",
        "third_party/tennis-tracking/VideoInput/video_input3.mp4 这段正手帮我看看",
        "你好，你能做什么",
    ]
    for q in queries:
        print(f"\n用户：{q}\n{'-' * 60}")
        print("回答：", ask(q)[:400], "...")
