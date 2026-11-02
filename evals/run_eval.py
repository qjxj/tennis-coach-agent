"""
Agent 回归评测脚本。

核心思想：**断言过程，不断言文本**。
  - 该不该调度某个子智能体（assistant_call）
  - 该不该调用某个主智能体工具（tool_result）
  - 是否有最终输出（task_result）
  - 是否抛出未处理异常（error）

为什么不断言文本：LLM 输出天然不确定，比字符串只会带来 flaky 测试；
而"调度了谁、调了什么工具"是稳定的行为信号。

用法：
    python evals/run_eval.py                # 跑全部用例
    python evals/run_eval.py --only weather # 只跑指定用例
    python evals/run_eval.py --list         # 列出所有用例
"""

import argparse
import asyncio
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from api.monitor import monitor                    # noqa: E402
from agent.main_agent import run_deep_agent        # noqa: E402

OUTPUT_DIR = PROJECT_ROOT / "output"
UPDATED_DIR = PROJECT_ROOT / "updated"

DEFAULT_TIMEOUT = 180  # 单条用例超时（秒）


@dataclass
class Case:
    name: str
    query: str
    # 期望被调度的子智能体名（子集匹配，允许额外调度）
    assistants: List[str] = field(default_factory=list)
    # 明确不应该被调度的子智能体
    forbid: List[str] = field(default_factory=list)
    # 期望出现的主智能体工具函数名（来自 tools 节点的 tool_result）
    tools: List[str] = field(default_factory=list)
    # 是否要求"完全不调度任何子智能体"
    no_assistant: bool = False
    # 是否需要预置上传文件
    upload_files: Optional[List[str]] = None


CASES: List[Case] = [
    Case(
        name="chitchat",
        query="你好，只回复两个字：收到",
        no_assistant=True,
    ),
    Case(
        name="math",
        query="1+1 等于几？只回答数字",
        no_assistant=True,
    ),
    Case(
        name="weather",
        query="北京今天天气怎么样？",
        assistants=["天气查询助手"],
    ),
    Case(
        name="weather_boundary",
        query="上海天气如何？",
        assistants=["天气查询助手"],
        forbid=["市场情报助手", "数据分析助手", "内部知识助手"],
    ),
    Case(
        name="market",
        query="搜索一下最近 AI Agent 领域的最新进展",
        assistants=["市场情报助手"],
        forbid=["天气查询助手"],
    ),
    Case(
        name="db_tables",
        query="查一下数据库里都有哪些表",
        assistants=["数据分析助手"],
        forbid=["天气查询助手"],
    ),
    Case(
        name="rag_sop",
        query="查一下内部知识库里的运营 SOP",
        assistants=["内部知识助手"],
        forbid=["天气查询助手"],
    ),
    Case(
        name="doc_read",
        query="读取上传的 hello.txt，用一句话总结内容",
        tools=["read_file_content"],
        upload_files=["hello.txt"],
    ),
]


class Recorder:
    """临时接管 monitor._emit，把事件记录下来用于断言。"""

    def __init__(self):
        self.events = []
        self._original = monitor._emit

    def __enter__(self):
        def emit(event_type, message, data=None):
            self.events.append((event_type, message, data or {}))
            return self._original(event_type, message, data)

        monitor._emit = emit
        return self

    def __exit__(self, *exc_info):
        monitor._emit = self._original
        return False

    def assistants(self) -> List[str]:
        return [
            d.get("assistant_name")
            for t, _, d in self.events
            if t == "assistant_call"
        ]

    def tool_names(self) -> List[str]:
        return [
            d.get("tool_name")
            for t, _, d in self.events
            if t == "tool_result"
        ]

    def final_result(self) -> Optional[str]:
        for t, _, d in self.events:
            if t == "task_result":
                return d.get("result")
        return None

    def errors(self) -> List[str]:
        return [m for t, m, _ in self.events if t == "error"]


def prepare_session(session_id: str, files: Optional[List[str]]) -> str:
    """清理旧会话目录，并按需预置上传文件。返回 session_dir（绝对路径）。"""
    session_dir = OUTPUT_DIR / f"session_{session_id}"
    shutil.rmtree(session_dir, ignore_errors=True)
    session_dir.mkdir(parents=True, exist_ok=True)

    updated_dir = UPDATED_DIR / f"session_{session_id}"
    shutil.rmtree(updated_dir, ignore_errors=True)

    if files:
        updated_dir.mkdir(parents=True, exist_ok=True)
        for filename in files:
            (updated_dir / filename).write_text(
                "会议纪要：本周完成天气助手接入，下周做 PDF 替换。\n",
                encoding="utf-8",
            )

    return str(session_dir)


def check(case: Case, rec: Recorder) -> List[str]:
    """返回失败原因列表；空列表表示通过。"""
    failures = []
    assistants = rec.assistants()
    tools = rec.tool_names()
    final = rec.final_result()

    if case.no_assistant and assistants:
        failures.append(f"不应调度子智能体，实际调用了: {assistants}")

    for expected in case.assistants:
        if expected not in assistants:
            failures.append(f"期望调度 {expected}，实际: {assistants or '无'}")

    for forbidden in case.forbid:
        if forbidden in assistants:
            failures.append(f"不应调度 {forbidden}，实际: {assistants}")

    for tool in case.tools:
        if tool not in tools:
            failures.append(f"期望调用工具 {tool}，实际: {tools or '无'}")

    if not final:
        failures.append("没有产生最终结果（task_result 缺失）")

    errors = rec.errors()
    if errors:
        failures.append(f"出现未处理异常: {errors[0]}")

    return failures


async def run_case(case: Case, timeout: int) -> dict:
    session_id = f"eval-{case.name}"
    prepare_session(session_id, case.upload_files)

    started = time.time()
    with Recorder() as rec:
        try:
            await asyncio.wait_for(
                run_deep_agent(case.query, session_id),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            return {
                "name": case.name,
                "ok": False,
                "elapsed": time.time() - started,
                "failures": [f"超时（>{timeout}s）"],
                "assistants": rec.assistants(),
                "tools": rec.tool_names(),
            }

    failures = check(case, rec)
    return {
        "name": case.name,
        "ok": not failures,
        "elapsed": time.time() - started,
        "failures": failures,
        "assistants": rec.assistants(),
        "tools": rec.tool_names(),
        "final_preview": (rec.final_result() or "")[:60].replace("\n", " "),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="MultiAgent-Search 回归评测")
    parser.add_argument("--only", help="只跑指定用例名（逗号分隔）")
    parser.add_argument("--list", action="store_true", help="列出所有用例")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    args = parser.parse_args()

    if args.list:
        for c in CASES:
            print(f"- {c.name:18} {c.query}")
        return 0

    selected = CASES
    if args.only:
        wanted = {n.strip() for n in args.only.split(",") if n.strip()}
        selected = [c for c in CASES if c.name in wanted]
        if not selected:
            print(f"未找到用例: {args.only}")
            return 2

    results = []

    async def run_all():
        # 必须共用同一个事件循环：主智能体内部的 aiosqlite 连接绑定在首次创建时的 loop 上，
        # 每条用例单独 asyncio.run() 会让连接落到已关闭的 loop。
        collected = []
        for case in selected:
            print(f"\n{'=' * 60}\n[RUN] {case.name} | {case.query}\n{'=' * 60}")
            collected.append(await run_case(case, args.timeout))
        return collected

    results = asyncio.run(run_all())

    print(f"\n{'=' * 60}\n评测结果\n{'=' * 60}")
    passed = 0
    for r in results:
        status = "PASS" if r["ok"] else "FAIL"
        passed += r["ok"]
        print(f"[{status}] {r['name']:18} {r['elapsed']:6.1f}s  "
              f"调度={r['assistants'] or '-'} 工具={r['tools'] or '-'}")
        for reason in r["failures"]:
            print(f"        ✗ {reason}")

    total = len(results)
    print(f"\n通过 {passed}/{total}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # AsyncSqliteSaver 持有 aiosqlite 的非守护线程，正常退出会一直挂住，
    # 因此打印完结果后强制结束进程。
    os._exit(exit_code)
