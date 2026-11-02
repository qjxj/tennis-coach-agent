"""
阶段 6：评测。

断言"过程"而不是"文本"：
    该调度谁、不该调度谁、该拒答还是该给建议。

用法：
    python -m tennis_coach.evals.run_eval
    python -m tennis_coach.evals.run_eval --only gear_basic
    python -m tennis_coach.evals.run_eval --list
"""

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from tennis_coach.agent.orchestrator import ask_traced   # noqa: E402

CASES_PATH = Path(__file__).parent / "cases.jsonl"


def load_cases():
    return [json.loads(line) for line in
            CASES_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def check(case: dict, result: dict) -> list:
    errors = []
    dispatched = result["dispatches"]
    answer = result["answer"]

    for name in case.get("expect_dispatch", []):
        if name not in dispatched:
            errors.append(f"未调度 {name}（实际 {dispatched}）")

    for name in case.get("forbid", []):
        if name in dispatched:
            errors.append(f"不该调度 {name}")

    if case.get("expect_no_dispatch") and dispatched:
        errors.append(f"不该调度任何子智能体（实际 {dispatched}）")

    if case.get("expect_refuse") and not result["refusals"] and "质量门禁" not in answer:
        errors.append("未触发质量门禁拒答")

    for kw in case.get("must_contain", []):
        if kw not in answer:
            errors.append(f"回答缺少关键词「{kw}」")

    if (any_kw := case.get("must_contain_any")) and not any(k in answer for k in any_kw):
        errors.append(f"回答未包含任一关键词 {any_kw}")

    if not answer.strip():
        errors.append("回答为空")

    # 工具级断言：子智能体内部到底调了哪个工具
    used = [t["tool"] for t in result.get("tools", [])]
    for name in case.get("expect_tool", []):
        if name not in used:
            errors.append(f"未调用工具 {name}（实际 {used}）")
    for name in case.get("forbid_tool", []):
        if name in used:
            errors.append(f"不该调用工具 {name}")

    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", help="只跑某个用例 id")
    parser.add_argument("--list", action="store_true", help="列出用例")
    parser.add_argument("--repeat", type=int, default=1,
                        help="每个用例重复跑几次，用于看稳定性（默认 1）")
    args = parser.parse_args()

    cases = load_cases()
    if args.list:
        for c in cases:
            label = " | ".join(c["turns"]) if c.get("turns") else c["query"]
            print(f"{c['id']:18s} [{c['type']}] {label[:56]}")
        return

    if args.only:
        cases = [c for c in cases if c["id"] == args.only] or cases[:1]

    results = []
    for case in cases:
        t0 = time.time()
        runs = []
        first_errors = []
        for _ in range(args.repeat):
            try:
                result = ask_traced(case.get("turns") or case["query"])
                errors = check(case, result)
            except Exception as e:
                result, errors = ({"dispatches": [], "answer": "", "tools": []},
                                  [f"异常：{type(e).__name__}: {e}"])
            runs.append({"ok": not errors, "dispatches": result["dispatches"],
                         "tools": [t["tool"] for t in result.get("tools", [])]})
            if errors and not first_errors:
                first_errors = errors

        elapsed = time.time() - t0
        pass_count = sum(r["ok"] for r in runs)
        status = ("PASS" if pass_count == args.repeat
                  else "FLAKY" if pass_count else "FAIL")
        results.append({"id": case["id"], "type": case["type"], "ok": pass_count == args.repeat,
                        "flaky": 0 < pass_count < args.repeat,
                        "pass_count": pass_count, "repeat": args.repeat,
                        "errors": first_errors, "elapsed": round(elapsed, 1)})
        print(f"[{status}] {case['id']:18s} {pass_count}/{args.repeat}  {elapsed:5.1f}s  "
              f"调度={runs[0]['dispatches']} 工具={runs[0]['tools']}")
        for e in first_errors:
            print(f"        ✗ {e}")

    total = len(results)
    stable = sum(r["ok"] for r in results)
    flaky = [r["id"] for r in results if r["flaky"]]
    failed = [r["id"] for r in results if r["pass_count"] == 0]
    print("-" * 62)
    print(f"稳定通过 {stable}/{total}（每个用例跑 {args.repeat} 次）")
    if flaky:
        print(f"不稳定：{flaky}")
    if failed:
        print(f"全失败：{failed}")
    by_type = {}
    for r in results:
        g = by_type.setdefault(r["type"], [0, 0])
        g[0] += r["ok"]
        g[1] += 1
    for t, (p, n) in by_type.items():
        print(f"  {t:10s} {p}/{n}")


if __name__ == "__main__":
    main()
