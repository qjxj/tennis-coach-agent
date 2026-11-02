"""
阶段 10：技术知识检索（自建语料 + BM25）。

为什么不用向量库：
    语料只有 27 条，纯 Python BM25 已经够用 —— 没有额外依赖、离线可跑、
    打分可解释（能说清"命中了哪些词"）。
    将来语料上千条、或者要跨语言检索时，只需把 _build_index 换成 embedding，
    search_technique 的调用方不用改。

为什么不用 jieba 分词：
    中文按「单字 + 相邻二元组」切分，对短查询（"正手击球点"）已经够用，
    省掉一个依赖和词典维护成本。idf 会自然压低"的""是"这类高频单字。

为什么检索必须是工具：
    知识库内容如果不经检索就塞进 system prompt，会挤占每一轮的上下文；
    而且模型会"记得"某些条目却记错细节。让它查，而不是让它背。
"""

import math
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Annotated, Any, Dict, List, Tuple

from langchain_core.tools import tool

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from tennis_coach.core.schema import build_err, build_ok     # noqa: E402

KB_PATH = PROJECT_ROOT / "tennis_coach" / "knowledge" / "technique.json"

K1, B = 1.5, 0.75          # BM25 常用参数
MIN_COVERAGE = 0.20        # 命中比例低于此值：视为不相关（与得分门槛二选一）
MIN_SCORE = 6.0            # 绝对得分门槛：长问句覆盖度天然偏低，靠得分兜底
WEAK_COVERAGE = 0.25       # 低于此值：返回但标记 weak_match，让模型自己谨慎

_CJK = re.compile(r"[\u4e00-\u9fff]+")
_ASCII = re.compile(r"[a-z0-9_]+")

_index: Dict[str, Any] = {}


def _tokenize(text: str) -> List[str]:
    """中文切单字 + 二元组，英文数字按词切。"""
    text = text.lower()
    tokens = _ASCII.findall(text)
    for seg in _CJK.findall(text):
        tokens.extend(seg)
        tokens.extend(seg[i:i + 2] for i in range(len(seg) - 1))
    return tokens


def _doc_text(entry: Dict[str, Any]) -> str:
    return " ".join([
        entry["title"], entry["summary"], entry["scene"],
        " ".join(entry.get("key_points") or []),
        " ".join(entry.get("common_errors") or []),
        " ".join(entry.get("tags") or []),
    ])


def _build_index() -> Dict[str, Any]:
    """建立倒排与 idf。语料很小，全量放内存即可。"""
    import json

    raw = json.loads(KB_PATH.read_text(encoding="utf-8"))
    entries = raw["entries"]

    docs = [Counter(_tokenize(_doc_text(e))) for e in entries]
    df = Counter()
    for d in docs:
        df.update(d.keys())
    n = len(docs)
    idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
    avgdl = sum(sum(d.values()) for d in docs) / max(n, 1)

    return {"entries": entries, "docs": docs, "idf": idf, "avgdl": avgdl,
            "version": raw.get("version", 0)}


def _get_index() -> Dict[str, Any]:
    if not _index:
        _index.update(_build_index())
    return _index


def search(query: str, top_k: int = 3) -> Tuple[List[Dict[str, Any]], List[str]]:
    """返回 [(条目, 得分, 覆盖度)] 与命中的查询词。"""
    idx = _get_index()
    q_tokens = _tokenize(query)

    # 只用「二元组及以上」参与打分：单字（天、样、么）在中文里太容易误命中，
    # 曾出现"今天天气怎么样"匹配到"每拍准备动作都不一样"（命中单字"样"）。
    # 查询本身没有二元组时（如单字"肘"），才退回用单字。
    q_terms = {t for t in q_tokens if len(t) >= 2} or set(q_tokens)
    if not q_terms:
        return [], []

    scored = []
    for entry, doc in zip(idx["entries"], idx["docs"]):
        dl = sum(doc.values()) or 1
        score = 0.0
        for t in q_terms:
            f = doc.get(t, 0)
            if not f:
                continue
            score += idx["idf"].get(t, 0.0) * f * (K1 + 1) / (
                f + K1 * (1 - B + B * dl / idx["avgdl"]))
        if score <= 0:
            continue
        hit_terms = sorted(t for t in q_terms if t in doc)
        coverage = len(hit_terms) / len(q_terms)
        # 覆盖度低说明只是撞上了个别字词（"网球鞋"撞到"网球肘"），给模型会误导；
        # 但模型改写出的长问句覆盖度天然偏低，所以得分足够高时仍然放行。
        if coverage < MIN_COVERAGE and score < MIN_SCORE:
            continue
        scored.append((entry, round(score, 3), round(coverage, 3), hit_terms))

    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:top_k]
    matched = top[0][3] if top else []
    return top, matched


@tool
def search_technique(
    query: Annotated[str, "要查的技术问题，用中文自然语言描述，如「正手击球点应该在哪」"],
    top_k: Annotated[int, "返回条数，1~5，默认 3"] = 3,
) -> str:
    """检索网球知识库：技术动作（正手/反手/发球/步伐）、体能伤病、战术、
    以及比赛规则（计分、抢七、发球判罚、换边、场地边界、器材规格）。
    返回要点与常见错误。

    每条结果包含：
      title / summary        要点与一句话解释
      key_points             2~3 条可执行的技术要领
      common_errors          常见错误
      metric                 对应 analyze_stroke 的哪个指标（null 表示没有对应）
      source                 依据来源，回答时必须转述
      score / coverage       检索得分与查询词命中比例

    另有 related：同一场景下的其它条目，可在回答末尾作为延伸推荐。

    适用：用户问技术原理、动作要领、"为什么会这样"、规则与训练常识。
    不适用：分析用户自己的视频（用 analyze_stroke）、推荐球拍（装备顾问）。
    注意：hits 为空表示知识库没有收录，必须如实说明"知识库中没有相关条目"，
          不要用通用印象补写细节。
    """
    started = time.time()
    top_k = max(1, min(int(top_k), 5))

    try:
        top, matched = search(query, top_k)
    except Exception as e:
        return build_err(f"知识库检索失败：{type(e).__name__}: {e}", started).to_json()

    if not top:
        return build_ok(
            data={"query": query, "hits": [], "related": [], "matched_terms": [],
                  "note": "知识库中没有匹配条目。请如实说明未收录，不要编造细节。"},
            evidence={"kb_path": str(KB_PATH)},
            confidence=0.1, started=started,
        ).to_json()

    hits = []
    for entry, score, coverage, _ in top:
        hits.append({
            "id": entry["id"],
            "scene": entry["scene"],
            "title": entry["title"],
            "summary": entry["summary"],
            "key_points": entry.get("key_points") or [],
            "common_errors": entry.get("common_errors") or [],
            "metric": entry.get("metric"),
            "source": entry.get("source"),
            "score": score,
            "coverage": coverage,
        })

    # 同场景的其它条目：把"一问一答"变成"可以顺着看下去"
    shown = {h["id"] for h in hits}
    scene = hits[0]["scene"]
    related = [{"id": e["id"], "title": e["title"]}
               for e in _get_index()["entries"]
               if e.get("scene") == scene and e["id"] not in shown][:3]

    best_coverage = hits[0]["coverage"]
    # 覆盖度比得分更可解释：得分受文档长度影响，覆盖度直接回答"问的词命中了多少"
    confidence = round(min(0.9, 0.3 + 0.6 * best_coverage), 3)

    return build_ok(
        data={"query": query, "hits": hits, "related": related,
              "matched_terms": matched,
              "weak_match": best_coverage < WEAK_COVERAGE},
        evidence={"kb_path": str(KB_PATH)},
        confidence=confidence, started=started,
    ).to_json()


if __name__ == "__main__":
    # 检索冒烟测试：期望命中的条目是否排在第一位
    import json

    CASES = [
        ("正手击球点应该在身体什么位置", "fh_contact_point"),
        ("网球肘怎么预防", "elbow_tennis_elbow"),
        ("发球抛球总是不稳", "serve_toss"),
        ("随挥要多完整才够", "fh_followthrough"),
        ("二发怎么才能更安全", "serve_second_kick"),
        ("分腿垫步什么时候做", "split_step"),
        ("抢七怎么计分，几分算赢", "rule_tiebreak_score"),
        ("换边休息几分钟", "rule_changeover_time"),
        ("双打打到单打线算出界吗", "rule_doubles_court"),
        ("发球踩线算什么", "rule_foot_fault"),
        ("球拍穿线磅数有什么限制", "rule_string_tension"),
        ("今天天气怎么样", None),
        ("网球鞋应该怎么挑", None),
    ]
    ok = 0
    for q, expect in CASES:
        top, matched = search(q, 1)
        got = top[0][0]["id"] if top else None
        passed = (got == expect) if expect else (got is None)
        ok += passed
        print(f"{'PASS' if passed else 'FAIL'}  {q:22s} -> {got}  (期望 {expect})")
    print(f"\nhit@1 = {ok}/{len(CASES)}")
    print(json.dumps(json.loads(search_technique.invoke(
        {"query": "正手击球点应该在身体什么位置"})), ensure_ascii=False)[:400])
