# 定义一个网络搜索的工具！
# ======================== 导入核心依赖 ========================
from typing import Annotated, Literal
from langchain_core.tools import tool
from tavily import TavilyClient

import os
from dotenv import load_dotenv

from api.monitor import monitor
from utils.retry import retry_with_backoff

# ======================== 初始化配置 ========================
load_dotenv()

# Tavily 客户端（超时 30 秒，防止网络卡死）
tavily_client = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY"),
)


# ======================== 带重试的搜索封装 ========================

@retry_with_backoff(
    max_retries=3,
    base_delay=1.0,
    max_delay=10.0,
    exceptions=(Exception,),
)
def _search_with_retry(query: str, topic: str, max_results: int, include_raw_content: bool):
    """
    对 Tavily 搜索调用包装指数退避重试。
    网络抖动或 Tavily 临时不可用时自动重试，最多 3 次。
    """
    return tavily_client.search(
        query=query,
        topic=topic,
        max_results=max_results,
        include_raw_content=include_raw_content,
    )


# ======================== LangChain 工具定义 ========================

@tool
def internet_search(
    query: Annotated[str, "用户的查询信息"],
    topic: Annotated[
        Literal["news", "finance", "general"],
        "查询类型：general 通用 / news 新闻 / finance 财经",
    ] = "general",
    max_results: Annotated[int, "返回的最大条数"] = 5,
    include_raw_content: Annotated[
        bool, "是否返回网页原文：False 精简 / True 详细"
    ] = False,
):
    """根据用户问题搜索公开网络信息，返回搜索结果。

    适用范围：行业趋势、竞品动态、市场新闻等公开信息。
    不适用：数据库经营数据、企业内部知识库，这两类请交给对应的子智能体。
    """
    # 每次调用工具，都向前端推送调用进度
    monitor.report_tool(
        tool_name="网络搜索工具",
        args={
            "query": query,
            "topic": topic,
            "max_results": max_results,
            "include_raw_content": include_raw_content,
        },
    )

    # 带重试的搜索调用（指数退避，最多 3 次）
    # 工具失败不应打崩整条任务链路：返回错误文本，交给模型决定下一步
    try:
        return _search_with_retry(
            query=query,
            topic=topic,
            max_results=max_results,
            include_raw_content=include_raw_content,
        )
    except Exception as e:
        return f"网络搜索失败：{type(e).__name__}: {e}"
