import requests
from typing import Annotated
from langchain_core.tools import tool
from api.monitor import monitor
from utils.retry import retry_with_backoff

@retry_with_backoff(max_retries=2, base_delay=1.0)
def _fetch(city: str):
    return requests.get(
        f"https://wttr.in/{city}?format=j1",
        timeout=15,
    ).json()

@tool
def get_weather(
    city: Annotated[str, "要查询天气的城市名，如 北京"],
):
    """
    查询指定城市的实时天气与未来天气概况。
    注意：只用于天气信息，不能查询市场、销售或知识库内容。
    """
    monitor.report_tool(tool_name="天气查询工具", args={"city": city})
    data = _fetch(city)
    cur = data["current_condition"][0]
    return (
        f"{city} 当前 {cur['temp_C']}°C，"
        f"体感 {cur['FeelsLikeC']}°C，"
        f"{cur['weatherDesc'][0]['value']}，湿度 {cur['humidity']}%"
    )