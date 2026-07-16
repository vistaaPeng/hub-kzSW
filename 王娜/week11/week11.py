"""
weather_backend.py — 天气查询后端（三种方式共享的业务逻辑）

教学重点：
  1. 同样是"纯业务逻辑"，与 rag_backend 平级，被三种方式复用
  2. Agent Loop 形式：将"城市名→经纬度"和"经纬度→天气"拆成两个可独立调用的工具，
     支持单查坐标、单查天气（自动链式调用坐标）、多城市批量查询
  3. 错误处理返回可读字符串而非抛异常，方便 LLM 直接消费

使用方式（作为模块）：
  from src.weather_backend import get_weather, get_coordinates, run_weather_agent
  print(get_coordinates("北京"))           # 仅坐标
  print(get_weather("北京"))               # 单城市天气
  print(run_weather_agent("查询北京、上海、天津的坐标和天气"))

命令行：
  python src/weather_backend.py --coordinates 北京
  python src/weather_backend.py --city 北京
  python src/weather_backend.py --query "查询北京、上海、天津的坐标和天气"

依赖：
  pip install httpx
  Open-Meteo API 完全免费，无需注册
"""

import re
import httpx

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

# Open-Meteo 天气代码 → 中文描述映射
WEATHER_CODE_MAP = {
    0: "晴天", 1: "大致晴朗", 2: "局部多云", 3: "阴天",
    45: "雾", 48: "冻雾",
    51: "小毛毛雨", 53: "中毛毛雨", 55: "大毛毛雨",
    61: "小雨", 63: "中雨", 65: "大雨",
    71: "小雪", 73: "中雪", 75: "大雪",
    80: "小阵雨", 81: "中阵雨", 82: "大阵雨",
    95: "雷暴", 96: "雷暴伴小冰雹", 99: "雷暴伴大冰雹",
}


def _http_client():
    return httpx.Client(timeout=10.0)


# ── 工具 1：Geocoding（城市名 → 经纬度）─────────────────────────────────────

def get_coordinates(city: str) -> dict | str:
    """
    查询城市经纬度。

    Args:
        city: 城市名称，支持中文，例如 "宁德"、"北京"、"上海"

    Returns:
        成功返回 dict：{"name", "latitude", "longitude", "country", "admin1"}
        失败返回可读错误字符串
    """
    with _http_client() as client:
        def _geocode(name: str):
            resp = client.get(GEOCODING_URL, params={
                "name": name, "count": 10, "language": "zh", "format": "json",
            })
            resp.raise_for_status()
            return resp.json().get("results") or []

        results = _geocode(city)
        # 中国地名常有歧义：裸"宁德"会命中西藏那曲市的一个村（PPL），
        # 而宁德时代总部所在的福建宁德是地级市"宁德市"（PPLA2）。
        # 策略：先按用户输入查；若命中的只是低级行政点（feature_code 纯 PPL），
        # 且用户没带"市/县/区"后缀，就用 city+"市" 重查一次并优先采用。
        is_low_admin = all(
            str(r.get("feature_code", "")).startswith("PPL")
            and not str(r.get("feature_code", "")).startswith("PPLA")
            for r in results
        ) if results else True
        has_suffix = any(city.endswith(s) for s in ("市", "县", "区", "镇"))
        if is_low_admin and not has_suffix:
            retry = _geocode(city + "市")
            if retry:
                results = retry

        if not results:
            return f"未找到城市 '{city}'，请尝试其他写法（如'宁德市'改'宁德'）"

        # 在候选里优先取行政级别更高的（feature_code 含 A = 某级政府驻地），
        # 其次取有人口数据的，避免落到同名小村庄
        def _rank(r):
            fc = str(r.get("feature_code", ""))
            admin_priority = 1 if fc.startswith("PPLA") or fc.startswith("ADM") else 0
            pop = r.get("population") or 0
            return (admin_priority, pop)

        loc = max(results, key=_rank)
        return {
            "name": loc.get("name", city),
            "latitude": loc["latitude"],
            "longitude": loc["longitude"],
            "country": loc.get("country", ""),
            "admin1": loc.get("admin1", ""),
        }


# ── 工具 2：Weather（经纬度 → 天气）──────────────────────────────────────────

def _fetch_weather(lat: float, lon: float) -> dict:
    """调用 Open-Meteo 获取原始天气数据。"""
    with _http_client() as client:
        resp = client.get(WEATHER_URL, params={
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code",
            "timezone": "Asia/Shanghai",
            "forecast_days": 3,
        })
        resp.raise_for_status()
        return resp.json()


def _format_weather(data: dict) -> str:
    """将原始天气数据格式化为可读字符串。"""
    cur = data["current"]
    daily = data["daily"]
    weather_desc = WEATHER_CODE_MAP.get(cur["weather_code"], f"代码{cur['weather_code']}")

    lines = [
        f"当前天气：{weather_desc}",
        f"  温度：{cur['temperature_2m']}°C",
        f"  相对湿度：{cur['relative_humidity_2m']}%",
        f"  风速：{cur['wind_speed_10m']} km/h",
        "",
        "未来3天预报：",
    ]
    for i in range(3):
        day_desc = WEATHER_CODE_MAP.get(daily["weather_code"][i], "")
        lines.append(
            f"  {daily['time'][i]}：{day_desc}，"
            f"{daily['temperature_2m_max'][i]}°C / {daily['temperature_2m_min'][i]}°C，"
            f"降水 {daily['precipitation_sum'][i]} mm"
        )
    return "\n".join(lines)


def get_weather_by_coordinates(lat: float, lon: float, city_name: str = "") -> str:
    """
    根据经纬度查询天气。

    Args:
        lat: 纬度
        lon: 经度
        city_name: 城市名（仅用于展示）

    Returns:
        包含当前天气和未来3天预报的文字描述
    """
    try:
        data = _fetch_weather(lat, lon)
    except httpx.RequestError as e:
        return f"天气数据获取失败：{e}"
    return _format_weather(data)


# ── 兼容旧接口：城市名 → 天气（内部自动链式调用坐标）─────────────────────────

def get_weather(city: str) -> str:
    """
    查询指定城市的当前天气及未来3天预报（保持原有接口）。

    Args:
        city: 城市名称，支持中文

    Returns:
        包含坐标、温度、湿度、风速、天气状况和3天预报的文字描述
    """
    coords = get_coordinates(city)
    if isinstance(coords, str):
        return coords

    try:
        data = _fetch_weather(coords["latitude"], coords["longitude"])
    except httpx.RequestError as e:
        return f"天气数据获取失败：{e}"

    location_str = f"{coords['country']} {coords['admin1']} {coords['name']}".strip()
    lines = [
        f"【{location_str}】天气报告",
        f"坐标：{coords['latitude']:.2f}°N, {coords['longitude']:.2f}°E",
        "",
    ]
    lines.append(_format_weather(data))
    return "\n".join(lines)


# ── Agent Loop：自然语言查询 → 多城市坐标/天气 ───────────────────────────────

def _extract_cities(query: str) -> list[str]:
    """从自然语言查询中提取城市名列表。"""
    cleaned = query
    remove_phrases = ["查询", "查一下", "帮我查", "告诉我", "的坐标", "和天气", "天气", "坐标", "一下"]
    for phrase in remove_phrases:
        cleaned = cleaned.replace(phrase, "")

    parts = re.split(r"[、，,和；;]+", cleaned)
    cities = []
    for p in parts:
        p = p.strip()
        p = re.sub(r"^(查|查询)", "", p)
        if p:
            cities.append(p)
    return cities


def _need_weather(query: str) -> bool:
    """判断用户是否需要天气信息。"""
    return "天气" in query


def _format_location(coords: dict) -> str:
    """格式化坐标信息。"""
    return f"【{coords['name']}】坐标：{coords['latitude']:.2f}°N, {coords['longitude']:.2f}°E"


def run_weather_agent(query: str) -> str:
    """
    Agent Loop：解析用户查询，批量查询城市坐标，按需链式查询天气。

    支持：
      - "查询北京、上海、天津的坐标和天气" → 坐标 + 天气
      - "查询北京、上海、天津的坐标"       → 仅坐标
      - "查北京天气"                       → 单城市天气

    实现：
      1. 解析出城市列表与意图（是否需要天气）
      2. 对每个城市生成两步计划：geocode → [weather]
      3. 按顺序执行计划，weather 步骤依赖 geocode 步骤的结果
      4. 汇总输出

    Args:
        query: 自然语言查询

    Returns:
        包含各城市坐标和（或）天气的文字描述
    """
    cities = _extract_cities(query)
    if not cities:
        return "未从查询中识别出城市名，请尝试类似：'查询北京的坐标和天气'"

    need_weather = _need_weather(query)

    # Step 1: 生成计划
    plan = []
    for city in cities:
        plan.append({"tool": "geocode", "args": {"city": city}, "result": None})
        if need_weather:
            plan.append({
                "tool": "weather",
                "args": {"lat": None, "lon": None, "city_name": None},
                "result": None,
            })

    # Step 2: Agent Loop 执行计划
    for i, step in enumerate(plan):
        if step["tool"] == "geocode":
            step["result"] = get_coordinates(step["args"]["city"])
        elif step["tool"] == "weather":
            prev = plan[i - 1]["result"]
            if isinstance(prev, str):
                # 上一步 geocode 失败，跳过天气查询
                step["result"] = "无法查询天气：坐标获取失败"
            else:
                step["result"] = get_weather_by_coordinates(
                    prev["latitude"], prev["longitude"], prev["name"]
                )

    # Step 3: 格式化输出
    reports = []
    i = 0
    while i < len(plan):
        step = plan[i]
        if step["tool"] == "geocode":
            coords = step["result"]
            if isinstance(coords, str):
                reports.append(f"【{step['args']['city']}】{coords}")
                i += 1
                continue

            if need_weather and i + 1 < len(plan):
                weather_report = plan[i + 1]["result"]
                reports.append(_format_location(coords) + "\n\n" + weather_report)
                i += 2
            else:
                reports.append(_format_location(coords))
                i += 1
        else:
            i += 1

    return "\n\n".join(reports)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--city", help="查询单个城市天气")
    group.add_argument("--query", help="Agent Loop 查询，如 '查询北京、上海、天津的坐标和天气'")
    group.add_argument("--coordinates", help="查询单个城市坐标")
    args = parser.parse_args()

    if args.city:
        print(get_weather(args.city))
    elif args.coordinates:
        result = get_coordinates(args.coordinates)
        if isinstance(result, dict):
            print(f"【{result['name']}】坐标：{result['latitude']:.2f}°N, {result['longitude']:.2f}°E")
        else:
            print(result)
    else:
        print(run_weather_agent(args.query))
