"""
weather_backend.py — 天气查询后端（拆分版）

改造要点（相比原 function_call_mcp_cli/src/weather_backend.py）：
  原来的 get_weather(city) 内部一次性做了两件事：
    ① Geocoding：城市名 → 经纬度
    ② 天气查询：经纬度 → 天气
  本文件把它拆成两个独立、单一职责的函数：
    · get_coordinates(city)                 —— 只做 ①，返回经纬度（JSON 字符串）
    · get_weather(latitude, longitude, ...) —— 只做 ②，输入经纬度返回天气
  拆开后，前一个函数的结果（经纬度）正好成为后一个函数的输入，
  从而可以被模型"链式调用"串起来（见 run_chain_function_call.py）。

设计约定：
  · get_coordinates 返回 JSON 字符串——因为它的结果要回填给模型，
    模型再从中取出 latitude/longitude 作为 get_weather 的入参，这是链条的接缝。
  · 错误一律返回可读字符串（而非抛异常），方便 LLM 直接消费。

依赖：
  pip install httpx
  Open-Meteo API 完全免费，无需注册。
"""

import json

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


# ── 工具一：城市名 → 经纬度 ──────────────────────────────────────────────────

def get_coordinates(city: str) -> str:
    """
    查询指定城市的经纬度坐标。

    Args:
        city: 城市名称，支持中文，例如 "北京"、"上海"、"天津"

    Returns:
        JSON 字符串，形如：
        {"city":"北京","name":"Beijing","latitude":39.90,"longitude":116.41,
         "admin1":"Beijing","country":"中国"}
        查询失败时返回 {"error": "..."} 的 JSON 字符串。
    """
    with httpx.Client(timeout=10.0) as client:
        # 中国地名常有歧义：裸"宁德"会命中西藏那曲市的一个村（PPL），
        # 而地级市宁德是 PPLA2。策略：先按用户输入查；若命中的只是低级行政点
        # （feature_code 纯 PPL）且用户没带"市/县/区"后缀，就用 city+"市" 重查一次并优先采用。
        def _geocode(name: str):
            resp = client.get(GEOCODING_URL, params={
                "name": name, "count": 10, "language": "zh", "format": "json",
            })
            resp.raise_for_status()
            return resp.json().get("results") or []

        try:
            results = _geocode(city)
        except httpx.RequestError as e:
            return json.dumps({"error": f"经纬度查询失败：{e}"}, ensure_ascii=False)

        is_low_admin = all(
            str(r.get("feature_code", "")).startswith("PPL")
            and not str(r.get("feature_code", "")).startswith("PPLA")
            for r in results
        ) if results else True
        has_suffix = any(city.endswith(s) for s in ("市", "县", "区", "镇"))
        if is_low_admin and not has_suffix:
            try:
                retry = _geocode(city + "市")
            except httpx.RequestError:
                retry = []
            if retry:
                results = retry

        if not results:
            return json.dumps(
                {"error": f"未找到城市 '{city}'，请尝试其他写法（如'宁德市'改'宁德'）"},
                ensure_ascii=False,
            )

        # 在候选里优先取行政级别更高的（PPLA/ADM = 某级政府驻地），
        # 其次取有人口数据的，避免落到同名小村庄。
        def _rank(r):
            fc = str(r.get("feature_code", ""))
            admin_priority = 1 if fc.startswith("PPLA") or fc.startswith("ADM") else 0
            pop = r.get("population") or 0
            return (admin_priority, pop)

        loc = max(results, key=_rank)
        return json.dumps({
            "city": city,
            "name": loc.get("name", city),
            "latitude": round(loc["latitude"], 4),
            "longitude": round(loc["longitude"], 4),
            "admin1": loc.get("admin1", ""),   # 省/州级行政区
            "country": loc.get("country", ""),
        }, ensure_ascii=False)


# ── 工具二：经纬度 → 天气 ────────────────────────────────────────────────────

def get_weather(latitude: float, longitude: float, name: str = "") -> str:
    """
    根据经纬度查询当前天气及未来3天预报。

    Args:
        latitude:  纬度，例如 39.90（可由 get_coordinates 得到）
        longitude: 经度，例如 116.41
        name:      可选，地点显示名称，仅用于报告标题美化

    Returns:
        包含温度、湿度、风速、天气状况和3天预报的文字描述。
    """
    with httpx.Client(timeout=10.0) as client:
        try:
            weather_resp = client.get(WEATHER_URL, params={
                "latitude": latitude,
                "longitude": longitude,
                "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code",
                "timezone": "Asia/Shanghai",
                "forecast_days": 3,
            })
            weather_resp.raise_for_status()
        except httpx.RequestError as e:
            return f"天气数据获取失败：{e}"

        data = weather_resp.json()
        cur = data["current"]
        daily = data["daily"]

        weather_desc = WEATHER_CODE_MAP.get(cur["weather_code"], f"代码{cur['weather_code']}")
        location_str = name or f"{latitude:.2f}°N, {longitude:.2f}°E"

        lines = [
            f"【{location_str}】天气报告",
            f"坐标：{latitude:.2f}°N, {longitude:.2f}°E",
            "",
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


if __name__ == "__main__":
    # 手动串一遍，验证"经纬度 → 天气"的链条能跑通
    import argparse
    import sys

    # 强制 UTF-8 输出，避免中文 Windows 控制台（GBK 码页）显示乱码
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="拆分版天气后端：先查经纬度，再查天气")
    parser.add_argument("--city", default="北京")
    args = parser.parse_args()

    coord_json = get_coordinates(args.city)
    print("① get_coordinates →", coord_json)
    coord = json.loads(coord_json)
    if "error" in coord:
        raise SystemExit(coord["error"])
    print("② get_weather →")
    print(get_weather(coord["latitude"], coord["longitude"], coord["name"]))
