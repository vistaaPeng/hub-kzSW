"""
weather_backend.py — 天气查询后端（三种方式共享的业务逻辑）

教学重点：
  1. 同样是"纯业务逻辑"，与 rag_backend 平级，被三种方式复用
  2. 内部两次 HTTP 请求：Geocoding（城市名→经纬度）+ 天气查询
  3. 错误处理返回可读字符串而非抛异常，方便 LLM 直接消费

使用方式（作为模块）：
  from src.weather_backend import get_weather
  print(get_weather("宁德"))

依赖：
  pip install httpx
  Open-Meteo API 完全免费，无需注册
"""

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


def get_coordinates(city: str) -> str:
    """
    根据城市名称获取经纬度信息。

    Args:
        city: 城市名称，支持中文，例如 "宁德"、"北京"、"上海"

    Returns:
        包含经度、纬度、城市名、国家和省份的字符串，格式为 JSON 风格
    """
    with httpx.Client(timeout=10.0) as client:
        def _geocode(name: str):
            resp = client.get(GEOCODING_URL, params={
                "name": name, "count": 10, "language": "zh", "format": "json",
            })
            resp.raise_for_status()
            return resp.json().get("results") or []

        results = _geocode(city)
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

        def _rank(r):
            fc = str(r.get("feature_code", ""))
            admin_priority = 1 if fc.startswith("PPLA") or fc.startswith("ADM") else 0
            pop = r.get("population") or 0
            return (admin_priority, pop)

        loc = max(results, key=_rank)
        lat = loc["latitude"]
        lon = loc["longitude"]
        city_name = loc.get("name", city)
        country = loc.get("country", "")
        admin1 = loc.get("admin1", "")

        return (
            f"城市：{city_name}\n"
            f"国家：{country}\n"
            f"省份：{admin1}\n"
            f"纬度：{lat}\n"
            f"经度：{lon}"
        )


def get_weather_by_coords(lat: float, lon: float) -> str:
    """
    根据经纬度查询天气。

    Args:
        lat: 纬度，例如 26.64
        lon: 经度，例如 119.31

    Returns:
        包含温度、湿度、风速、天气状况和3天预报的文字描述
    """
    import time

    timeout = httpx.Timeout(30.0, connect=15.0)
    max_retries = 3

    for attempt in range(max_retries):
        with httpx.Client(timeout=timeout, verify=False) as client:
            try:
                weather_resp = client.get(WEATHER_URL, params={
                    "latitude": lat,
                    "longitude": lon,
                    "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
                    "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code",
                    "timezone": "Asia/Shanghai",
                    "forecast_days": 3,
                })
                weather_resp.raise_for_status()
            except httpx.RequestError as e:
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                return f"天气数据获取失败：{e}"

            data = weather_resp.json()
            cur = data["current"]
            daily = data["daily"]

            weather_desc = WEATHER_CODE_MAP.get(cur["weather_code"], f"代码{cur['weather_code']}")

            lines = [
                f"【坐标 {lat:.2f}°N, {lon:.2f}°E】天气报告",
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


def get_weather(city: str) -> str:
    """
    查询指定城市的当前天气及未来3天预报（链式调用封装）。

    Args:
        city: 城市名称，支持中文，例如 "宁德"、"北京"、"上海"

    Returns:
        包含温度、湿度、风速、天气状况和3天预报的文字描述
    """
    coords_str = get_coordinates(city)
    if "未找到" in coords_str:
        return coords_str

    import re
    lat_match = re.search(r"纬度：([\d.]+)", coords_str)
    lon_match = re.search(r"经度：([\d.]+)", coords_str)

    if not lat_match or not lon_match:
        return f"无法解析坐标信息：{coords_str}"

    lat = float(lat_match.group(1))
    lon = float(lon_match.group(1))
    return get_weather_by_coords(lat, lon)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", required=True)
    args = parser.parse_args()
    print(get_weather(args.city))
