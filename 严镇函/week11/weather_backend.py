"""
weather_backend.py — 天气查询后端（核心业务逻辑）

功能：
  1. 通过 Geocoding API 将城市名转换为经纬度
  2. 调用 Open-Meteo API 获取天气数据
  3. 格式化输出天气报告（当前天气 + 未来3天预报）

使用方式：
  from weather_backend import get_weather
  print(get_weather("北京"))

依赖：
  pip install httpx
  Open-Meteo API 完全免费，无需注册
"""

import ssl
import time
import httpx

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

WEATHER_CODE_MAP = {
    0: "晴天", 1: "大致晴朗", 2: "局部多云", 3: "阴天",
    45: "雾", 48: "冻雾",
    51: "小毛毛雨", 53: "中毛毛雨", 55: "大毛毛雨",
    61: "小雨", 63: "中雨", 65: "大雨",
    71: "小雪", 73: "中雪", 75: "大雪",
    80: "小阵雨", 81: "中阵雨", 82: "大阵雨",
    95: "雷暴", 96: "雷暴伴小冰雹", 99: "雷暴伴大冰雹",
}


def get_weather(city: str) -> str:
    """
    查询指定城市的当前天气及未来3天预报。

    Args:
        city: 城市名称，支持中文，例如 "北京"、"上海"、"宁德"

    Returns:
        包含温度、湿度、风速、天气状况和3天预报的文字描述
    """
    def _retry_with_backoff(func, max_retries=3, initial_delay=1):
        for attempt in range(max_retries):
            try:
                return func()
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = initial_delay * (2 ** attempt)
                    time.sleep(delay)
                    continue
                raise e

    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    try:
        with httpx.Client(timeout=20.0, verify=False, http2=False) as client:
            def _geocode(name: str):
                resp = _retry_with_backoff(lambda: client.get(GEOCODING_URL, params={
                    "name": name, "count": 10, "language": "zh", "format": "json",
                }))
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

            weather_resp = _retry_with_backoff(lambda: client.get(WEATHER_URL, params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code",
                "timezone": "Asia/Shanghai",
                "forecast_days": 3,
            }))
            weather_resp.raise_for_status()

            data = weather_resp.json()
            cur = data["current"]
            daily = data["daily"]

            weather_desc = WEATHER_CODE_MAP.get(cur["weather_code"], f"代码{cur['weather_code']}")
            location_str = f"{country} {admin1} {city_name}".strip()

            lines = [
                f"【{location_str}】天气报告",
                f"坐标：{lat:.2f}°N, {lon:.2f}°E",
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

    except httpx.RequestError as e:
        return f"天气数据获取失败：网络连接异常，请检查网络后重试"
    except Exception as e:
        return f"天气数据获取失败：{str(e)[:100]}"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", required=True)
    args = parser.parse_args()
    print(get_weather(args.city))