"""测试 vLLM OpenAI 兼容 API 是否可用"""
import sys
import requests

BASE = "http://localhost:8000/v1"

def main():
    print("查询模型列表...")
    r = requests.get(f"{BASE}/models", timeout=10)
    r.raise_for_status()
    print(r.json())

    print("\n发送对话请求...")
    r = requests.post(
        f"{BASE}/chat/completions",
        json={
            "model": "qwen2-0.5b",
            "messages": [{"role": "user", "content": "用一句话解释什么是 vLLM"}],
            "max_tokens": 80,
        },
        timeout=60,
    )
    r.raise_for_status()
    content = r.json()["choices"][0]["message"]["content"]
    print(f"回复: {content}")
    print("\n服务正常！")

if __name__ == "__main__":
    try:
        main()
    except requests.exceptions.ConnectionError:
        print("连接失败：请先启动 vLLM server（bash start_server.sh）")
        sys.exit(1)
