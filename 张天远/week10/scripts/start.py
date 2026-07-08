#!/usr/bin/env python
"""
一键启动 RAG 系统 Web 界面

同时启动：
  - FastAPI 后端 (端口 8000)
  - StreamLit 前端 (端口 8501)

用法: python scripts/start.py
      python scripts/start.py --backend-only
      python scripts/start.py --frontend-only
"""

import subprocess
import sys
import time
import atexit
import os
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
PYTHON = "S:/condaEnvs/py312/python.exe"
ENV = os.environ.copy()
ENV["PYTHONPATH"] = "S:/condaEnvs/py312/Lib/site-packages"
ENV["HF_HOME"] = "M:/huggingface_cache"
ENV["HF_HUB_OFFLINE"] = "1"

_backend_proc = None


def start_backend(port: int = 8000) -> subprocess.Popen:
    """启动 FastAPI 后端。"""
    print(f"🔧 启动后端 (FastAPI, 端口 {port})...")
    proc = subprocess.Popen(
        [PYTHON, "scripts/app.py", "--port", str(port), "--host", "127.0.0.1"],
        cwd=str(PROJECT_DIR),
        env=ENV,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    # 等待后端就绪
    import requests
    for _ in range(30):
        time.sleep(1)
        try:
            r = requests.get(f"http://127.0.0.1:{port}/health", timeout=2)
            if r.status_code == 200:
                print(f"✅ 后端就绪: http://127.0.0.1:{port}")
                return proc
        except Exception:
            pass
    print("⚠️ 后端启动超时，请检查日志")
    return proc


def start_frontend(port: int = 8501, backend_port: int = 8000):
    """启动 StreamLit 前端。"""
    print(f"🎨 启动前端 (StreamLit, 端口 {port})...")
    print(f"   打开 http://localhost:{port}")
    print(f"   Ctrl+C 停止全部服务\n")
    subprocess.run(
        [PYTHON, "-m", "streamlit", "run", "web/app.py",
         "--server.port", str(port),
         "--server.address", "127.0.0.1",
         "--browser.serverAddress", "localhost",
         "--theme.base", "dark"],
        cwd=str(PROJECT_DIR),
        env=ENV,
    )


def cleanup():
    global _backend_proc
    if _backend_proc and _backend_proc.poll() is None:
        print("\n🛑 关闭后端...")
        _backend_proc.terminate()
        _backend_proc.wait(timeout=5)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="一键启动 RAG Web")
    parser.add_argument("--backend-only", action="store_true")
    parser.add_argument("--frontend-only", action="store_true")
    parser.add_argument("--backend-port", type=int, default=8000)
    parser.add_argument("--frontend-port", type=int, default=8501)
    args = parser.parse_args()

    global _backend_proc
    atexit.register(cleanup)

    backend_only = args.backend_only
    frontend_only = args.frontend_only

    if not frontend_only:
        _backend_proc = start_backend(port=args.backend_port)
        if backend_only:
            print("后端运行中，Ctrl+C 停止")
            try:
                while _backend_proc.poll() is None:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
            return

    if not backend_only:
        start_frontend(port=args.frontend_port, backend_port=args.backend_port)


if __name__ == "__main__":
    main()
