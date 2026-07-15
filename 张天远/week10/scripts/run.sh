#!/usr/bin/env bash
# 在 conda py312 环境中运行 Python 命令，隔离 Hermes venv 的 path 污染。
#
# 用法: ./scripts/run.sh pytest tests/ -v
#       ./scripts/run.sh python src/downloader.py
#
# 等价于: PYTHONPATH=... S:/condaEnvs/py312/python.exe <args>

set -e

CONDA_SP="S:/condaEnvs/py312/Lib/site-packages"
CONDA_PYTHON="S:/condaEnvs/py312/python.exe"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

cd "$PROJECT_DIR"
export PYTHONPATH="$CONDA_SP"
exec "$CONDA_PYTHON" "$@"
