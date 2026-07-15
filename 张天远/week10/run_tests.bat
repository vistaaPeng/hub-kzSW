@echo off
REM 统一测试入口 —— 自动处理 Hermes venv path 污染
if "%RAG_PYTHONPATH%"=="" set RAG_PYTHONPATH=S:/condaEnvs/py312/Lib/site-packages
if "%RAG_PYTHON%"=="" set RAG_PYTHON=S:\condaEnvs\py312\python.exe
set PYTHONPATH=%RAG_PYTHONPATH%
cd /d E:\npl\workspaces\npl_tran\rag_scratch
"%RAG_PYTHON%" -m pytest tests/ -v %*
