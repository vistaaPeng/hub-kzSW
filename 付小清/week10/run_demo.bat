@echo off
chcp 65001 >nul
echo ========================================
echo   医学科普 RAG 问答系统 - 演示
echo ========================================
echo.

if "%DASHSCOPE_API_KEY%"=="" (
    echo [错误] 请先设置 API Key:
    echo   set DASHSCOPE_API_KEY=sk-你的密钥
    echo.
    pause
    exit /b 1
)

set PYTHONIOENCODING=utf-8

echo [1/3] 感冒有哪些主要症状
python src/qa.py --query "感冒有哪些主要症状"
echo.

echo [2/3] 高血压的诊断标准是什么
python src/qa.py --query "高血压的诊断标准是什么"
echo.

echo [3/3] 突发剧烈胸痛伴随大汗应该怎么办
python src/qa.py --query "突发剧烈胸痛伴随大汗应该怎么办"
echo.

echo ========================================
echo   演示完成
echo ========================================
pause
