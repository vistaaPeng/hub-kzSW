# Week11 作业：天气查询循环工具调用

将 `function_call_mcp_cli` 中的天气工具调用从**单轮闭环**改造为 **ReAct 循环调用**。

## 快速开始

```powershell
cd work11
pip install -r requirements.txt
$env:DEEPSEEK_API_KEY = "sk-xxx"

python run_weather_loop.py --demo
```

## 文件说明

| 文件 | 说明 |
|------|------|
| `run_weather_loop.py` | 循环调用主程序 |
| `run_single_round.py` | 单轮对照组 |
| `作业提交说明.md` | 完整作业文档 |

天气后端复用 `../function_call_mcp_cli/src/weather_backend.py`。
