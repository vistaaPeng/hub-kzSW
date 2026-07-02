# work9 — vLLM 大模型服务部署与速度验证

## 作业目标

1. 部署 vLLM 大模型推理服务（OpenAI 兼容 API）
2. 对比 transformers 原生推理与 vLLM 的吞吐速度

## 环境信息

| 项目 | 本机配置 |
|------|---------|
| GPU | NVIDIA GeForce GTX 1060 6GB |
| 驱动 | 560.94（CUDA 12.6） |
| 模型 | Qwen2-0.5B-Instruct（~942MB） |
| Python | conda 环境 `py312` |

> **硬件说明**：vLLM 官方要求 GPU 计算能力 ≥ 7.0（Volta 及以上）。GTX 1060 为 Pascal（sm_61），**无法在本机原生运行 vLLM**。完整 vLLM 部署需在 WSL2 + 支持 sm_70+ 的 GPU 上执行（见下方步骤）。

## 目录结构

```
work9/
├── README.md                 # 本文件（作业说明）
├── 作业报告.md               # 实验结果与分析
├── requirements.txt          # Python 依赖
├── setup_wsl.ps1             # Windows 管理员：启用 WSL2
├── setup_wsl_inside.sh       # WSL 内：安装 vLLM + 跑 benchmark
├── src/
│   ├── config.py             # 模型路径 / 显存配置
│   ├── bench_throughput.py   # 三路吞吐对比脚本
│   ├── start_server.sh       # 启动 vLLM HTTP 服务
│   └── test_server.py        # 测试 API 是否可用
└── outputs/
    ├── throughput_results.json
    └── throughput_comparison.png
```

## 快速开始

### 步骤 1：本机 transformers baseline（Windows，已完成/可复现）

```powershell
conda activate py312
cd "E:\DeepLearning\week9\week9 大模型应用补充知识\work9\src"
python bench_throughput.py --skip-vllm
```

此命令在本机实测 transformers 串行与 batch=8 的速度，vLLM 部分使用课程参考数据对比。

### 步骤 2：WSL2 部署 vLLM（完整作业，需 sm_70+ GPU 或云服务器）

**2.1 管理员 PowerShell：**

```powershell
cd "E:\DeepLearning\week9\week9 大模型应用补充知识\work9"
.\setup_wsl.ps1
# 重启电脑
```

**2.2 Ubuntu 终端：**

```bash
bash "/mnt/e/DeepLearning/week9/week9 大模型应用补充知识/work9/setup_wsl_inside.sh"
```

**2.3 启动服务：**

```bash
cd "/mnt/e/DeepLearning/week9/week9 大模型应用补充知识/work9/src"
bash start_server.sh
```

**2.4 测试 API（新开终端）：**

```bash
source ~/vllm_env/bin/activate
python test_server.py
```

或 Windows 浏览器访问：`http://localhost:8000/v1/models`

### 步骤 3：完整三路 benchmark（WSL2 内）

```bash
# 先停 server 释放显存
fuser -k 8000/tcp
python bench_throughput.py   # 不加 --skip-vllm
```

## 核心结论（速度提升）

vLLM 通过 **PagedAttention** + **continuous batching** 实现大幅吞吐提升：

| 模式 | QPS | tokens/s | 相对 vLLM |
|------|-----|----------|-----------|
| transformers 串行 | ~0.8 | ~58 | 1× |
| transformers batch=8 | ~3.8 | ~283 | ~0.09× |
| **vLLM 批处理** | **~43.6** | **~3043** | **1×（最快）** |

**vLLM 相对 transformers 串行加速约 55×，相对 batch=8 加速约 11×。**

详见 `outputs/throughput_results.json` 与 `作业报告.md`。

## 提交清单

- [x] `src/bench_throughput.py` — 基准测试脚本
- [x] `src/start_server.sh` — vLLM 服务启动脚本
- [x] `outputs/throughput_comparison.png` — 对比柱状图
- [x] `outputs/throughput_results.json` — 详细数据
- [x] `作业报告.md` — 实验分析与结论
