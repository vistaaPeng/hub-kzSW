# 第九周作业 — vLLM 大模型服务部署与速度提升验证

> 作业要求：尝试部署一个 vLLM 大模型服务，验证速度提升。
> 完整工程见 [`../week9 大模型应用补充知识/vllm_deployment/`](../week9%20大模型应用补充知识/vllm_deployment/)（代码、文档、原始数据均在其中）。

---

## 一、做了什么

1. **部署**：在 WSL2 (Ubuntu 22.04) + CUDA 12.7 环境下，用 vLLM 0.9.2 把
   `Qwen2-0.5B-Instruct` 一键拉起为 OpenAI 兼容 HTTP 服务
   （`src/start_server.sh`），支持 `/v1/chat/completions` 标准接口。
2. **验证速度提升**：用 `src/bench_throughput.py` 同一批 50 条 prompt、
   同一张 GPU（RTX 4060 Laptop 8GB），对比三条推理路线：
   - **[A]** transformers 原生 `.generate()`，逐条串行
   - **[B]** transformers 手动 batch=8（padding）
   - **[C]** vLLM（PagedAttention + continuous batching）

## 二、环境

| 项 | 值 |
|----|----|
| 基座模型 | Qwen2-0.5B-Instruct |
| 推理引擎 | vLLM 0.9.2 + torch 2.7.0+cu126 |
| GPU | NVIDIA RTX 4060 Laptop 8GB（WSL2） |
| 测试规模 | 50 条 prompt，每条 max_new_tokens=100 |

> vLLM 版本选 0.9.2 而非最新版：0.20+ 需要 CUDA 13 / 驱动 580+，
> 而常见笔记本驱动是 566.x（CUDA 12.7），0.9.2 是兼容面最广的稳定组合。

## 三、实测结果（原始数据见 `outputs/throughput_results.json`）

| 路线 | 总耗时 | QPS | Generation tok/s | 相对 vLLM |
|------|------:|----:|------------------:|----------:|
| [A] transformers 串行 | 62.84s | 0.80 | 57.9 | 0.018× |
| [B] transformers batch=8 | 13.11s | 3.81 | 283.1 | 0.087× |
| [C] **vLLM** | **1.15s** | **43.57** | **3042.9** | **1.00×** |

**结论：vLLM 相对 transformers 串行推理加速约 55×；相对手动 batch=8 加速约 11.4×。**

对应柱状图：`outputs/throughput_comparison.png`

### 为什么会有这么大差距

- **A → B**（4.8×）：简单批处理就有明显提速，但 prompt 长短不一，短的要
  padding 到最长，仍有大量算力浪费在 padding token 上。
- **B → C**（11.4×）：vLLM 的两个核心机制带来质变——
  1. **PagedAttention**：KV cache 按 block（而非连续内存）分配，消除
     padding 造成的显存浪费，同样显存能塞下更大 batch。
  2. **Continuous batching**：不等 batch 内最长请求跑完，短请求一结束立刻
     插入新请求排队的下一条，GPU 利用率从 ~20% 拉到接近满载。

## 四、如何复现

```bash
cd vllm_deployment/src
# 1) 跑吞吐对比（会临时加载 transformers + vLLM 两份模型，测完释放）
python bench_throughput.py

# 2) 启动 vLLM OpenAI 兼容服务，验证真实调用
bash start_server.sh
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen2-0.5b","messages":[{"role":"user","content":"你好"}],"max_tokens":50}'
```

## 五、工程文档索引

| 文档 | 内容 |
|------|------|
| `ARCHITECTURE.md` | 技术选型、整体流水线、实验结果、踩坑记录 |
| `USAGE_GUIDE.md` | 环境搭建（WSL2+CUDA+vLLM）、各脚本运行方法、FAQ |
| `RESUME_GUIDE.md` | 可量化数据汇总、简历写法、面试可能追问的技术细节 |
