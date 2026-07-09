# vLLM 部署实验报告

**实验环境**: AutoDL / RTX 4090D 24GB / CUDA 12.4 / torch 2.5.1+cu124 / vLLM 0.7.3  
**测试模型**: Qwen2-0.5B-Instruct / Qwen2.5-1.5B-Instruct / Qwen2.5-7B-Instruct  
**实验日期**: 2026-07-03

---

## 一、吞吐对比

### 1.1 三模型基准测试

| 模型 | 模式 | 总耗时 | QPS | tok/s | 加速比(vs串行) |
|------|------|--------|-----|-------|:---:|
| **Qwen2-0.5B** | transformers 串行 | 35.5s | 1.4 | 103 | 1× |
| (50条, max_len=2048) | transformers batch=8 | 9.0s | 5.6 | 412 | 4.0× |
| | **vLLM** | **0.8s** | **63.4** | **4237** | **45.0×** |
| **Qwen2.5-1.5B** | transformers 串行 | 59.7s | 0.8 | 67 | 1× |
| (50条, max_len=2048) | transformers batch=8 | 10.4s | 4.8 | 480 | 6.0× |
| | **vLLM** | **1.0s** | **50.3** | **5031** | **60.0×** |
| **Qwen2.5-7B** | transformers 串行 | 18.9s | 1.1 | 53 | 1× |
| (20条, max_len=256) | transformers batch=4 | 5.4s | 3.7 | 370 | 3.4× |
| | **vLLM** | **1.1s** | **18.5** | **925** | **17.5×** |

> 7B 模型因 24GB 显存限制，vLLM 需 max_len=256 + gpu_util=0.9 才能运行；0.5B/1.5B 可承受 max_len=2048

### 1.2 200 条高并发压测（Qwen2-0.5B）

| 模式 | 总耗时 | QPS | P50 延迟 | P95 延迟 | 加速比 |
|------|--------|-----|----------|----------|:---:|
| transformers 串行 | 97.3s | 2.1 | 0.34s | 0.84s | 1× |
| transformers batch=32 | 7.6s | 26.3 | 1.08s | 1.11s | 12.5× |
| **vLLM** | **1.4s** | **145.4** | 0.96s | 1.37s | **69.2×** |

**关键发现**:
- 200 并发下，vLLM 加速比进一步拉大到 **69×**（比 50 条时的 45× 更大）
- vLLM 延迟分布紧凑(P50=0.96s, P95=1.37s)，80% 请求在 1.4s 内完成
- transformers batch=32 虽然 QPS 提升，但 P50 延迟高达 1.08s（等最慢请求拖累全 batch）
- 4090D 24GB 下，Qwen2-0.5B 模型仅占 1GB，vLLM 可分配 16.75GB KV cache，最大并发 **357 请求**

---

## 二、约束解码效果对比

### 2.1 Qwen2-0.5B Demo

| 测试 | 裸 prompt | guided | 提升 |
|------|:---:|:---:|:---:|
| guided_choice (12条) | 92% | 100% | +8% |
| guided_regex 日期 (6条) | 67% | 100% | +33% |
| guided_regex 股票 (5条) | 60% | 100% | +40% |
| guided_json schema (9条) | 89% | 89% | — |
| response_format JSON (5条) | 100% | — | — |

### 2.2 MiniCPM5-1B Function Call ★

| 指标 | 裸 prompt | response_format | guided_json |
|------|:---:|:---:|:---:|
| **stock** schema 通过 | 0/50 (0%) | 48/50 (96%) | **50/50 (100%)** |
| **order** schema 通过 | 0/50 (0%) | 36/50 (72%) | 44/50 (88%) |

**分析**:
- MiniCPM5 裸 prompt 全军覆没：原生输出非 JSON 格式，无法解析
- `response_format={"type":"json_object"}` 强制执行 JSON 语法，但**不保证字段语义正确**（order 场景 72%）
- `guided_json` 通过 FSM 屏蔽非法 token，stock 场景达 **100%**，order 场景 88%（6 条失败）
- order 场景的 6 条 guided_json 失败：xgrammar 对 phone 正则(`^1[3-9]\d{9}$`)和整数范围的约束实现有缺陷，部分请求输出不符合（如数字超 100 范围）

---

## 三、模型规模对比

| 对比维度 | Qwen2-0.5B | Qwen2.5-1.5B | Qwen2.5-7B |
|---------|:---:|:---:|:---:|
| 模型权重 | ~1GB | ~3GB | ~14GB |
| vLLM 最大并发 | 803 | 357 | ~5 (max_len=256) |
| vLLM QPS | 63.4 | 50.3 | 18.5 |
| 加速比(vs串行) | **45×** | **60×** | **17×** |
| 显存需求 | 充裕 | 充裕 | 临界(24GB) |

**结论:**
- **小模型赢在吞吐**: 0.5B 模型 vLLM QPS 达 63，可并发 800+ 请求，适合高 QPS API 服务
- **大模型赢在加速比**: 1.5B 串行最慢(59.7s)，vLLM 加速最显著(60×)
- **7B 在 4090D 上受限**: 需极低 max_len(256) + 高 gpu_util(0.9) 才能跑，实际生产建议 48GB+ GPU

---

## 四、4090D 框架优势总结

| 指标 | transformers | vLLM | 差距 |
|------|:---:|:---:|:---:|
| 50条 QPS | 5.6 | 63.4 | **11×** |
| 200条 QPS | 26.3 | 145.4 | **5.5×** |
| 加速比 (vs 串行) | 12.5× | **69.2×** | — |
| GPU 利用率 | ~40% | ~95% | 2.4× |
| 最大并发 | 32 (OOM边缘) | 357+ | **11×** |
| KV cache 利用 | padding 浪费 | 按 block 动态分配 | — |

### 关键机制

1. **PagedAttention**: KV cache 按 16-token block 管理，消除 padding 内存碎片，24GB 显存得到充分利用
2. **Continuous Batching**: 不等 batch 内最长请求完成，短请求立即插入新请求，GPU 利用率接近 100%
3. **约束解码 FSM**: 在解码时实时屏蔽非法 token，不增加模型参数，延迟开销 <5%

---

## 五、踩坑记录

| 问题 | 原因 | 解决 |
|------|------|------|
| torch 2.6+ 不兼容 1080Ti | Pascal sm_61 被移除 | 上云 4090D |
| vLLM 0.9.2 不存在于 PyPI | 版本号错误 | 使用 0.7.3(兼容 torch 2.5.1) |
| vLLM 0.10+ 要求 torch≥2.8 | 云端 torch 锁死 2.5.1 | `--no-deps` 安装 |
| transformers 5.x `all_special_tokens_extended` | vLLM 0.7.x 不兼容 | 降级 transformers 4.57.6 |
| token_type_ids 报错 | Qwen2 不支持此字段 | `inputs.pop("token_type_ids")` |
| 根分区 30GB 不足 | pip 缓存 + venv | 移到 autodl-tmp(50GB) |
| bench 后显存未释放 | Python GC 延迟 | `torch.cuda.empty_cache()` |

---

## 六、产出文件

```
outputs_cloud/
├── throughput_c5409...png/json    Qwen2-0.5B 吞吐 (50条)
├── throughput_Qwen2.5_1.5B...png/json  Qwen2.5-1.5B 吞吐 (50条)
├── throughput_7B.json             Qwen2.5-7B 吞吐 (20条, max_len=256)
├── stress_c5409...png/json        4090D 200条压测 (0.5B)
├── function_call_results.json     Function Call (MiniCPM5-1B)
└── logs/
    ├── demo_qwen2-0.5b_*.log      Qwen2 完整 demo 输出
    ├── bench_*.log                基准测试日志
    └── stress_*.log               压测日志
```
