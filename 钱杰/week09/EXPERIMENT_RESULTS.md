# vLLM 部署与约束解码实验报告

## 一、实验环境

| 项目 | 配置 |
|------|------|
| GPU | NVIDIA GeForce RTX 2060（6GB 显存） |
| NVIDIA 驱动 | 560.94（CUDA 12.6） |
| CPU | Intel Core i7-10750H @ 2.60GHz |
| 内存 | 16GB DDR4 |
| 操作系统 | Windows 11 + WSL2 Ubuntu 22.04 |
| Python | 3.10+ |
| vLLM | 0.9.2 |
| PyTorch | 2.7.0+cu126 |
| Transformers | 4.52.4 |
| 模型 | Qwen2-0.5B-Instruct（约 1GB） |

---

## 二、实验一：吞吐量对比

### 2.1 实验目的
验证 vLLM 相对原生 transformers 的推理性能提升。

### 2.2 实验设置
- 测试样本：50 条长短混合的金融问答 prompt
- 生成长度：每条最多 100 token
- 三种模式对比：
  - [A] transformers 串行推理（一次一条）
  - [B] transformers 手动批处理（batch=8，padding 到最长）
  - [C] vLLM continuous batching（动态批处理）

### 2.3 实验结果

| 模式 | 总耗时 | QPS | Generation tok/s | 相对 vLLM |
|------|--------|-----|------------------|-----------|
| [A] transformers 串行 | 60.98s | 0.82 | 60 | 0.017× |
| [B] transformers batch=8 | 12.85s | 3.89 | 289 | 0.080× |
| [C] vLLM continuous batching | **1.03s** | **48.59** | **3394** | **1.00×** |

### 2.4 结果分析

**速度提升倍数：**
- vLLM 相对 transformers 串行加速：**59.3×**
- vLLM 相对 transformers batch=8 加速：**12.5×**

**关键机制：**

1. **PagedAttention（内存优化）**
   - 传统方式：KV cache 按连续内存分配，批处理必须 padding 到最长，导致大量显存浪费
   - vLLM 方式：KV cache 按 block（16 token）动态分配，消除 padding 浪费，显存利用率提高后能塞下更大的 batch size

2. **Continuous Batching（调度优化）**
   - 传统方式：等待 batch 内最长请求完成后才能处理新请求，GPU 长时间空闲
   - vLLM 方式：短请求完成后立即插入新请求，长请求不拖累短请求，GPU 利用率从 ~20% 提升到接近 100%

**实验截图示意：**

```
┌─────────────────────────────────────────────────────────────┐
│  总耗时对比（越低越好）                                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ████████████████████████████████████████  61s  [A]串行      │
│  ██████                                      13s  [B]batch   │
│  █                                            1s  [C]vLLM    │
│                                                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  QPS 对比（越高越好）                                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  █                                                  0.8 QPS │
│  ████                                               3.9 QPS │
│  ████████████████████████████████████████         48.6 QPS  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 三、实验二：约束解码效果对比（get_stock_quote）

### 3.1 实验目的
验证三种 JSON 输出方式在股票查询场景下的可靠性。

### 3.2 实验设置
- 工具：`get_stock_quote`（查询股票行情）
- 测试样本：50 条用户请求
- Schema 复杂度：含 string + enum + regex + array + minItems

### 3.3 实验结果

| 指标 | 裸 prompt | response_format | guided_json |
|------|----------|-----------------|-------------|
| JSON 语法合法 | 86% | 100% | 100% |
| 必选字段齐全 | 86% | 100% | 100% |
| **完整 schema 通过** | **60%** | **68%** | **100%** |
| 平均延迟（秒）| 0.43 | 0.41 | 0.43 |

### 3.4 典型失败案例

**Prompt**: "300750 宁德时代最高价"

```json
// 裸 prompt / response_format 输出（失败）
{
  "symbol": "300750",
  "market": "SH",              // 错误：3开头应为 SZ
  "date": "2026-05-12",
  "fields": ["最高价"],         // 错误：中文"最高价"不在枚举内
  "adjust": "none"
}

// guided_json 输出（成功）
{
  "symbol": "300750",
  "market": "SZ",              // 正确
  "date": "2026-05-12",
  "fields": ["high"],          // 正确：映射到合法枚举
  "adjust": "none"
}
```

---

## 四、实验三：约束解码效果对比（create_order）

### 4.1 实验目的
验证三种 JSON 输出方式在电商下单场景下的可靠性。

### 4.2 实验设置
- 工具：`create_order`（创建订单）
- 测试样本：50 条用户请求
- Schema 复杂度：含 integer 范围 + 手机号正则 + 多枚举

### 4.3 实验结果

| 指标 | 裸 prompt | response_format | guided_json |
|------|----------|-----------------|-------------|
| JSON 语法合法 | 96% | 100% | 100% |
| 必选字段齐全 | 96% | 100% | 100% |
| **完整 schema 通过** | **42%** | **42%** | **100%** |
| 平均延迟（秒）| 0.57 | 0.56 | 0.58 |

### 4.4 典型失败案例

**Prompt**: "订 5 本《三体》，联系人 13711112222，快递"

```json
// 裸 prompt / response_format 输出（失败）
{
  "product": "《三体》",
  "quantity": 5,
  "user_phone": "+13711112222",  // 错误：前缀"+"违反正则 ^1[3-9]\d{9}$
  "priority": "快递",              // 错误："快递"不在枚举内
  "payment_method": "alipay"
}

// guided_json 输出（成功）
{
  "product": "《三体》",
  "quantity": 5,
  "user_phone": "13711112222",   // 正确：去掉前缀"+"
  "priority": "normal",           // 正确：映射到合法枚举
  "payment_method": "alipay"
}
```

---

## 五、实验四：约束解码类型对比

### 5.1 枚举约束（guided_choice）

**场景**：金融问答意图路由

**结果**：
- 输出合法率：裸 prompt 83% → guided_choice **100%**
- 分类准确率：25%（模型本身能力限制）

**结论**：guided_choice 能保证输出在枚举范围内，但分类正确率取决于模型能力。

### 5.2 正则约束（guided_regex）

**场景**：日期标准化、股票代码抽取

**结果**：
- 日期格式合法率：裸 prompt ~80% → guided_regex **100%**
- 股票代码合法率：裸 prompt ~70% → guided_regex **100%**

**结论**：正则约束能彻底解决"模型说对但格式错"的问题。

### 5.3 JSON Schema 约束（guided_json）

**场景**：财报问答三元组抽取

**结果**：

| 指标 | 裸 prompt | response_format | guided_json |
|------|----------|-----------------|-------------|
| 合法 JSON | 78% | 100% | 100% |
| 字段齐全 | 78% | 100% | 100% |
| year 在范围 | 67% | 67% | 100% |
| metric 在枚举 | 56% | 56% | 100% |
| schema 完全通过 | 56% | 56% | 100% |

---

## 六、核心结论

### 6.1 性能结论
- **vLLM 相对 transformers 串行加速 59.3×**（48.6 QPS vs 0.82 QPS）
- **vLLM 相对 transformers batch=8 加速 12.5×**
- 核心机制：PagedAttention（消除内存浪费）+ Continuous Batching（提高 GPU 利用率）

### 6.2 约束解码结论
- **`response_format={"type":"json_object"}`**：只保证 JSON 语法合法，字段语义准确率不改善
- **`guided_json=schema`**：唯一能把完整 schema 通过率拉到 100% 的方式
- **约束解码几乎不增加延迟**（FSM 一次构建长期复用）

### 6.3 工程价值
| 场景 | 裸 prompt | guided_json | 提升 |
|------|----------|-------------|------|
| get_stock_quote | 60% | 100% | +40% |
| create_order | 42% | 100% | +58% |

**对于 Agent 系统**：0.5B 模型裸 prompt 42% 完整通过率意味着 58% 的工具调用会失败，几乎不可用；使用 guided_json 后 100% 通过，小模型从不可用变为可靠。

---

## 七、消融实验建议

| 消融维度 | 预期观察 |
|---------|---------|
| `enforce_eager` on/off | CUDA Graph 优化约 +20% QPS，但首次启动慢 5~10s |
| `gpu_memory_utilization` 0.3/0.6/0.9 | KV cache 越大，batch concurrency 越高 |
| 模型规模 0.5B/1.5B/3B | tok/s 下降但 guided_json schema 通过率上升 |
| prompt 长度变化 | 短请求占 batch slot 少，长请求放大 continuous batching 收益 |

---

## 八、部署与复现步骤

### 8.1 环境搭建（WSL2 Ubuntu 22.04）

```bash
# 安装依赖
sudo apt install python3-pip python3-venv build-essential

# 创建虚拟环境
python3 -m venv ~/vllm_env
source ~/vllm_env/bin/activate

# 安装 Python 包
pip install vllm==0.9.2 torch==2.7.0+cu126 transformers==4.52.4
pip install openai>=1.40.0 jsonschema>=4.20.0 matplotlib>=3.7.0
```

### 8.2 启动 vLLM Server

```bash
cd /path/to/vllm_deployment/src
bash start_server.sh
```

### 8.3 运行吞吐量测试

```bash
# 先停掉 server
fuser -k 8000/tcp

# 运行 benchmark
python bench_throughput.py

# 重新启动 server
bash start_server.sh
```

### 8.4 运行约束解码 demo

```bash
python demo_guided_choice.py
python demo_guided_regex.py
python demo_guided_json.py
python demo_response_format.py
python demo_function_call.py
```

---

## 九、关键工程决策与踩坑

| 问题 | 根因 | 解法 |
|------|------|------|
| `torch.cuda.is_available()` 返回 False | 安装了 CUDA 13 版本的 torch（需驱动 580+） | 降级 `pip install vllm==0.9.2` |
| `aimv2 is already used by a Transformers config` | transformers 5.x 与 vLLM 0.9.2 冲突 | `pip install transformers==4.52.4` |
| server 启动报内存不足 | 显存不够 | 降低 `gpu-memory-utilization`（如 0.6→0.4） |
| demo 脚本报 Connection refused | vLLM server 未启动 | 另开终端运行 `bash start_server.sh` |

---

## 十、优化方向

1. **数据层面**：构造更贴近业务的 function call 测试集，加入多轮对话场景
2. **模型层面**：换 Qwen2.5-1.5B/3B-AWQ，观察 schema 通过率是否继续上升
3. **解码层面**：`guided_grammar` EBNF 构造更复杂的命令解析
4. **工程层面**：接入 Prometheus 监控，集成到 MCP Server
