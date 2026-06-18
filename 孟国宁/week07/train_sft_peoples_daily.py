"""
LLM SFT（监督微调）训练脚本 — 基于 LoRA 高效微调 Qwen2-0.5B-Instruct 做人民日报 NER

═══════════════════════ 整体运行流程（含数据格式示例）═══════════════════════

  ┌────────────────────────────────────────────────────────────────────────┐
  │ 1. 解析命令行参数 (parse_args)                                          │
  │    - 微调模式: --full_ft (全量) vs 默认 (LoRA)                           │
  │    - 学习率:   全量=2e-5, LoRA=2e-4                                     │
  │    - 数据量:   --num_train 控制，默认用全部                               │
  │                                                                        │
  │    输入: 命令行参数                                                      │
  │      python train.py --epochs 3 --batch_size 4 --lr 2e-4               │
  │    输出: args 对象                                                      │
  │      args.epochs=3, args.batch_size=4, args.lr=2e-4                    │
  └───────────────────────────┬────────────────────────────────────────────┘
                              ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │ 2. 加载数据 (train.json / validation.json)                              │
  │    可选: --num_train 随机抽样部分训练数据                                 │
  │                                                                        │
  │    输入: JSON 文件                                                      │
  │      [                                                                 │
  │        {                                                               │
  │          "tokens":   ["厦", "门", "是", "美", "丽", "的"],              │
  │          "ner_tags": ["B-LOC","I-LOC","O","O","O","O"]                 │
  │        },                                                              │
  │        {                                                               │
  │          "tokens":   ["张", "三", "去", "北", "京"],                    │
  │          "ner_tags": ["B-PER","I-PER","O","B-LOC","I-LOC"]             │
  │        }                                                               │
  │      ]                                                                 │
  │    输出: Python 列表 train_raw（每个元素是一个 dict）                      │
  │      [{"tokens": [...], "ner_tags": [...]}, ...]                       │
  └───────────────────────────┬────────────────────────────────────────────┘
                              ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │ 3. 加载 Tokenizer (Qwen2-0.5B-Instruct)                                │
  │    - apply_chat_template: 构建 system+user 的对话格式                    │
  │    - 若无 pad_token，则用 eos_token 充当                                 │
  │                                                                        │
  │    输入: 模型路径字符串                                                  │
  │      "pretrain_models/Qwen2-0.5B-Instruct"                            │
  │    输出: tokenizer 对象                                                 │
  │      tokenizer.encode("厦门") → [89642, 91827]                         │
  │      tokenizer.eos_token_id   → 151643                                │
  │      tokenizer.pad_token_id   → 151643（用 eos 代替）                   │
  └───────────────────────────┬────────────────────────────────────────────┘
                              ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │ 4. 构建 Dataset & DataLoader                                            │
  │                                                                        │
  │    === 4a. record_to_target(): BIO → JSON 字符串 ===                    │
  │    输入:                                                                │
  │      {"tokens": ["厦","门","是","美","丽","的"],                         │
  │       "ner_tags": ["B-LOC","I-LOC","O","O","O","O"]}                   │
  │    输出:                                                                │
  │      '{"entities": [{"text": "厦门", "type": "LOC"}]}'                  │
  │                                                                        │
  │    === 4b. apply_chat_template(): 构建对话 prompt ===                    │
  │    输入: messages 列表                                                  │
  │      [                                                                 │
  │        {"role": "system",  "content": "你是一个命名实体识别助手..."},     │
  │        {"role": "user",    "content": "厦门是美丽的"}                    │
  │      ]                                                                 │
  │    输出: prompt_text 字符串                                             │
  │      "<|system|>\n你是一个命名实体识别助手...<|user|>\n厦门是美丽的<|assistant|>\n"
  │                                                                        │
  │    === 4c. encode(): 文本 → token id ===                               │
  │    输入: prompt_text = "<|system|>...厦门是美丽的...<|assistant|>"       │
  │    输出: prompt_ids = [151644, 8948, ..., 89642, 91827, ..., 151645]   │
  │           (约 50~80 个 token id)                                       │
  │                                                                        │
  │    === 4d. encode(target) + EOS → response_ids ===                     │
  │    输入: target = '{"entities": [{"text": "厦门", "type": "LOC"}]}'     │
  │    输出: response_ids = [90065, 91827, 10418, ..., 151643]             │
  │           (约 20~40 个 token id) + [151643] (EOS)                      │
  │                                                                        │
  │    === 4e. 拼接 + loss mask ===                                        │
  │    input_ids = prompt_ids + response_ids                               │
  │      [151644, 8948, ..., 89642, 91827, ..., 90065, 91827, ..., 151643]│
  │      ├────── prompt 部分 ──────────────┤├──── response 部分 ──────┤     │
  │                                                                        │
  │    labels = [-100, -100, ...] + response_ids                           │
  │      [-100, -100, ..., -100, -100, ..., 90065, 91827, ..., 151643]    │
  │      ├──── prompt 全部 -100 ──────────┤├──── response 真实 id ───┤     │
  │      （不算 loss）                        （要算 loss）                  │
  │                                                                        │
  │    === 4f. collate_fn: 把一个 batch 填充到相同长度 ===                    │
  │    输入: batch = [sample1(长度70), sample2(长度55), sample3(长度62)]    │
  │    输出: 三个 tensor，都填充到长度 70                                    │
  │      input_ids:      shape=(3, 70), 填充位置用 pad_token_id             │
  │      labels:         shape=(3, 70), 填充位置用 -100                     │
  │      attention_mask: shape=(3, 70), 真实token=1, 填充=0                 │
  └───────────────────────────┬────────────────────────────────────────────┘
                              ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │ 5. 加载模型 + 设置微调方式                                               │
  │    ├── --full_ft: 加载全参数 → 100% 可训练（495M 参数）                  │
  │    └── 默认 LoRA:                                                      │
  │         LoraConfig(r=8, alpha=16, target=q/k/v/o_proj)                 │
  │         get_peft_model() → 仅 ~0.22% 参数可训练（~1M 参数）             │
  │                                                                        │
  │    输入: 模型路径 + 配置                                                │
  │      LoraConfig(r=8, alpha=16,                                        │
  │                target_modules=["q_proj","k_proj","v_proj","o_proj"])   │
  │    输出: model 对象（已注入 LoRA 层）                                    │
  │      trainable params: 1,081,344 || all params: 495,168,512           │
  │      trainable%: 0.2184%                                              │
  └───────────────────────────┬────────────────────────────────────────────┘
                              ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │ 6. 训练循环 (epochs × batches)                                          │
  │    for epoch in 1..epochs:                                             │
  │      ┌── Train Phase ───────────────────────────────────────────────┐  │
  │      │ 输入: batch = {                                              │  │
  │      │   "input_ids":      (4, 70)  ← 4条句子，每条70个token         │  │
  │      │   "labels":         (4, 70)  ← 前50个=-100，后20个=真实id     │  │
  │      │   "attention_mask": (4, 70)  ← 真实token=1，填充=0            │  │
  │      │ }                                                            │  │
  │      │ 输出: loss = 2.3456（标量，越小越好）                          │  │
  │      │ 反向传播 → 更新参数                                           │  │
  │      └──────────────────────────────────────────────────────────────┘  │
  │      ┌── Val Phase ─────────────────────────────────────────────────┐  │
  │      │ 输入: 同 Train，但用验证集                                     │  │
  │      │ 输出: val_loss = 1.8923（不更新参数，只记录）                   │  │
  │      └──────────────────────────────────────────────────────────────┘  │
  │      保存最优 checkpoint (val_loss 最低时)                              │
  └───────────────────────────┬────────────────────────────────────────────┘
                              ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │ 7. 保存训练日志 + 输出路径提示                                            │
  │                                                                        │
  │    输出文件 1: 训练日志                                                  │
  │      logs/train_sft_peoples_daily.json                                 │
  │      [                                                                 │
  │        {"epoch":1, "train_loss":2.3456, "val_loss":1.8923, "elapsed":45},│
  │        {"epoch":2, "train_loss":1.2345, "val_loss":1.1234, "elapsed":43},│
  │        {"epoch":3, "train_loss":0.8901, "val_loss":0.9876, "elapsed":44} │
  │      ]                                                                 │
  │                                                                        │
  │    输出文件 2: 模型权重                                                  │
  │      outputs/sft_adapter_peoples_daily/                                │
  │        ├── adapter_config.json    ← LoRA 配置                          │
  │        ├── adapter_model.safetensors  ← LoRA 权重（~几MB）             │
  │        └── tokenizer.json         ← tokenizer 配置                     │
  │                                                                        │
  │    下一步: python evaluate_sft_peoples_daily.py 查看 entity F1          │
  └────────────────────────────────────────────────────────────────────────┘

教学重点：
  1. NER 的指令微调格式：输入是文本，输出是 JSON 实体列表
     与分类任务的区别：TARGET 是多 token 的结构化 JSON，而非单个类别名
  2. Loss masking：同样只在 JSON 输出部分计算 loss，prompt 全为 -100
  3. LoRA 高效微调：参数量约 0.22%，与全量微调的对比（--full_ft 开关）
  4. 生成式 NER vs 序列标注（BERT+CRF）：各自的优劣和适用场景

使用方式：
  python train_sft_peoples_daily.py                        # LoRA，全量训练数据（默认）
  python train_sft_peoples_daily.py --num_train 2000       # LoRA，2000 条快速演示
  python train_sft_peoples_daily.py --epochs 1             # 快速验证流程

  # 全量微调（需显存 ≥ 16GB）
  python train_sft_peoples_daily.py --full_ft --lr 2e-5

依赖：
  pip install torch transformers peft tqdm   # LoRA 模式
  pip install torch transformers tqdm        # 全量微调模式（不需要 peft）
"""

# ════════════════════════════════════════════════════════════════════════════════
# 导入区
# ════════════════════════════════════════════════════════════════════════════════
import os
import argparse        # 命令行参数解析
import json            # 读取 JSON 数据文件、序列化训练日志
import random          # 随机抽样训练数据、设置随机种子
import time            # 计算每个 epoch 的耗时
from pathlib import Path  # 跨平台路径操作

import torch
import torch.nn as nn  # 用于 clip_grad_norm_ 梯度裁剪
from torch.utils.data import Dataset, DataLoader  # PyTorch 数据加载框架
from torch.optim import AdamW  # AdamW 优化器（带权重衰减的 Adam）
from transformers import AutoTokenizer, AutoModelForCausalLM  # HuggingFace 模型加载
from tqdm import tqdm  # 训练/验证进度条

# macOS / 某些 Linux 环境下 OpenMP 库冲突的常见修复
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# 尝试导入 PEFT 库（LoRA 依赖），若未安装则标记为不可用
# - 全量微调 (--full_ft) 不需要此库
# - LoRA 微调必须有此库，否则运行时报错
try:
    from peft import get_peft_model, LoraConfig, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

# ════════════════════════════════════════════════════════════════════════════════
# 全局常量 — 路径与实体类型定义
# ════════════════════════════════════════════════════════════════════════════════
ROOT       = Path(__file__).parent.parent          # 项目根目录（src_llm 的上一级）
DATA_DIR   = ROOT / "data" / "peoples_daily"       # 人民日报 NER 数据存放目录
MODEL_PATH = ROOT / "pretrain_models" / "Qwen2-0.5B-Instruct"  # 预训练模型本地路径
OUTPUT_DIR = ROOT / "outputs"                      # 输出目录（checkpoint + 训练日志）

# 人民日报 NER 数据集定义的三种实体类型
# PER = Person（人名）、ORG = Organization（组织机构）、LOC = Location（地点）
ENTITY_TYPES = ["PER", "ORG", "LOC"]

# ════════════════════════════════════════════════════════════════════════════════
# SYSTEM_PROMPT: 告诉 LLM 它的任务是什么
# ════════════════════════════════════════════════════════════════════════════════
# 这段 prompt 会在每条训练数据的前面拼接，构成完整的对话上下文（system + user）。
# 关键设计：它严格约束了输出格式为 JSON，LLM 只需学会填充 entities 字段。
# 与 BERT+CRF 做序列标注不同，这里是"生成式 NER"——模型以自回归方式逐 token 生成 JSON。
SYSTEM_PROMPT = (
    "你是一个命名实体识别助手。从文本中识别命名实体，以 JSON 格式输出。\n"
    "实体类型（英文标识）：PER（人名）、ORG（组织机构）、LOC（地点）\n"
    '输出格式（严格遵守，不输出其他内容）：{"entities": [{"text": "实体文本", "type": "实体类型"}]}\n'
    '无实体时输出：{"entities": []}'
)


# ════════════════════════════════════════════════════════════════════════════════
# record_to_target: BIO 标注 → JSON 实体列表字符串
# ════════════════════════════════════════════════════════════════════════════════
# 将人民日报数据集中的 BIO 序列标注格式（B-PER, I-PER, O, ...）转换为
# SFT 训练所需的 target 字符串（JSON 格式的实体列表）。
#
# BIO 标注规则：
#   B-XXX = 实体 XXX 的 Begin（开始标记）
#   I-XXX = 实体 XXX 的 Inside（内部标记，必须与前面的 B 或 I 同类型才能续接）
#   O     = Outside（非实体 token）
#
# 示例：
#   输入: tokens=["厦","门","是","美","丽","的"]
#         ner_tags=["B-LOC","I-LOC","O","O","O","O"]
#   输出: '{"entities": [{"text": "厦门", "type": "LOC"}]}'
def record_to_target(record: dict) -> str:
    """把人民日报 BIO 格式转为 SFT 目标 JSON 字符串。
    输入：{"tokens": ["海", "钓", ...], "ner_tags": ["O", "O", ...]}
    输出：'{"entities": [{"text": "厦门", "type": "LOC"}, ...]}'
    """
    tokens = record["tokens"]
    ner_tags = record["ner_tags"]

    entities = []          # 最终收集到的所有实体
    current_entity = []    # 当前正在构建的实体的 token 列表
    current_type = None    # 当前实体的类型（PER/ORG/LOC）

    for token, tag in zip(tokens, ner_tags):
        if tag.startswith("B-"):
            # ── 遇到 B-XXX：开始一个新实体 ──
            # 先把之前积累的旧实体保存（如果有）
            if current_entity and current_type:
                entity_text = "".join(current_entity)
                entities.append({"text": entity_text, "type": current_type})

            # 开始记录新实体
            current_entity = [token]
            current_type = tag[2:]  # 去掉 "B-" 前缀，得到类型名（如 "PER"）

        elif tag.startswith("I-") and current_type == tag[2:]:
            # ── 遇到 I-XXX 且类型匹配：续接当前实体 ──
            # 只有当 I- 的类型与当前实体类型一致时才追加
            # 否则视为标注异常，走 else 分支重置
            current_entity.append(token)

        else:
            # ── 遇到 O 或类型不匹配的 I-：结束当前实体（如果有） ──
            if current_entity and current_type:
                entity_text = "".join(current_entity)
                entities.append({"text": entity_text, "type": current_type})

            # 重置状态，等待下一个 B-
            current_entity = []
            current_type = None

    # 循环结束后，别忘了保存最后一个实体（如果序列末尾就是实体）
    if current_entity and current_type:
        entity_text = "".join(current_entity)
        entities.append({"text": entity_text, "type": current_type})

    return json.dumps({"entities": entities}, ensure_ascii=False)


# ══════════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════════

class SFTDataset(Dataset):
    """
    把人民日报 NER 数据转换为 chat-format SFT 训练样本。

    与分类任务的关键区别：
      - 分类：TARGET = "科技"（1~2 个 token，极短）
      - NER：TARGET = '{"entities": [...]}' （20~150 个 token，结构化 JSON）

    Loss mask 结构：
      ┌──────────────────────────────────────────────────────────────┐
      │ <system>...<user>{text}<assistant>\n                         │  → -100
      │ {"entities": [{"text": "厦门", "type": "LOC"}]} <EOS>│  → 真实 id
      └──────────────────────────────────────────────────────────────┘
    """

    def __init__(self, data, tokenizer, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        target = record_to_target(item)
        text = "".join(item["tokens"])

        # ── Step 1：构建 prompt 文本（tokenize=False 兼容 transformers 5.x）──
        prompt_text = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": text},
            ],  # messages — 对话内容
            tokenize=False, # tokenize=False — 返回什么格式
            add_generation_prompt=True, # add_generation_prompt=True — 添加生成提示
        )
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)   # 是否在首尾添加特殊符号
        # 为什么这里用False，因为 apply_chat_template 已经处理好了特殊符号，prompt_text = "<|system|>\n你是一个助手
        # ── Step 2：response = JSON 字符串 + EOS ──────────────────────────────
        response_ids = (
            self.tokenizer.encode(target, add_special_tokens=False)
            + [self.tokenizer.eos_token_id]
        )

        # ── Step 3：拼接 + 截断 ───────────────────────────────────────────────
        input_ids = (prompt_ids + response_ids)[: self.max_length]

        # ── Step 4：loss mask：prompt 全 -100，只在 JSON 部分计算 loss ──────
        labels = ([-100] * len(prompt_ids) + response_ids)[: self.max_length]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels":    torch.tensor(labels,    dtype=torch.long),
        }


# ════════════════════════════════════════════════════════════════════════════════
# collate_fn: 将一个 batch 的样本填充到相同长度
# ════════════════════════════════════════════════════════════════════════════════
# DataLoader 默认的 collate 无法处理不同长度的 tensor，需要自定义填充逻辑。
# 三个张量的填充策略：
#   - input_ids:      用 pad_id（通常是 eos_token_id）填充
#   - labels:         用 -100 填充（PyTorch CrossEntropyLoss 会忽略 -100 的位置）
#   - attention_mask: 用 0 填充（0 = 不参与 attention 计算，1 = 真实 token）
def collate_fn(batch, pad_id):
    # 找出本 batch 中最长的序列长度，所有样本都填充到这个长度
    max_len = max(item["input_ids"].size(0) for item in batch)
    input_ids_list, labels_list, mask_list = [], [], []

    for item in batch:
        n   = item["input_ids"].size(0)       # 当前样本的实际长度
        pad = max_len - n                      # 需要填充的长度

        # input_ids: 真实 token + pad_id 填充
        input_ids_list.append(torch.cat([item["input_ids"],
                                         torch.full((pad,), pad_id, dtype=torch.long)]))
        # labels: 真实标签 + -100 填充（-100 的位置不计算 loss）
        labels_list.append(torch.cat([item["labels"],
                                      torch.full((pad,), -100, dtype=torch.long)]))
        # attention_mask: 真实 token=1，填充=0
        mask_list.append(torch.cat([torch.ones(n, dtype=torch.long),
                                    torch.zeros(pad, dtype=torch.long)]))

    # 将列表中的 tensor 堆叠成 (batch_size, seq_len) 的二维 tensor
    return {
        "input_ids":      torch.stack(input_ids_list),
        "labels":         torch.stack(labels_list),
        "attention_mask": torch.stack(mask_list),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════════════════════
# parse_args: 命令行参数解析
# ════════════════════════════════════════════════════════════════════════════════
# 使用示例：
#   # LoRA 微调（默认，显存 ~8GB）
#   python train_sft_peoples_daily.py
#
#   # 全量微调（显存 ≥ 16GB）
#   python train_sft_peoples_daily.py --full_ft
#
#   # 自定义训练轮数和 batch size
#   python train_sft_peoples_daily.py --epochs 5 --batch_size 8 --grad_accum 2
def parse_args():
    parser = argparse.ArgumentParser(description="LLM SFT 人民日报 NER 训练（LoRA / 全量微调）")

    # ── 路径参数 ──
    parser.add_argument("--model_path",  default=str(MODEL_PATH),
                        help="预训练模型路径（默认 Qwen2-0.5B-Instruct）")
    parser.add_argument("--data_dir",    default=str(DATA_DIR),
                        help="数据集目录（需包含 train.json / validation.json）")
    parser.add_argument("--output_dir",  default=str(OUTPUT_DIR),
                        help="输出目录（checkpoint + 训练日志）")

    # ── 训练超参 ──
    parser.add_argument("--num_train",   default=-1,   type=int,
                        help="训练样本数，-1 使用全部数据（默认）")
    parser.add_argument("--epochs",      default=3,    type=int,
                        help="训练轮数（默认 3）")
    parser.add_argument("--batch_size",  default=4,    type=int,
                        help="每张卡的 batch size（默认 4）")
    parser.add_argument("--grad_accum",  default=4,    type=int,
                        help="梯度累积步数；等效 batch = batch_size × grad_accum（默认 4）")
    parser.add_argument("--lr",          default=None, type=float,
                        help="学习率；默认 LoRA=2e-4，全量=2e-5（自动判断）")
    parser.add_argument("--max_length",  default=256,  type=int,
                        help="序列最大长度；NER 的 JSON 输出比分类长，建议 256")

    # ── 微调模式 ──
    # 全量微调开关：不加此 flag 则默认使用 LoRA
    parser.add_argument("--full_ft",     action="store_true",
                        help="全量微调：跳过 LoRA，更新所有 495M 参数（需显存 ≥ 16GB）")

    # ── LoRA 超参（--full_ft 时忽略）──
    # r: 低秩矩阵的秩，越大表达能力越强，但参数量也越多
    # alpha: 缩放因子，通常设为 2×r
    parser.add_argument("--lora_r",      default=8,    type=int,
                        help="LoRA 秩 r（默认 8）")
    parser.add_argument("--lora_alpha",  default=16,   type=int,
                        help="LoRA 缩放因子 alpha（默认 16）")

    # ── 其他 ──
    parser.add_argument("--seed",        default=42,   type=int,
                        help="随机种子（默认 42）")

    return parser.parse_args()


# ════════════════════════════════════════════════════════════════════════════════
# main: 训练主流程
# ════════════════════════════════════════════════════════════════════════════════
# 整体流程：
#   1. 解析命令行参数，设置随机种子
#   2. 加载人民日报 NER 数据集（train.json / validation.json）
#   3. 加载 Qwen2-0.5B-Instruct 的 tokenizer 和模型
#   4. 根据 --full_ft 决定：LoRA 微调 or 全量微调
#   5. 训练循环：每个 epoch 计算 train_loss + val_loss，保存最优 checkpoint
#   6. 保存训练日志（JSON 格式，方便后续可视化）
def main():
    # ── Step 1：解析参数 + 设置随机种子（保证可复现）──
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 学习率默认值：LoRA 用较大的 lr（2e-4），全量微调用较小的 lr（2e-5）
    if args.lr is None:
        args.lr = 2e-5 if args.full_ft else 2e-4

    # 路径配置：数据目录、输出目录、checkpoint 目录
    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    ckpt_dir   = output_dir / ("sft_full_ckpt_peoples_daily" if args.full_ft else "sft_adapter_peoples_daily")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 设备选择：优先 CUDA（GPU），否则 CPU
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode_str = "全量微调" if args.full_ft else "LoRA 微调"
    print(f"使用设备: {device}  |  微调模式: {mode_str}")

    # ── Step 2：加载数据 ────────────────────────────────────────────────────────
    # 人民日报 NER 数据集格式：每条记录包含 tokens（字列表）和 ner_tags（BIO 标签列表）
    # 示例：{"tokens": ["厦", "门", "是"], "ner_tags": ["B-LOC", "I-LOC", "O"]}
    with open(data_dir / "train.json", encoding="utf-8") as f:
        train_raw = json.load(f)
    with open(data_dir / "validation.json", encoding="utf-8") as f:
        val_raw = json.load(f)

    # num_train > 0 时随机采样子集（快速调试用），-1 使用全部数据
    if args.num_train > 0:
        train_raw = random.sample(train_raw, min(args.num_train, len(train_raw)))   # random.sample(列表, n)，随机抽 n 个不重复元素
    # 验证集固定取前 300 条，避免验证时间过长
    print(f"训练集: {len(train_raw)} 条 | 验证集（前300条）: 300 条")

    # ── Step 3：加载 Tokenizer ───────────────────────────────────────────────────
    # Tokenizer 负责：文本 → token id 序列（编码）；token id → 文本（解码）
    # Qwen2 使用 BPE（Byte Pair Encoding）分词器
    print(f"\n加载 tokenizer: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(Path(args.model_path).resolve()), trust_remote_code=True    # .resolve() — 转为绝对路径，trust_remote_code=True信任模型仓库中的自定义代码
    )
    # 某些模型（如 Qwen2）没有预定义 pad_token，用 eos_token 代替
    # pad_token 用于 batch 内不同长度序列的填充
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ── Step 4：构建 Dataset + DataLoader ──────────────────────────────────────
    # SFTDataset 将原始数据转换为 {input_ids, labels} 格式
    # DataLoader 负责：分批、打乱、填充（通过 collate_fn）
    train_dataset = SFTDataset(train_raw, tokenizer, args.max_length)
    val_dataset   = SFTDataset(val_raw[:300], tokenizer, args.max_length)

    # collate_fn 需要 pad_id 参数，用 lambda 包装一下
    _collate = lambda b: collate_fn(b, tokenizer.pad_token_id)
    # 训练集 shuffle=True（打乱顺序），验证集 shuffle=False（顺序不变）
    # 验证集 batch_size 翻倍（eval 不需要梯度，显存占用更小，可以跑更快）
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, collate_fn=_collate)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size * 2,
                              shuffle=False, collate_fn=_collate)

    # ── Step 5：加载预训练模型 ──────────────────────────────────────────────────
    # AutoModelForCausalLM 会自动加载 causal language model（GPT/Qwen 都属于这类）
    # Qwen2-0.5B-Instruct 有 ~495M 参数
    print(f"加载 base model: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        str(Path(args.model_path).resolve()),
        dtype=torch.float32,  # transformers 5.x 用 dtype= 不用 torch_dtype=
        trust_remote_code=True,
    )

    # ── Step 6：选择微调方式（LoRA vs 全量微调）────────────────────────────────
    #
    # 全量微调（--full_ft）：
    #   - 更新模型所有参数（495M）
    #   - 显存占用大（batch_size=4 需要 ~16GB 显存）
    #   - 效果最好，但训练时间长
    #
    # LoRA 微调（默认）：
    #   - 只更新低秩分解矩阵（~几 M 参数，约占总参数 1%）
    #   - 显存占用小（batch_size=4 只需 ~8GB 显存）
    #   - 训练速度快，效果接近全量微调
    if args.full_ft:
        # ── 全量微调：所有参数都可训练 ──
        total = sum(p.numel() for p in model.parameters())
        print(f"trainable params: {total:,} || all params: {total:,} || trainable%: 100.0000")
    else:
        # ── LoRA 微调：冻结原始参数，只训练低秩适配器 ──
        if not PEFT_AVAILABLE:
            raise ImportError("LoRA 模式需要 peft 库：pip install peft>=0.14.0")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,    # 任务类型：因果语言模型
            r=args.lora_r,                    # 低秩矩阵的秩（越大越强，但参数越多）
            lora_alpha=args.lora_alpha,       # 缩放因子（通常 = 2r）
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # 对 attention 的 Q/K/V/O 矩阵加 LoRA
            lora_dropout=0.05,                # LoRA 层的 dropout（防过拟合）
            bias="none",                      # 不训练 bias（减少参数量）
        )
        model = get_peft_model(model, lora_config)
        # 打印可训练参数数量（通常只有总参数的 ~1%）
        model.print_trainable_parameters()

    # 将模型移到 GPU（如果有）或 CPU
    model = model.to(device)

    # ── Step 7：配置优化器 ─────────────────────────────────────────────────────
    # AdamW = Adam + 权重衰减（weight decay），是目前 LLM 微调的标准优化器
    # weight_decay=0.01 是常用的默认值，防止参数过大
    optimizer   = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # 总训练步数 = (训练样本数 / batch_size) × epochs / 梯度累积步数
    # 梯度累积：每 grad_accum 个 micro-batch 才更新一次参数
    # 等效 batch_size = batch_size × grad_accum（例如 4×4=16）
    total_steps = len(train_loader) * args.epochs // args.grad_accum
    print(f"总训练步数: {total_steps}（batch={args.batch_size}, "
          f"grad_accum={args.grad_accum}, epochs={args.epochs}, lr={args.lr}）\n")

    # ── Step 8：训练循环 ──────────────────────────────────────────────────────
    # 核心流程：每个 epoch → 训练集训练 → 验证集评估 → 保存最优 checkpoint
    best_val_loss = float("inf")    # 记录历史最优验证 loss（用于 early stopping）
    log_records   = []              # 训练日志记录（每 epoch 一条，最后保存为 JSON）

    for epoch in range(1, args.epochs + 1):
        # ── 训练阶段 ──
        model.train()   # 切换到训练模式（启用 dropout、LoRA 等）
        total_loss, total_tokens = 0.0, 0
        optimizer.zero_grad()   # 清零梯度（每个 epoch 开始时）
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]", leave=False)
        for step, batch in enumerate(pbar):
            # 将数据移到 GPU/CPU
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            # 前向传播：模型自动计算 cross-entropy loss
            # labels 中的 -100 会被 CrossEntropyLoss 忽略（只计算 JSON 部分的 loss）
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels)
            loss = outputs.loss   # 已经是 token 级别的平均 loss

            # 反向传播：loss / grad_accum 是因为梯度累积
            # 例如 grad_accum=4 时，每 4 个 micro-batch 的梯度累加后才更新一次参数
            (loss / args.grad_accum).backward()

            # 每 grad_accum 步更新一次参数
            if (step + 1) % args.grad_accum == 0:
                # 梯度裁剪：防止梯度爆炸（将梯度范数限制在 1.0 以内）
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()        # 更新参数
                optimizer.zero_grad()   # 清零梯度，为下一轮累积做准备

            # 统计：只统计真实 token（labels != -100）的 loss
            n_tokens      = (labels != -100).sum().item()
            total_loss   += loss.item() * n_tokens
            total_tokens += n_tokens
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # 计算整个训练集的平均 loss（按 token 数加权平均）
        avg_train_loss = total_loss / max(total_tokens, 1)

        # ── 验证阶段 ─────────────────────────────────────────────────────────
        # model.eval()：切换到推理模式（关闭 dropout，LoRA 参数固定）
        model.eval()
        val_loss, val_tokens = 0.0, 0
        # torch.no_grad()：禁用梯度计算，节省显存和计算时间
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Val", leave=False):
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels         = batch["labels"].to(device)
                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=labels)
                # 同样只统计真实 token 的 loss
                n_tokens   = (labels != -100).sum().item()
                val_loss   += outputs.loss.item() * n_tokens
                val_tokens += n_tokens
        avg_val_loss = val_loss / max(val_tokens, 1)

        # 打印本轮训练结果
        elapsed = time.time() - t0
        print(f"Epoch {epoch}/{args.epochs} | "
              f"train_loss={avg_train_loss:.4f}  val_loss={avg_val_loss:.4f} | {elapsed:.0f}s")

        # 记录日志（用于后续可视化训练曲线）
        log_records.append({
            "epoch": epoch, "train_loss": avg_train_loss,
            "val_loss": avg_val_loss, "elapsed_s": elapsed,
        })

        # ── 保存最优 checkpoint ─────────────────────────────────────────────
        # 只在验证 loss 创新低时保存（early stopping 的思路）
        # 这样可以避免保存过拟合的模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # save_pretrained 会保存：
            #   - 全量微调：完整模型权重（model.safetensors）
            #   - LoRA：只有 adapter 权重（很小，几十 MB）
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            ckpt_label = "完整模型" if args.full_ft else "LoRA adapter"
            print(f"  ✓ 最优{ckpt_label}已保存 → {ckpt_dir}  (val_loss={avg_val_loss:.4f})")

    # ── 保存训练日志 ──────────────────────────────────────────────────────────
    log_tag  = "full_ft" if args.full_ft else "sft"
    log_path = output_dir / "logs" / f"train_{log_tag}_peoples_daily.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_records, f, ensure_ascii=False, indent=2)

    ckpt_label = "完整模型" if args.full_ft else "LoRA adapter"
    print(f"\n训练完成。最优 val_loss={best_val_loss:.4f}")
    print(f"训练日志 → {log_path}")
    print(f"{ckpt_label} → {ckpt_dir}")
    print(f"\n下一步：python evaluate_sft_peoples_daily.py 查看 entity F1 与多方对比")


if __name__ == "__main__":
    main()
