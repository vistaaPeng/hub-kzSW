"""
LLM SFT（监督微调）训练脚本 — 基于 LoRA 高效微调 Qwen2-0.5B-Instruct 做 NER
适配新数据集格式：{"tokens": [...], "ner_tags": [...]}  (BIO, 类型: PER/ORG/LOC)

教学重点：
  1. NER 的指令微调格式：输入是文本，输出是 JSON 实体列表（三类）
  2. Loss masking：只在 JSON 输出部分计算 loss
  3. LoRA 高效微调 vs 全量微调
  4. 生成式 NER 的优势

使用方式：
  python train_sft.py                        # LoRA，全量训练数据
  python train_sft.py --num_train 2000       # LoRA，2000 条快速演示
  python train_sft.py --epochs 1             # 快速验证流程
  python train_sft.py --full_ft --lr 2e-5    # 全量微调（需显存 ≥ 16GB）

依赖：
  pip install torch transformers peft tqdm
"""

import os
import argparse
import json
import random
import time
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

try:
    from peft import get_peft_model, LoraConfig, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

ROOT       = Path(__file__).parent.parent
DATA_DIR   = ROOT / "data" / "peoples_daily"
MODEL_PATH = "/root/autodl-tmp/pretrain_models/Qwen2-0.5B-Instruct"
OUTPUT_DIR = ROOT / "outputs"

# 新数据集的标签映射 (BIO 类型 -> 英文实体类型)
TAG_TO_TYPE = {
    "PER": "name",
    "ORG": "organization",
    "LOC": "location",
}
ENTITY_TYPES = list(TAG_TO_TYPE.values())  # ["name", "organization", "location"]

SYSTEM_PROMPT = (
    "你是一个命名实体识别助手。从文本中识别命名实体，以 JSON 格式输出。\n"
    "实体类型（英文标识）：name（人名）、organization（组织机构，如公司、政府、球队等）、location（地点，如城市、景点、地址）\n"
    '输出格式（严格遵守，不输出其他内容）：{"entities": [{"text": "实体文本", "type": "实体类型"}]}\n'
    '无实体时输出：{"entities": []}'
)


def extract_entities_from_tags(tokens: list[str], ner_tags: list[str]) -> list[dict]:
    """
    从 BIO 标签序列中提取实体列表。
    输入：
        tokens: ["浙", "商", "银", "行", ...]
        ner_tags: ["B-ORG", "I-ORG", "I-ORG", "I-ORG", ...]
    输出：
        [{"text": "浙商银行", "type": "organization"}, ...]
    """
    entities = []
    i = 0
    n = len(tokens)
    while i < n:
        tag = ner_tags[i]
        if tag == "O" or not tag.startswith(("B-", "I-")):
            i += 1
            continue

        # 实体开头必须为 B-
        if tag.startswith("B-"):
            bio_type = tag[2:]          # 例如 "ORG"
            etype = TAG_TO_TYPE.get(bio_type)
            if etype is None:
                i += 1
                continue

            start = i
            entity_tokens = [tokens[i]]
            i += 1
            # 收集连续的 I- 标签（必须与 B- 类型一致）
            while i < n and ner_tags[i] == f"I-{bio_type}":
                entity_tokens.append(tokens[i])
                i += 1
            end = i - 1
            surface = "".join(entity_tokens)
            entities.append({"text": surface, "type": etype})
        else:
            # 非 B- 开头的 I- 标签（理论上不合法），跳过
            i += 1
    return entities


def record_to_target(record: dict) -> str:
    """把新格式记录转为 SFT 目标 JSON 字符串。"""
    tokens = record.get("tokens", [])
    tags   = record.get("ner_tags", [])
    if not tokens or not tags or len(tokens) != len(tags):
        return json.dumps({"entities": []}, ensure_ascii=False)
    entities = extract_entities_from_tags(tokens, tags)
    return json.dumps({"entities": entities}, ensure_ascii=False)


# ══════════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════════

class SFTDataset(Dataset):
    """
    将 NER 数据转换为 chat-format SFT 训练样本。
    输入文本：将 tokens 列表拼接成字符串。
    """

    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # 构建输入文本（将 tokens 拼成字符串）
        text = "".join(item.get("tokens", []))
        target = record_to_target(item)

        # ── Step 1：构建 prompt 文本（chat template）──
        prompt_text = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": text},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)

        # ── Step 2：response = JSON 字符串 + EOS ──
        response_ids = (
            self.tokenizer.encode(target, add_special_tokens=False)
            + [self.tokenizer.eos_token_id]
        )

        # ── Step 3：拼接 + 截断 ──
        input_ids = (prompt_ids + response_ids)[: self.max_length]

        # ── Step 4：loss mask：prompt 全 -100，只在 JSON 部分计算 loss ──
        labels = ([-100] * len(prompt_ids) + response_ids)[: self.max_length]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels":    torch.tensor(labels,    dtype=torch.long),
        }


def collate_fn(batch, pad_id):
    max_len = max(item["input_ids"].size(0) for item in batch)
    input_ids_list, labels_list, mask_list = [], [], []
    for item in batch:
        n   = item["input_ids"].size(0)
        pad = max_len - n
        input_ids_list.append(torch.cat([item["input_ids"],
                                         torch.full((pad,), pad_id, dtype=torch.long)]))
        labels_list.append(torch.cat([item["labels"],
                                      torch.full((pad,), -100, dtype=torch.long)]))
        mask_list.append(torch.cat([torch.ones(n, dtype=torch.long),
                                    torch.zeros(pad, dtype=torch.long)]))
    return {
        "input_ids":      torch.stack(input_ids_list),
        "labels":         torch.stack(labels_list),
        "attention_mask": torch.stack(mask_list),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="LLM SFT NER 训练（LoRA / 全量微调）— 新数据集格式")
    parser.add_argument("--model_path",  default=str(MODEL_PATH))
    parser.add_argument("--data_dir",    default=str(DATA_DIR))
    parser.add_argument("--output_dir",  default=str(OUTPUT_DIR))
    parser.add_argument("--num_train",   default=-1,   type=int,
                        help="训练样本数，-1 使用全部")
    parser.add_argument("--epochs",      default=3,    type=int)
    parser.add_argument("--batch_size",  default=4,    type=int)
    parser.add_argument("--grad_accum",  default=4,    type=int)
    parser.add_argument("--lr",          default=None, type=float,
                        help="学习率；默认 LoRA=2e-4，全量=2e-5（自动判断）")
    parser.add_argument("--max_length",  default=512,  type=int,
                        help="序列最大长度；NER 输出较长，建议 512")
    parser.add_argument("--full_ft",     action="store_true",
                        help="全量微调：跳过 LoRA，更新所有参数（需显存 ≥ 16GB）")
    parser.add_argument("--lora_r",      default=8,    type=int)
    parser.add_argument("--lora_alpha",  default=16,   type=int)
    parser.add_argument("--seed",        default=42,   type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.lr is None:
        args.lr = 2e-5 if args.full_ft else 2e-4

    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    ckpt_dir   = output_dir / ("sft_full_ckpt" if args.full_ft else "sft_adapter")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode_str = "全量微调" if args.full_ft else "LoRA 微调"
    print(f"使用设备: {device}  |  微调模式: {mode_str}")

    # ── 加载新格式数据 ──────────────────────────────────────────────────────────────
    train_file = data_dir / "train.json"
    val_file   = data_dir / "validation.json"
    if not train_file.exists():
        raise FileNotFoundError(f"训练数据文件不存在: {train_file}")
    if not val_file.exists():
        raise FileNotFoundError(f"验证数据文件不存在: {val_file}")

    with open(train_file, encoding="utf-8") as f:
        train_raw = json.load(f)
    with open(val_file, encoding="utf-8") as f:
        val_raw = json.load(f)

    # 验证数据格式
    for sample in train_raw[:5]:
        if "tokens" not in sample or "ner_tags" not in sample:
            raise ValueError("数据格式错误，每条记录必须包含 'tokens' 和 'ner_tags' 字段")

    if args.num_train > 0:
        train_raw = random.sample(train_raw, min(args.num_train, len(train_raw)))
    print(f"训练集: {len(train_raw)} 条 | 验证集: {len(val_raw)} 条")

    # ── 加载 Tokenizer ──
    print(f"\n加载 tokenizer: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(Path(args.model_path).resolve()), trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ── 构建数据集 ──
    train_dataset = SFTDataset(train_raw, tokenizer, args.max_length)
    val_dataset   = SFTDataset(val_raw, tokenizer, args.max_length)

    _collate = lambda b: collate_fn(b, tokenizer.pad_token_id)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, collate_fn=_collate)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size * 2,
                              shuffle=False, collate_fn=_collate)

    # ── 加载模型 ──
    print(f"加载 base model: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        str(Path(args.model_path).resolve()),
        dtype=torch.float32,      # 使用 float32 兼容性更好（可根据显存改为 bf16）
        trust_remote_code=True,
    )

    # ── LoRA 或全量微调 ──
    if args.full_ft:
        total = sum(p.numel() for p in model.parameters())
        print(f"trainable params: {total:,} || all params: {total:,} || trainable%: 100.0000")
    else:
        if not PEFT_AVAILABLE:
            raise ImportError("LoRA 模式需要 peft 库：pip install peft>=0.14.0")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    model = model.to(device)

    # ── 优化器 ──
    optimizer   = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs // args.grad_accum
    print(f"总训练步数: {total_steps}（batch={args.batch_size}, "
          f"grad_accum={args.grad_accum}, epochs={args.epochs}, lr={args.lr}）\n")

    # ── 训练循环 ──
    best_val_loss = float("inf")
    log_records   = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total_tokens = 0.0, 0
        optimizer.zero_grad()
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]", leave=False)
        for step, batch in enumerate(pbar):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels)
            loss = outputs.loss

            (loss / args.grad_accum).backward()
            if (step + 1) % args.grad_accum == 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            n_tokens      = (labels != -100).sum().item()
            total_loss   += loss.item() * n_tokens
            total_tokens += n_tokens
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = total_loss / max(total_tokens, 1)

        # 验证 loss
        model.eval()
        val_loss, val_tokens = 0.0, 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Val", leave=False):
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels         = batch["labels"].to(device)
                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=labels)
                n_tokens   = (labels != -100).sum().item()
                val_loss   += outputs.loss.item() * n_tokens
                val_tokens += n_tokens
        avg_val_loss = val_loss / max(val_tokens, 1)

        elapsed = time.time() - t0
        print(f"Epoch {epoch}/{args.epochs} | "
              f"train_loss={avg_train_loss:.4f}  val_loss={avg_val_loss:.4f} | {elapsed:.0f}s")

        log_records.append({
            "epoch": epoch, "train_loss": avg_train_loss,
            "val_loss": avg_val_loss, "elapsed_s": elapsed,
        })

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            ckpt_label = "完整模型" if args.full_ft else "LoRA adapter"
            print(f"  ✓ 最优{ckpt_label}已保存 → {ckpt_dir}  (val_loss={avg_val_loss:.4f})")

    # 保存训练日志
    log_tag  = "full_ft" if args.full_ft else "sft"
    log_path = output_dir / "logs" / f"train_{log_tag}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_records, f, ensure_ascii=False, indent=2)

    ckpt_label = "完整模型" if args.full_ft else "LoRA adapter"
    print(f"\n训练完成。最优 val_loss={best_val_loss:.4f}")
    print(f"训练日志 → {log_path}")
    print(f"{ckpt_label} → {ckpt_dir}")
    print(f"\n下一步：python evaluate_sft.py 查看实体 F1 与多方对比")


if __name__ == "__main__":
    main()
