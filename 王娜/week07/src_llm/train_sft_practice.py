"""
LLM SFT（监督微调）训练脚本 — 基于 LoRA 高效微调 Qwen2-0.5B-Instruct 做 NER
针对 peoples_daily 数据集（BIO 格式）

教学重点：
  1. NER 的指令微调格式：输入是文本，输出是 JSON 实体列表
  2. Loss masking：只在 JSON 输出部分计算 loss，prompt 全为 -100
  3. LoRA 高效微调：参数量约 0.22%
  4. 处理 BIO 格式数据（tokens + ner_tags）

使用方式：
  python train_sft_practice.py                        # LoRA，全量训练数据（默认）
  python train_sft_practice.py --num_train 2000       # LoRA，2000 条快速演示
  python train_sft_practice.py --epochs 1             # 快速验证流程

依赖：
  pip install torch transformers peft tqdm
"""

import os
import argparse
import json
import random
import time
from pathlib import Path

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
MODEL_PATH = ROOT.parent.parent.parent.parent / "pretrain_models" / "Qwen2-0.5B-Instruct"
OUTPUT_DIR = ROOT / "outputs"

# peoples_daily 实体类型
ENTITY_TYPES = ["PER", "ORG", "LOC"]
ENTITY_TYPE_ZH = {
    "PER": "人名",
    "ORG": "机构",
    "LOC": "地名",
}

SYSTEM_PROMPT = (
    "你是一个命名实体识别助手。从文本中识别命名实体，以 JSON 格式输出。\n"
    "实体类型（英文标识）：PER（人名）、ORG（机构）、LOC（地名）\n"
    '输出格式（严格遵守，不输出其他内容）：{"entities": [{"text": "实体文本", "type": "实体类型"}]}\n'
    '无实体时输出：{"entities": []}'
)


def record_to_target(record: dict) -> str:
    """把 peoples_daily BIO 格式转为 SFT 目标 JSON 字符串。
    输入：{"tokens": ["海", "钓", "比", "赛", ...], "ner_tags": ["O", "O", "B-LOC", "I-LOC", ...]}
    输出：'{"entities": [{"text": "厦门", "type": "LOC"}, ...]}'
    """
    tokens = record["tokens"]
    ner_tags = record["ner_tags"]
    entities = []
    n = len(ner_tags)
    i = 0
    while i < n:
        tag = ner_tags[i]
        if tag.startswith("B-"):
            etype = tag[2:]
            start = i
            i += 1
            while i < n and ner_tags[i] == f"I-{etype}":
                i += 1
            end = i - 1
            surface = "".join(tokens[start:end + 1])
            entities.append({"text": surface, "type": etype})
        else:
            i += 1
    return json.dumps({"entities": entities}, ensure_ascii=False)


# ══════════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════════

class SFTDataset(Dataset):
    """
    把 peoples_daily NER 数据转换为 chat-format SFT 训练样本。
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

        # ── Step 1：构建 prompt 文本（tokenize=False 兼容 transformers 5.x）──
        text = "".join(item["tokens"])
        prompt_text = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": text},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)

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
    parser = argparse.ArgumentParser(description="LLM SFT NER 训练（peoples_daily）")
    parser.add_argument("--model_path",  default=str(MODEL_PATH))
    parser.add_argument("--data_dir",    default=str(DATA_DIR))
    parser.add_argument("--output_dir",  default=str(OUTPUT_DIR))
    parser.add_argument("--num_train",   default=-1,   type=int,
                        help="训练样本数，-1 使用全部数据（默认）")
    parser.add_argument("--epochs",      default=3,    type=int)
    parser.add_argument("--batch_size",  default=4,    type=int)
    parser.add_argument("--grad_accum",  default=4,    type=int)
    parser.add_argument("--lr",          default=2e-4, type=float)
    parser.add_argument("--max_length",  default=256,  type=int)
    parser.add_argument("--lora_r",      default=8,    type=int)
    parser.add_argument("--lora_alpha",  default=16,   type=int)
    parser.add_argument("--seed",        default=42,   type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    ckpt_dir   = output_dir / "sft_adapter_peoples_daily"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}  |  微调模式: LoRA")

    # ── 加载数据 ──────────────────────────────────────────────────────────────
    with open(data_dir / "train.json", encoding="utf-8") as f:
        train_raw = json.load(f)
    with open(data_dir / "validation.json", encoding="utf-8") as f:
        val_raw = json.load(f)

    if args.num_train > 0:
        train_raw = random.sample(train_raw, min(args.num_train, len(train_raw)))
    print(f"训练集: {len(train_raw)} 条 | 验证集（前300条）: {min(300, len(val_raw))} 条")

    # ── 加载 Tokenizer ─────────────────────────────────────────────────────────
    print(f"\n加载 tokenizer: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(Path(args.model_path).resolve()), trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ── 构建数据集 ─────────────────────────────────────────────────────────────
    train_dataset = SFTDataset(train_raw, tokenizer, args.max_length)
    val_dataset   = SFTDataset(val_raw[:300], tokenizer, args.max_length)

    _collate = lambda b: collate_fn(b, tokenizer.pad_token_id)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, collate_fn=_collate)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size * 2,
                              shuffle=False, collate_fn=_collate)

    # ── 加载模型 ───────────────────────────────────────────────────────────────
    print(f"加载 base model: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        str(Path(args.model_path).resolve()),
        dtype=torch.float32,
        trust_remote_code=True,
    )

    # ── LoRA 配置 ─────────────────────────────────────────────────────────────
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

    # ── 优化器 ────────────────────────────────────────────────────────────────
    optimizer   = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs // args.grad_accum
    print(f"总训练步数: {total_steps}（batch={args.batch_size}, "
          f"grad_accum={args.grad_accum}, epochs={args.epochs}, lr={args.lr}）\n")

    # ── 训练循环 ──────────────────────────────────────────────────────────────
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

        # ── 验证 loss ─────────────────────────────────────────────────────────
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
            print(f"  ✓ 最优 LoRA adapter 已保存 → {ckpt_dir}  (val_loss={avg_val_loss:.4f})")

    # ── 保存训练日志 ──────────────────────────────────────────────────────────
    log_path = output_dir / "logs" / "train_sft_peoples_daily.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_records, f, ensure_ascii=False, indent=2)

    print(f"\n训练完成。最优 val_loss={best_val_loss:.4f}")
    print(f"训练日志 → {log_path}")
    print(f"LoRA adapter → {ckpt_dir}")


if __name__ == "__main__":
    main()