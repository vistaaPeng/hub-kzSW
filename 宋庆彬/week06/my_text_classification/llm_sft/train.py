"""LLM SFT LoRA 训练"""

import os
import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType
from tqdm import tqdm

from common.config import DATA_DIR, OUTPUT_DIR, ADAPTER_DIR, QWEN_MODEL_NAME, LABEL_NAMES, SFT_DEFAULTS
from common.utils import get_device

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

SYSTEM_PROMPT = (
    "你是一个新闻标题分类助手。请将给定的新闻标题分类到以下类别之一，"
    "只输出类别名称，不要输出任何其他内容。\n"
    "可选类别：" + "、".join(LABEL_NAMES)
)


class SFTDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=64):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        label_name = LABEL_NAMES[item["label"]]

        # prompt text
        prompt_text = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"新闻标题：{item['sentence']}\n类别："},
            ],
            tokenize=False, add_generation_prompt=True,
        )
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        prompt_len = len(prompt_ids)

        # response: 类别名 + EOS
        response_ids = self.tokenizer.encode(label_name, add_special_tokens=False) + [self.tokenizer.eos_token_id]

        # 拼接 + 截断
        input_ids = (prompt_ids + response_ids)[: self.max_length]

        # loss mask: prompt 部分 = -100
        labels = ([-100] * prompt_len + response_ids)[: self.max_length]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def collate_fn(batch, pad_id):
    max_len = max(item["input_ids"].size(0) for item in batch)
    input_ids_list, labels_list, mask_list = [], [], []
    for item in batch:
        n = item["input_ids"].size(0)
        pad = max_len - n
        input_ids_list.append(torch.cat([item["input_ids"], torch.full((pad,), pad_id, dtype=torch.long)]))
        labels_list.append(torch.cat([item["labels"], torch.full((pad,), -100, dtype=torch.long)]))
        mask_list.append(torch.cat([torch.ones(n, dtype=torch.long), torch.zeros(pad, dtype=torch.long)]))
    return {"input_ids": torch.stack(input_ids_list), "labels": torch.stack(labels_list), "attention_mask": torch.stack(mask_list)}


def compute_val_loss(model, loader, device):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            outputs = model(input_ids=batch["input_ids"].to(device),
                            attention_mask=batch["attention_mask"].to(device),
                            labels=batch["labels"].to(device))
            n = (batch["labels"] != -100).sum().item()
            total_loss += outputs.loss.item() * n
            total_tokens += n
    return total_loss / max(total_tokens, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_train", type=int, default=SFT_DEFAULTS["num_train"])
    parser.add_argument("--epochs", type=int, default=SFT_DEFAULTS["epochs"])
    parser.add_argument("--batch_size", type=int, default=SFT_DEFAULTS["batch_size"])
    parser.add_argument("--max_length", type=int, default=SFT_DEFAULTS["max_length"])
    parser.add_argument("--grad_accum", type=int, default=SFT_DEFAULTS["grad_accum"])
    parser.add_argument("--lr", type=float, default=SFT_DEFAULTS["lr"])
    parser.add_argument("--lora_r", type=int, default=SFT_DEFAULTS["lora_r"])
    parser.add_argument("--lora_alpha", type=int, default=SFT_DEFAULTS["lora_alpha"])
    parser.add_argument("--seed", type=int, default=SFT_DEFAULTS["seed"])
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = get_device()
    print(f"设备: {device}  |  LoRA r={args.lora_r}  |  训练 {args.num_train} 条 × {args.epochs} epoch")

    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)

    # Data
    with open(DATA_DIR / "train.json", encoding="utf-8") as f:
        train_raw = json.load(f)
    with open(DATA_DIR / "val.json", encoding="utf-8") as f:
        val_raw = json.load(f)

    if args.num_train > 0:
        train_raw = random.sample(train_raw, min(args.num_train, len(train_raw)))
    val_raw = val_raw[:SFT_DEFAULTS["val_subset"]]
    print(f"train={len(train_raw)}  val={len(val_raw)}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    train_ds = SFTDataset(train_raw, tokenizer, args.max_length)
    val_ds = SFTDataset(val_raw, tokenizer, args.max_length)
    _collate = lambda b: collate_fn(b, tokenizer.pad_token_id)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=_collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False, collate_fn=_collate)

    # Model + LoRA
    print(f"加载 {QWEN_MODEL_NAME} ...")
    model = AutoModelForCausalLM.from_pretrained(QWEN_MODEL_NAME, torch_dtype=torch.float32, trust_remote_code=True)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=args.lora_r, lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj"], lora_dropout=SFT_DEFAULTS["lora_dropout"], bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model = model.to(device)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # Training
    best_val_loss = float("inf")
    log_records = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total_tokens = 0.0, 0
        optimizer.zero_grad()
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for step, batch in enumerate(pbar):
            outputs = model(input_ids=batch["input_ids"].to(device),
                            attention_mask=batch["attention_mask"].to(device),
                            labels=batch["labels"].to(device))
            loss = outputs.loss / args.grad_accum
            loss.backward()

            if (step + 1) % args.grad_accum == 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            n = (batch["labels"] != -100).sum().item()
            total_loss += outputs.loss.item() * n
            total_tokens += n
            pbar.set_postfix(loss=f"{outputs.loss.item():.4f}")

        train_loss = total_loss / max(total_tokens, 1)
        val_loss = compute_val_loss(model, val_loader, device)
        elapsed = time.time() - t0

        print(f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f}  val_loss={val_loss:.4f} | {elapsed:.0f}s")
        log_records.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained(ADAPTER_DIR)
            tokenizer.save_pretrained(ADAPTER_DIR)
            print(f"  ✓ 最优 adapter → {ADAPTER_DIR}")

    log_path = OUTPUT_DIR / "train_log_sft.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_records, f, ensure_ascii=False, indent=2)
    print(f"训练完成。best_val_loss={best_val_loss:.4f} | 日志 → {log_path}")


if __name__ == "__main__":
    main()
