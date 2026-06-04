"""
SFT 训练 NER：使用 LoRA 微调 Qwen2-0.5B-Instruct，生成 JSON 实体列表。
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
from peft import get_peft_model, LoraConfig, TaskType

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "ner"
MODEL_PATH = ROOT.parent.parent / "pretrain_models" / "Qwen2-0.5B-Instruct"
OUTPUT_DIR = ROOT / "outputs_ner"
ADAPTER_DIR = OUTPUT_DIR / "sft_adapter"

def get_entity_types(label_map_path):
    with open(label_map_path, "r", encoding="utf-8") as f:
        lm = json.load(f)
    types = set()
    for label in lm["id2label"].values():
        if label != "O" and label.startswith("B-"):
            types.add(label[2:])
    return list(types)

SYSTEM_PROMPT_TEMPLATE = """
你是一个命名实体识别（NER）助手。请从给定的新闻标题中抽取实体，以 JSON 列表形式输出。
实体类型包括：{types}。
只输出 JSON，不要有其他内容。
"""

def entities_to_json(text, char_labels, id2label):
    """将 BIO 标签转换为 JSON 字符串"""
    tokens = list(text)
    entities = []
    current = None
    for i, label_id in enumerate(char_labels):
        label = id2label[label_id]
        if label.startswith("B-"):
            if current:
                entities.append(current)
            current = {"type": label[2:], "tokens": [tokens[i]]}
        elif label.startswith("I-") and current and current["type"] == label[2:]:
            current["tokens"].append(tokens[i])
        else:
            if current:
                entities.append(current)
                current = None
    if current:
        entities.append(current)
    ent_list = [{"text": "".join(e["tokens"]), "type": e["type"]} for e in entities]
    return json.dumps(ent_list, ensure_ascii=False)

class NERSFTDataset(Dataset):
    def __init__(self, data, tokenizer, id2label, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.id2label = id2label
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["text"]
        labels = item["labels"]
        response_str = entities_to_json(text, labels, self.id2label)

        # 构建 prompt
        entity_types = get_entity_types(Path(__file__).parent.parent / "data/ner/label_map.json")
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(types="、".join(entity_types))
        user_prompt = f"新闻标题：{text}\n实体列表："
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        response_ids = self.tokenizer.encode(response_str, add_special_tokens=False) + [self.tokenizer.eos_token_id]

        input_ids = (prompt_ids + response_ids)[:self.max_length]
        labels_ids = ([-100] * len(prompt_ids) + response_ids)[:self.max_length]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels_ids, dtype=torch.long),
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
    return {
        "input_ids": torch.stack(input_ids_list),
        "labels": torch.stack(labels_list),
        "attention_mask": torch.stack(mask_list),
    }

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=str(MODEL_PATH))
    parser.add_argument("--data_dir", default=str(DATA_DIR))
    parser.add_argument("--output_dir", default=str(OUTPUT_DIR))
    parser.add_argument("--num_train", default=5000, type=int)
    parser.add_argument("--epochs", default=3, type=int)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--grad_accum", default=4, type=int)
    parser.add_argument("--lr", default=2e-4, type=float)
    parser.add_argument("--max_length", default=256, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--seed", default=42, type=int)
    return parser.parse_args()

def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    ckpt_dir = output_dir / "sft_adapter"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device} | LoRA 微调")

    # 加载数据
    with open(data_dir / "train.json", "r", encoding="utf-8") as f:
        train_raw = json.load(f)
    with open(data_dir / "val.json", "r", encoding="utf-8") as f:
        val_raw = json.load(f)
    with open(data_dir / "label_map.json", "r", encoding="utf-8") as f:
        label_map = json.load(f)
    id2label = {int(k): v for k, v in label_map["id2label"].items()}

    if args.num_train > 0:
        train_raw = random.sample(train_raw, min(args.num_train, len(train_raw)))
    print(f"训练集: {len(train_raw)} 条 | 验证集（前200条）: 200 条")
    val_raw = val_raw[:200]

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(Path(args.model_path).resolve()), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Dataset
    train_dataset = NERSFTDataset(train_raw, tokenizer, id2label, args.max_length)
    val_dataset = NERSFTDataset(val_raw, tokenizer, id2label, args.max_length)

    collate = lambda b: collate_fn(b, tokenizer.pad_token_id)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size * 2, shuffle=False, collate_fn=collate)

    # 模型
    model = AutoModelForCausalLM.from_pretrained(
        str(Path(args.model_path).resolve()),
        dtype=torch.float32,
        trust_remote_code=True,
    )
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

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs // args.grad_accum
    print(f"总训练步数: {total_steps} (batch={args.batch_size}, grad_accum={args.grad_accum})")

    best_val_loss = float("inf")
    log_records = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()
        t0 = time.time()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]", leave=False)
        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            (loss / args.grad_accum).backward()
            total_loss += loss.item()
            if (step + 1) % args.grad_accum == 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        avg_train_loss = total_loss / len(train_loader)

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()
        avg_val_loss = val_loss / len(val_loader)
        elapsed = time.time() - t0
        print(f"Epoch {epoch}/{args.epochs} | train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f} | {elapsed:.0f}s")
        log_records.append({"epoch": epoch, "train_loss": avg_train_loss, "val_loss": avg_val_loss})

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            print(f"  ✓ 最优 LoRA adapter 已保存 → {ckpt_dir} (val_loss={avg_val_loss:.4f})")

    # 保存日志
    log_path = output_dir / "train_log_sft_ner.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_records, f, ensure_ascii=False, indent=2)
    print(f"\n训练完成。最优 val_loss={best_val_loss:.4f}")
    print(f"Log: {log_path}")

if __name__ == "__main__":
    main()