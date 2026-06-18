"""
QLoRA SFT 训练脚本 — 4-bit 量化大模型 + LoRA 微调做 NER

区别于 train_sft.py（0.5B 全量/LoRA 微调）：
  1. 支持 4-bit QLoRA（NF4 量化 + LoRA），跑 7B+ 模型只需 ~15GB 显存
  2. 支持 --dataset cluener2020 / peoples_daily 切换
  3. 模型路径支持 HuggingFace 模型名（不走 Path.resolve()）

使用方式：
  # 本地 4090D — 完整训练
  python train_sft_qlora.py --model_name Qwen/Qwen2.5-7B-Instruct --dataset peoples_daily

  # 快速验证（500 条，1 epoch）
  python train_sft_qlora.py --model_name Qwen/Qwen2.5-7B-Instruct --num_train 500 --epochs 1

依赖：
  pip install torch transformers peft bitsandbytes accelerate

显存参考（4090D 24GB）：
  Qwen2.5-7B + QLoRA:  ~14-16GB (bs=2, ga=4)
  Qwen2.5-14B + QLoRA: ~20-22GB (bs=1, ga=8)
"""

import os
import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm

# Reuse from existing train_sft.py
sys.path.insert(0, str(Path(__file__).parent))
from train_sft import SFTDataset, collate_fn

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT       = Path(__file__).parent.parent
OUTPUT_DIR = ROOT / "outputs"

# ── 实体类型定义 ─────────────────────────────────────────────────────────

ENTITY_TYPES = {
    "cluener2020": [
        "address", "book", "company", "game", "government",
        "movie", "name", "organization", "position", "scene",
    ],
    "peoples_daily": ["PER", "ORG", "LOC"],
}

TYPE_NAMES_CN = {
    "address": "地址", "book": "书名", "company": "公司",
    "game": "游戏", "government": "政府机构", "movie": "影视作品",
    "name": "人名", "organization": "组织机构", "position": "职位", "scene": "景点/场所",
    "PER": "人名", "ORG": "组织机构", "LOC": "地名",
}


def build_system_prompt(dataset: str, extra: str = "") -> str:
    """根据数据集生成对应的 system prompt。extra 为追加提示词。"""
    types = ENTITY_TYPES[dataset]
    type_desc = "、".join(f"{t}（{TYPE_NAMES_CN.get(t, t)}）" for t in types)
    base = (
        "你是一个命名实体识别助手。从文本中识别命名实体，以 JSON 格式输出。\n"
        f"实体类型（英文标识）：{type_desc}\n"
        '输出格式（严格遵守，不输出其他内容）：{"entities": [{"text": "实体文本", "type": "实体类型"}]}\n'
        '无实体时输出：{"entities": []}'
    )
    if extra:
        base += "\n" + extra
    return base


# ── 数据转换 ─────────────────────────────────────────────────────────────

def record_to_target_cluener(record: dict) -> str:
    """cluener2020 span 格式 → SFT 目标 JSON。"""
    entities = []
    for etype, surfaces in (record.get("label") or {}).items():
        for surface in surfaces:
            entities.append({"text": surface, "type": etype})
    return json.dumps({"entities": entities}, ensure_ascii=False)


def record_to_target_peoples_daily(record: dict) -> str:
    """peoples_daily BIO 格式 → SFT 目标 JSON。"""
    tokens = record["tokens"]
    tags = record["ner_tags"]
    entities = []
    i = 0
    while i < len(tags):
        if tags[i].startswith("B-"):
            etype = tags[i][2:]
            j = i + 1
            while j < len(tags) and tags[j] == f"I-{etype}":
                j += 1
            text = "".join(tokens[i:j])
            entities.append({"text": text, "type": etype})
            i = j
        else:
            i += 1
    return json.dumps({"entities": entities}, ensure_ascii=False)


# ── QLoRA Dataset（复用 SFTDataset，替换 record_to_target）─────────────────

class QLoRADataset(SFTDataset):
    """继承 SFTDataset，根据数据集类型选择不同的 record_to_target。"""

    def __init__(self, data, tokenizer, max_length=256, dataset="cluener2020", prompt_extra=""):
        self.dataset = dataset
        self.prompt_extra = prompt_extra
        super().__init__(data, tokenizer, max_length, dataset=dataset)

    def __getitem__(self, idx):
        item = self.data[idx]
        if self.dataset == "peoples_daily":
            target = record_to_target_peoples_daily(item)
        else:
            target = record_to_target_cluener(item)

        prompt_text = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": build_system_prompt(self.dataset, self.prompt_extra)},
                {"role": "user", "content": item.get("text", "".join(item.get("tokens", [])))},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)

        response_ids = (
            self.tokenizer.encode(target, add_special_tokens=False)
            + [self.tokenizer.eos_token_id]
        )

        input_ids = (prompt_ids + response_ids)[: self.max_length]
        labels = ([-100] * len(prompt_ids) + response_ids)[: self.max_length]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


# ── CLI ───────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="QLoRA SFT NER 训练")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-7B-Instruct",
                        help="HF 模型名或本地路径")
    parser.add_argument("--dataset", default="peoples_daily",
                        choices=["cluener2020", "peoples_daily"])
    parser.add_argument("--num_train", type=int, default=-1,
                        help="训练样本数，-1=全部")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_tag", default=None,
                        help="输出目录后缀（默认自动生成）")
    parser.add_argument("--prompt_extra", default="",
                        help="追加到 system prompt 的额外指令")
    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("ERROR: QLoRA 需要 CUDA GPU")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem  = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")

    # ── 数据 ──────────────────────────────────────────────────────────────
    data_dir = ROOT / "data" / ("cluener" if args.dataset == "cluener2020" else args.dataset)
    with open(data_dir / "train.json", encoding="utf-8") as f:
        train_raw = json.load(f)
    with open(data_dir / "validation.json", encoding="utf-8") as f:
        val_raw = json.load(f)

    if args.num_train > 0:
        train_raw = random.sample(train_raw, min(args.num_train, len(train_raw)))
    print(f"数据集: {args.dataset} | 训练={len(train_raw)}, 验证(前300)={min(300, len(val_raw))}")

    # ── Tokenizer ─────────────────────────────────────────────────────────
    print(f"加载 tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ── Dataset ───────────────────────────────────────────────────────────
    train_ds = QLoRADataset(train_raw, tokenizer, args.max_length, dataset=args.dataset, prompt_extra=args.prompt_extra)
    val_ds   = QLoRADataset(val_raw[:300], tokenizer, args.max_length, dataset=args.dataset, prompt_extra=args.prompt_extra)

    _collate = lambda b: collate_fn(b, tokenizer.pad_token_id)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, collate_fn=_collate)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size * 2,
                              shuffle=False, collate_fn=_collate)

    # ── 4-bit QLoRA 模型加载 ──────────────────────────────────────────────
    print(f"加载模型 (4-bit QLoRA): {args.model_name}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # ── LoRA 配置 ─────────────────────────────────────────────────────────
    from peft import LoraConfig, get_peft_model, TaskType
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

    # ── 优化器 ─────────────────────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs // args.grad_accum
    print(f"训练步数: {total_steps} (bs={args.batch_size}, ga={args.grad_accum}, "
          f"ep={args.epochs}, lr={args.lr})\n")

    # ── 输出目录 ──────────────────────────────────────────────────────────
    tag = args.output_tag or f"{args.dataset}_qlora"
    ckpt_dir = OUTPUT_DIR / f"sft_{tag}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir = OUTPUT_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # ── 训练循环 ──────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    log_records = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total_tokens = 0.0, 0
        optimizer.zero_grad()
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for step, batch in enumerate(pbar):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            (loss / args.grad_accum).backward()
            if (step + 1) % args.grad_accum == 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            n_tokens = (labels != -100).sum().item()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # 处理最后不足 grad_accum 的 batch
        remainder = len(train_loader) % args.grad_accum
        if remainder != 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        avg_train_loss = total_loss / max(total_tokens, 1)

        # ── 验证 ──────────────────────────────────────────────────────────
        model.eval()
        val_loss, val_tokens = 0.0, 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Val", leave=False):
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels         = batch["labels"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                n_tokens = (labels != -100).sum().item()
                val_loss += outputs.loss.item() * n_tokens
                val_tokens += n_tokens
        avg_val_loss = val_loss / max(val_tokens, 1)

        elapsed = time.time() - t0
        print(f"Epoch {epoch}/{args.epochs} | "
              f"train_loss={avg_train_loss:.4f}  val_loss={avg_val_loss:.4f} | {elapsed:.0f}s")

        log_records.append({
            "epoch": epoch, "train_loss": round(avg_train_loss, 6),
            "val_loss": round(avg_val_loss, 6), "elapsed_s": round(elapsed, 1),
        })

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            print(f"  ✓ 最优 QLoRA adapter 已保存 → {ckpt_dir} (val_loss={avg_val_loss:.4f})")

    # ── 保存日志 ──────────────────────────────────────────────────────────
    log_path = log_dir / f"train_{tag}.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_records, f, ensure_ascii=False, indent=2)

    print(f"\n训练完成。最优 val_loss={best_val_loss:.4f}")
    print(f"QLoRA adapter → {ckpt_dir}")
    print(f"训练日志     → {log_path}")
    print(f"\n下一步：python evaluate_sft.py --dataset {args.dataset} --ckpt_dir {ckpt_dir}")


if __name__ == "__main__":
    main()
