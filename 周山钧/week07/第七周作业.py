"""
第七周作业：基于人民日报 NER 数据集的序列标注模型训练
=====================================================

任务：使用 peoples_daily 数据集，实现 BERT + Linear 和 BERT + CRF 两种 NER 模型，
      对比两者在 entity-level F1 和非法序列上的差异。

数据集：人民日报 NER（PEOPLES_DAILY）
  - 训练集：20,864 条
  - 验证集：2,318 条
  - 测试集：4,636 条
  - 实体类型：PER（人名）、ORG（组织机构）、LOC（地点）
  - 标签体系：O / B-PER / I-PER / B-ORG / I-ORG / B-LOC / I-LOC（共7类）

使用方式：
  python 第七周作业.py                          # 训练 BERT+Linear + BERT+CRF
  python 第七周作业.py --epochs 5                # 自定义训练轮数
  python 第七周作业.py --bert_path <模型路径>    # 指定本地 BERT 模型路径
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertModel, BertTokenizer, BertTokenizerFast, get_linear_schedule_with_warmup
from tqdm import tqdm

# ═══════════════════════════════════════════════════════════════════════════════
# 路径配置
# ═══════════════════════════════════════════════════════════════════════════════
SCRIPT_DIR = Path(__file__).parent  # week07/

# 人民日报 NER 数据路径（week07 本地数据）
PEOPLES_DAILY_DIR = SCRIPT_DIR / "data" / "peoples_daily"

# 预训练模型路径（本地 BERT 模型）
DEFAULT_BERT_PATH = Path.home() / "Desktop/ai视频/第四周/week4语言模型/bert-base-chinese"

# 输出目录
OUTPUT_DIR = SCRIPT_DIR / "outputs"
CKPT_DIR = OUTPUT_DIR / "checkpoints"
LOG_DIR = OUTPUT_DIR / "logs"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 标签体系
# ═══════════════════════════════════════════════════════════════════════════════
LABELS = [
    "O",
    "B-PER", "I-PER",
    "B-ORG", "I-ORG",
    "B-LOC", "I-LOC",
]
LABEL2ID = {lbl: i for i, lbl in enumerate(LABELS)}
ID2LABEL = {i: lbl for lbl, i in LABEL2ID.items()}
NUM_LABELS = len(LABELS)

ENTITY_TYPES = ["PER", "ORG", "LOC"]


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset — 人民日报 NER 数据加载
# ═══════════════════════════════════════════════════════════════════════════════

class PeoplesDailyDataset(Dataset):
    """
    人民日报 NER 数据集。

    数据格式（与 cluener2020 不同，已是 BIO 格式）：
      {"tokens": ["海", "钓", ...], "ner_tags": ["O", "O", ...]}

    tokens 是已经分好的单字列表，ner_tags 是逐字对齐的 BIO 标签。
    直接用 BERT tokenizer 对字符列表编码，is_split_into_words=True。
    """

    def __init__(self, records: list, tokenizer: BertTokenizer, max_length: int = 128):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        row = self.records[idx]
        chars: list[str] = row["tokens"]       # 单字列表
        tags: list[str] = row["ner_tags"]      # BIO 标签列表（长度 = len(chars)）

        # 标签转 id
        tag_ids = [LABEL2ID.get(t, 0) for t in tags]

        # BERT tokenizer 逐字编码
        encoding = self.tokenizer(
            chars,
            is_split_into_words=True,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # 子词对齐：word_ids() 返回每个 token 对应的原始字索引
        word_ids = encoding.word_ids(batch_index=0)
        aligned_labels = []
        prev_word_idx = None
        for wid in word_ids:
            if wid is None:
                # [CLS]、[SEP]、[PAD]
                aligned_labels.append(-100)
            elif wid != prev_word_idx:
                # 该字的第一个子词，保留原始 BIO 标签
                if wid < len(tag_ids):
                    aligned_labels.append(tag_ids[wid])
                else:
                    aligned_labels.append(-100)
                prev_word_idx = wid
            else:
                # 同一字的后续子词（中文一般不会出现，但保留正确处理）
                aligned_labels.append(-100)

        labels_tensor = torch.tensor(aligned_labels, dtype=torch.long)

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding["token_type_ids"].squeeze(0),
            "labels": labels_tensor,
        }


def load_peoples_daily(data_dir: Path = None):
    """加载人民日报 NER 三个 split，返回 train/validation/test 记录的 list。"""
    d = data_dir or PEOPLES_DAILY_DIR

    def _load(split: str) -> list:
        path = d / f"{split}.json"
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    train = _load("train")
    val = _load("validation")
    test = _load("test")
    print(f"人民日报 NER 数据集：训练={len(train)}，验证={len(val)}，测试={len(test)}")
    return train, val, test


def build_dataloaders(
    tokenizer: BertTokenizer,
    batch_size: int = 32,
    max_length: int = 128,
    data_dir: Path = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_records, val_records, test_records = load_peoples_daily(data_dir)
    train_ds = PeoplesDailyDataset(train_records, tokenizer, max_length)
    val_ds = PeoplesDailyDataset(val_records, tokenizer, max_length)
    test_ds = PeoplesDailyDataset(test_records, tokenizer, max_length)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader


# ═══════════════════════════════════════════════════════════════════════════════
# 模型定义
# ═══════════════════════════════════════════════════════════════════════════════

class BertNER(nn.Module):
    """BERT + 线性分类头，逐 token 独立预测 BIO 标签。"""

    def __init__(self, bert_path: str, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        seq_output = outputs.last_hidden_state  # (B, L, H)
        logits = self.classifier(self.dropout(seq_output))  # (B, L, num_labels)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.num_labels),
                labels.view(-1),
                ignore_index=-100,
            )
        return logits, loss


class BertCRFNER(nn.Module):
    """BERT + CRF 层，Viterbi 全局最优解码，保证输出合法 BIO 序列。"""

    def __init__(self, bert_path: str, num_labels: int, dropout: float = 0.1):
        super().__init__()
        from torchcrf import CRF
        self.bert = BertModel.from_pretrained(bert_path)
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)
        self.num_labels = num_labels

    def _get_emissions(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        seq_output = outputs.last_hidden_state
        return self.classifier(self.dropout(seq_output))  # (B, L, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        emissions = self._get_emissions(input_ids, attention_mask, token_type_ids)
        mask = attention_mask.bool()

        loss = None
        if labels is not None:
            labels_crf = labels.clone()
            labels_crf[labels_crf == -100] = 0
            loss = -self.crf(emissions, labels_crf, mask=mask, reduction="mean")

        return emissions, loss

    def decode(self, input_ids, attention_mask, token_type_ids):
        emissions = self._get_emissions(input_ids, attention_mask, token_type_ids)
        mask = attention_mask.bool()
        return self.crf.decode(emissions, mask=mask)


def build_model(use_crf: bool, bert_path: str, num_labels: int, dropout: float = 0.1):
    model_cls = BertCRFNER if use_crf else BertNER
    model = model_cls(bert_path=bert_path, num_labels=num_labels, dropout=dropout)
    total = sum(p.numel() for p in model.parameters())
    name = "BERT + CRF" if use_crf else "BERT + Linear"
    print(f"模型：{name} | 标签数：{num_labels} | 参数总量：{total / 1e6:.1f}M")
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# 评估与训练
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate(model, loader, device, use_crf):
    """
    在 loader 上推理，返回：
      - avg_loss: 平均 loss
      - entity_f1: seqeval entity-level F1
      - all_preds: 预测的标签字符串序列列表
      - all_golds: 真实的标签字符串序列列表
    """
    from seqeval.metrics import f1_score as seqeval_f1

    model.eval()
    total_loss = 0.0
    all_preds, all_golds = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)

            if use_crf:
                emissions, loss = model(input_ids, attention_mask, token_type_ids, labels)
                pred_ids_list = model.decode(input_ids, attention_mask, token_type_ids)
            else:
                logits, loss = model(input_ids, attention_mask, token_type_ids, labels)
                pred_ids_list = logits.argmax(dim=-1).tolist()

            if loss is not None:
                total_loss += loss.item()

            labels_np = labels.cpu().tolist()
            for i in range(len(input_ids)):
                gold_seq, pred_seq = [], []
                token_labels = labels_np[i]
                for j, gold_id in enumerate(token_labels):
                    if gold_id == -100:
                        continue
                    gold_seq.append(ID2LABEL[gold_id])
                    if use_crf:
                        pred_seq.append(ID2LABEL.get(pred_ids_list[i][j] if j < len(pred_ids_list[i]) else 0, "O"))
                    else:
                        pred_seq.append(ID2LABEL.get(pred_ids_list[i][j], "O"))
                all_golds.append(gold_seq)
                all_preds.append(pred_seq)

    avg_loss = total_loss / len(loader)
    entity_f1 = seqeval_f1(all_golds, all_preds)
    return avg_loss, entity_f1, all_preds, all_golds


def count_illegal_sequences(pred_seqs: list[list[str]]) -> dict:
    """统计 BIO 非法序列。"""
    stats = {"illegal_start": 0, "illegal_transition": 0, "total": len(pred_seqs)}
    for seq in pred_seqs:
        if not seq:
            continue
        if seq[0].startswith("I-"):
            stats["illegal_start"] += 1
        for i in range(1, len(seq)):
            prev, curr = seq[i - 1], seq[i]
            if curr.startswith("I-"):
                curr_type = curr[2:]
                if prev == "O":
                    stats["illegal_transition"] += 1
                elif prev.startswith("B-") or prev.startswith("I-"):
                    if prev[2:] != curr_type:
                        stats["illegal_transition"] += 1
    stats["total_illegal"] = stats["illegal_start"] + stats["illegal_transition"]
    return stats


def train_one_epoch(model, loader, optimizer, scheduler, device, grad_accum):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    for step, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels = batch["labels"].to(device)

        _, loss = model(input_ids, attention_mask, token_type_ids, labels)
        (loss / grad_accum).backward()
        total_loss += loss.item()

        if (step + 1) % grad_accum == 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
    # 处理最后不足 grad_accum 的批次
    if len(loader) % grad_accum != 0:
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    return total_loss / len(loader)


def train(args):
    """训练主流程：分别训练 BERT+Linear 和 BERT+CRF。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备：{device}")

    # 确定 BERT 路径
    bert_path = str(args.bert_path) if args.bert_path.exists() else "bert-base-chinese"
    print(f"BERT 模型：{bert_path}")

    tokenizer = BertTokenizerFast.from_pretrained(bert_path)
    train_loader, val_loader, test_loader = build_dataloaders(
        tokenizer, args.batch_size, args.max_length, args.data_dir
    )

    results = {}

    for use_crf in [False, True]:
        run_tag = "crf" if use_crf else "linear"
        model_name = "BERT + CRF" if use_crf else "BERT + Linear"
        print(f"\n{'=' * 60}")
        print(f"开始训练：{model_name}")
        print(f"{'=' * 60}")

        model = build_model(use_crf, bert_path, NUM_LABELS, args.dropout).to(device)

        # 分层学习率
        bert_params = list(model.bert.parameters())
        head_params = [p for n, p in model.named_parameters() if "bert" not in n]
        optimizer = AdamW(
            [
                {"params": bert_params, "lr": args.lr},
                {"params": head_params, "lr": args.lr * args.head_lr_mult},
            ],
            weight_decay=0.01,
        )

        total_steps = len(train_loader) * args.epochs // args.grad_accum
        warmup_steps = int(total_steps * args.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
        print(f"训练步数：{total_steps}，预热步数：{warmup_steps}")

        best_f1 = 0.0
        ckpt_path = CKPT_DIR / f"pd_best_{run_tag}.pt"
        log_records = []

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            train_loss = train_one_epoch(
                model, train_loader, optimizer, scheduler, device, args.grad_accum
            )
            val_loss, val_f1, _, _ = evaluate(model, val_loader, device, use_crf)
            elapsed = time.time() - t0

            print(
                f"Epoch {epoch}/{args.epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"val_entity_f1={val_f1:.4f} | "
                f"time={elapsed:.0f}s"
            )

            log_records.append({
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "val_loss": round(val_loss, 6),
                "val_entity_f1": round(val_f1, 6),
                "elapsed_s": round(elapsed, 1),
            })

            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save(
                    {"epoch": epoch, "use_crf": use_crf, "state_dict": model.state_dict(),
                     "val_entity_f1": val_f1, "label2id": LABEL2ID, "id2label": ID2LABEL},
                    ckpt_path,
                )
                print(f"  ★ 最优 F1={val_f1:.4f}，已保存 → {ckpt_path}")

        # 保存训练日志
        log_path = LOG_DIR / f"pd_train_{run_tag}.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log_records, f, ensure_ascii=False, indent=2)

        # ── 最终评估（验证集） ──
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        _, val_f1, val_preds, val_golds = evaluate(model, val_loader, device, use_crf)

        illegal = count_illegal_sequences(val_preds)

        # 逐类型 F1
        from seqeval.metrics import classification_report

        print(f"\n{'─' * 60}")
        print(f"【{model_name} 最终评估 — 验证集】")
        print(f"  Entity F1: {val_f1:.4f}")
        print(f"  非法序列: {illegal['total_illegal']} / {illegal['total']} 条")
        if illegal["total_illegal"] > 0:
            print(f"    - 非法开头 (I-X 开头): {illegal['illegal_start']} 条")
            print(f"    - 非法转移 (类型不一致): {illegal['illegal_transition']} 条")
        else:
            print(f"    ✓ 全部合法！")
        print(f"\n逐类型 F1：")
        print(classification_report(val_golds, val_preds, digits=4))

        results[run_tag] = {
            "model": model_name,
            "val_f1": val_f1,
            "illegal": illegal,
        }

        # 保存评估日志
        eval_log = {
            "model": model_name,
            "split": "validation",
            "val_f1": round(val_f1, 6),
            "illegal": illegal,
        }
        eval_path = LOG_DIR / f"pd_eval_{run_tag}.json"
        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(eval_log, f, ensure_ascii=False, indent=2)

    # ── 最终对比 ──
    print(f"\n{'=' * 60}")
    print(f"人民日报 NER — 最终对比")
    print(f"{'=' * 60}")
    print(f"{'模型':<20} {'Val F1':>10} {'非法序列':>10}")
    print(f"{'-' * 42}")
    for tag, r in results.items():
        name = r["model"]
        f1 = r["val_f1"]
        il = r["illegal"]["total_illegal"]
        print(f"{name:<20} {f1:>10.4f} {il:>10}")


# ═══════════════════════════════════════════════════════════════════════════════
# 命令行参数
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="人民日报 NER 序列标注作业")
    parser.add_argument("--bert_path", type=Path, default=DEFAULT_BERT_PATH,
                        help="BERT 模型路径（本地目录或 HuggingFace 模型名）")
    parser.add_argument("--data_dir", type=Path, default=PEOPLES_DAILY_DIR,
                        help="人民日报 NER 数据目录")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-5, help="BERT 层学习率")
    parser.add_argument("--head_lr_mult", type=float, default=5.0, help="分类头学习率倍数")
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    args = parse_args()
    train(args)
