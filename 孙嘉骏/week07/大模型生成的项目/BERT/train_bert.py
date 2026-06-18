"""
任务一：BERT微调进行NER
使用bert-base-chinese预训练模型进行序列标注
"""
import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertTokenizerFast, BertForTokenClassification, get_linear_schedule_with_warmup
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from tqdm import tqdm

# ============ 配置 ============
class Config:
    # 数据路径
    train_path = "data/peoples_daily/train.json"
    val_path = "data/peoples_daily/validation.json"
    test_path = "data/peoples_daily/test.json"
    label_names_path = "data/peoples_daily/label_names.json"

    # 模型路径
    model_path = "pretrain_models/bert-base-chinese"

    # 训练参数
    max_length = 128
    batch_size = 32
    learning_rate = 3e-5
    num_epochs = 5
    warmup_ratio = 0.1
    weight_decay = 0.01

    # 输出路径
    output_dir = "BERT/output_bert"
    seed = 42

config = Config()

# 设置随机种子
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(config.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============ 加载标签 ============
with open(config.label_names_path, "r", encoding="utf-8") as f:
    label_names = json.load(f)

label2id = {label: i for i, label in enumerate(label_names)}
id2label = {i: label for i, label in enumerate(label_names)}
num_labels = len(label_names)
print(f"标签数量: {num_labels}, 标签列表: {label_names}")

# ============ 加载数据 ============
def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ============ 数据集 ============
class NERDataset(Dataset):
    def __init__(self, data, tokenizer, label2id, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["text"]
        labels = item["labels"]

        # 使用tokenizer进行分词，获取字符级别的对齐
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True
        )

        # 将标签对齐到token级别
        aligned_labels = []
        offset_mapping = encoding.pop("offset_mapping")

        for i, (start, end) in enumerate(offset_mapping):
            if start == 0 and end == 0:  # [CLS], [SEP], [PAD]
                aligned_labels.append(-100)
            else:
                # 对于中文字符，每个字通常是一个token
                # 使用第一个字符的标签
                if start < len(labels):
                    label = labels[start]
                    aligned_labels.append(self.label2id.get(label, 0))
                else:
                    aligned_labels.append(-100)

        encoding["labels"] = aligned_labels
        return {k: torch.tensor(v) for k, v in encoding.items()}

# ============ 评估函数 ============
def evaluate(model, dataloader, id2label, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )

            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)

            # 将预测和标签转换为seqeval格式
            for i in range(len(preds)):
                pred_seq = []
                label_seq = []
                for j in range(len(preds[i])):
                    if labels[i][j] != -100:
                        pred_seq.append(id2label[preds[i][j].item()])
                        label_seq.append(id2label[labels[i][j].item()])
                all_preds.append(pred_seq)
                all_labels.append(label_seq)

    # 使用实体级别的评价指标
    report = classification_report(all_labels, all_preds, output_dict=True)
    f1 = f1_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)

    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "report": report
    }

# ============ 训练函数 ============
def train():
    # 加载tokenizer和模型
    tokenizer = BertTokenizerFast.from_pretrained(config.model_path)
    model = BertForTokenClassification.from_pretrained(
        config.model_path,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )
    model.to(device)

    # 加载数据
    train_data = load_data(config.train_path)
    val_data = load_data(config.val_path)
    test_data = load_data(config.test_path)

    print(f"训练集: {len(train_data)}, 验证集: {len(val_data)}, 测试集: {len(test_data)}")

    # 创建数据集和数据加载器
    train_dataset = NERDataset(train_data, tokenizer, label2id, config.max_length)
    val_dataset = NERDataset(val_data, tokenizer, label2id, config.max_length)
    test_dataset = NERDataset(test_data, tokenizer, label2id, config.max_length)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    # 优化器和学习率调度器
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    total_steps = len(train_loader) * config.num_epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # 训练循环
    best_f1 = 0
    os.makedirs(config.output_dir, exist_ok=True)

    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels
            )

            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(train_loader)

        # 验证
        val_metrics = evaluate(model, val_loader, id2label, device)
        print(f"\nEpoch {epoch+1}: Train Loss={avg_loss:.4f}, Val F1={val_metrics['f1']:.4f}, "
              f"Precision={val_metrics['precision']:.4f}, Recall={val_metrics['recall']:.4f}")

        # 保存最佳模型
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            model.save_pretrained(os.path.join(config.output_dir, "best_model"))
            tokenizer.save_pretrained(os.path.join(config.output_dir, "best_model"))
            print(f"  -> 保存最佳模型 (F1={best_f1:.4f})")

    # 加载最佳模型进行测试
    print("\n加载最佳模型进行测试集评估...")
    best_model = BertForTokenClassification.from_pretrained(os.path.join(config.output_dir, "best_model"))
    best_model.to(device)

    test_metrics = evaluate(best_model, test_loader, id2label, device)
    print(f"\n测试集结果:")
    print(f"  F1: {test_metrics['f1']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall: {test_metrics['recall']:.4f}")
    print(f"\n详细分类报告:")
    report_str = classification_report(
        [[id2label[l] for l in seq if l != -100] for seq in [[]]],
        [[id2label[p] for p in seq] for seq in [[]]],
        zero_division=0
    )
    # 重新生成完整报告
    best_model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)
            outputs = best_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            preds = torch.argmax(outputs.logits, dim=-1)
            for i in range(len(preds)):
                pred_seq = []
                label_seq = []
                for j in range(len(preds[i])):
                    if labels[i][j] != -100:
                        pred_seq.append(id2label[preds[i][j].item()])
                        label_seq.append(id2label[labels[i][j].item()])
                all_preds.append(pred_seq)
                all_labels.append(label_seq)

    print(classification_report(all_labels, all_preds, zero_division=0))

    # 保存结果
    results = {
        "model": "BERT",
        "test_f1": test_metrics["f1"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "report": test_metrics["report"]
    }
    with open(os.path.join(config.output_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return results

if __name__ == "__main__":
    train()
