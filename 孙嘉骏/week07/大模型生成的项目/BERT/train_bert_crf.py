"""
任务二：BERT+CRF微调进行NER
使用bert-base-chinese预训练模型 + CRF层进行序列标注
"""
import json
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertTokenizerFast, BertModel, get_linear_schedule_with_warmup
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from tqdm import tqdm

# ============ 配置 ============
class Config:
    train_path = "data/peoples_daily/train.json"
    val_path = "data/peoples_daily/validation.json"
    test_path = "data/peoples_daily/test.json"
    label_names_path = "data/peoples_daily/label_names.json"

    model_path = "pretrain_models/bert-base-chinese"

    max_length = 128
    batch_size = 32
    learning_rate = 3e-5
    crf_learning_rate = 1e-3
    num_epochs = 5
    warmup_ratio = 0.1
    weight_decay = 0.01

    output_dir = "BERT/output_bert_crf"
    seed = 42

config = Config()

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(config.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============ CRF层 ============
class CRF(nn.Module):
    """条件随机场层"""
    def __init__(self, num_tags, batch_first=True):
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first
        # 转移矩阵 transitions[i][j] 表示从标签j转移到标签i的分数
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        # 起始和结束转移分数
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.end_transitions = nn.Parameter(torch.randn(num_tags))

    def forward(self, emissions, tags, mask=None):
        """计算CRF的负对数似然损失"""
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.uint8)

        if not self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            mask = mask.transpose(0, 1)

        # 计算黄金路径的分数
        gold_score = self._score_sentence(emissions, tags, mask)
        # 计算所有路径的log-sum-exp
        forward_score = self._forward_algorithm(emissions, mask)

        # 负对数似然
        loss = (forward_score - gold_score).mean()
        return loss

    def _forward_algorithm(self, emissions, mask):
        """前向算法计算所有路径的log-sum-exp"""
        batch_size, seq_len, num_tags = emissions.shape

        # 初始化: alpha[i] = start_transitions[i] + emissions[0][i]
        alpha = self.start_transitions.unsqueeze(0) + emissions[:, 0]  # [batch, num_tags]

        for i in range(1, seq_len):
            # alpha_new[j] = log_sum_exp(alpha[i] + transitions[j][i]) + emissions[t][j]
            emit_score = emissions[:, i].unsqueeze(1)  # [batch, 1, num_tags]
            trans_score = self.transitions.unsqueeze(0)  # [1, num_tags, num_tags]
            alpha_expand = alpha.unsqueeze(2)  # [batch, num_tags, 1]

            scores = alpha_expand + trans_score + emit_score  # [batch, num_tags, num_tags]
            new_alpha = torch.logsumexp(scores, dim=1)  # [batch, num_tags]

            # 只更新mask为1的位置
            mask_i = mask[:, i].unsqueeze(1).float()
            alpha = new_alpha * mask_i + alpha * (1 - mask_i)

        # 终止分数
        alpha = alpha + self.end_transitions.unsqueeze(0)
        return torch.logsumexp(alpha, dim=1)  # [batch]

    def _score_sentence(self, emissions, tags, mask):
        """计算给定标签序列的分数"""
        batch_size, seq_len, num_tags = emissions.shape

        # 起始分数
        score = self.start_transitions[tags[:, 0]]  # [batch]
        score += emissions[:, 0].gather(1, tags[:, 0].unsqueeze(1)).squeeze(1)

        for i in range(1, seq_len):
            mask_i = mask[:, i].float()
            # 转移分数
            trans = self.transitions[tags[:, i], tags[:, i-1]]
            # 发射分数
            emit = emissions[:, i].gather(1, tags[:, i].unsqueeze(1)).squeeze(1)

            score += (trans + emit) * mask_i

        # 终止分数 - 找到每个序列的最后一个有效标签
        seq_lengths = mask.long().sum(dim=1) - 1
        last_tags = tags.gather(1, seq_lengths.unsqueeze(1)).squeeze(1)
        score += self.end_transitions[last_tags]

        return score

    def decode(self, emissions, mask=None):
        """维特比解码，找到最优路径"""
        if mask is None:
            mask = torch.ones(emissions.shape[:2], dtype=torch.uint8, device=emissions.device)

        if not self.batch_first:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)

        batch_size, seq_len, num_tags = emissions.shape

        # 初始化
        alpha = self.start_transitions.unsqueeze(0) + emissions[:, 0]  # [batch, num_tags]

        # 存储回溯指针
        backpointers = []

        for i in range(1, seq_len):
            emit_score = emissions[:, i].unsqueeze(1)  # [batch, 1, num_tags]
            trans_score = self.transitions.unsqueeze(0)  # [1, num_tags, num_tags]
            alpha_expand = alpha.unsqueeze(2)  # [batch, num_tags, 1]

            scores = alpha_expand + trans_score + emit_score  # [batch, num_tags, num_tags]
            max_scores, max_indices = scores.max(dim=1)  # [batch, num_tags]

            # 只更新mask为1的位置
            mask_i = mask[:, i].unsqueeze(1).float()
            alpha = max_scores * mask_i + (alpha + emissions[:, i].unsqueeze(1).expand_as(alpha)) * (1 - mask_i)

            # 回溯指针也需要mask处理
            backpointers.append(max_indices)

        # 终止
        alpha += self.end_transitions.unsqueeze(0)

        # 回溯
        best_paths = []
        seq_lengths = mask.long().sum(dim=1)

        for b in range(batch_size):
            seq_len_b = seq_lengths[b].item()
            # 找到最后一步的最佳标签
            best_tag = alpha[b].argmax().item()
            best_path = [best_tag]

            # 回溯
            for ptr in reversed(backpointers[:seq_len_b-1]):
                best_tag = ptr[b, best_tag].item()
                best_path.append(best_tag)

            best_path.reverse()
            best_paths.append(best_path)

        return best_paths


# ============ BERT+CRF模型 ============
class BertCRF(nn.Module):
    def __init__(self, bert_model_path, num_labels):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_path)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        sequence_output = bert_outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        emissions = self.classifier(sequence_output)

        if labels is not None:
            # 创建mask: labels != -100 的位置为1
            crf_mask = (labels != -100).float()
            # 将-100替换为0（CRF不需要-100标记）
            crf_labels = labels.clone()
            crf_labels[crf_labels == -100] = 0

            loss = self.crf(emissions, crf_labels, mask=crf_mask.bool())
            return loss, emissions
        else:
            return emissions

    def decode(self, emissions, mask=None):
        return self.crf.decode(emissions, mask)


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

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True
        )

        aligned_labels = []
        offset_mapping = encoding.pop("offset_mapping")

        for i, (start, end) in enumerate(offset_mapping):
            if start == 0 and end == 0:
                aligned_labels.append(-100)
            else:
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

            emissions = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )

            # CRF解码
            crf_mask = (labels != -100)
            pred_paths = model.decode(emissions, crf_mask)

            for i in range(len(pred_paths)):
                pred_seq = [id2label[p] for p in pred_paths[i]]
                label_indices = labels[i][crf_mask[i]].tolist()
                label_seq = [id2label[l] for l in label_indices]
                all_preds.append(pred_seq)
                all_labels.append(label_seq)

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
    tokenizer = BertTokenizerFast.from_pretrained(config.model_path)
    model = BertCRF(config.model_path, num_labels)
    model.to(device)

    train_data = load_data(config.train_path)
    val_data = load_data(config.val_path)
    test_data = load_data(config.test_path)

    print(f"训练集: {len(train_data)}, 验证集: {len(val_data)}, 测试集: {len(test_data)}")

    train_dataset = NERDataset(train_data, tokenizer, label2id, config.max_length)
    val_dataset = NERDataset(val_data, tokenizer, label2id, config.max_length)
    test_dataset = NERDataset(test_data, tokenizer, label2id, config.max_length)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    # 分组优化器: BERT参数和CRF参数使用不同学习率
    bert_params = list(model.bert.parameters())
    crf_params = list(model.crf.parameters()) + list(model.classifier.parameters())

    optimizer = AdamW([
        {"params": bert_params, "lr": config.learning_rate},
        {"params": crf_params, "lr": config.crf_learning_rate}
    ], weight_decay=config.weight_decay)

    total_steps = len(train_loader) * config.num_epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

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

            loss, emissions = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels
            )

            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(train_loader)

        val_metrics = evaluate(model, val_loader, id2label, device)
        print(f"\nEpoch {epoch+1}: Train Loss={avg_loss:.4f}, Val F1={val_metrics['f1']:.4f}, "
              f"Precision={val_metrics['precision']:.4f}, Recall={val_metrics['recall']:.4f}")

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            # 保存模型
            torch.save(model.state_dict(), os.path.join(config.output_dir, "best_model.pt"))
            tokenizer.save_pretrained(os.path.join(config.output_dir, "best_tokenizer"))
            print(f"  -> 保存最佳模型 (F1={best_f1:.4f})")

    # 加载最佳模型进行测试
    print("\n加载最佳模型进行测试集评估...")
    best_model = BertCRF(config.model_path, num_labels)
    best_model.load_state_dict(torch.load(os.path.join(config.output_dir, "best_model.pt"), map_location=device))
    best_model.to(device)

    test_metrics = evaluate(best_model, test_loader, id2label, device)
    print(f"\n测试集结果:")
    print(f"  F1: {test_metrics['f1']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall: {test_metrics['recall']:.4f}")

    # 详细报告
    best_model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)
            emissions = best_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            crf_mask = (labels != -100)
            pred_paths = best_model.decode(emissions, crf_mask)
            for i in range(len(pred_paths)):
                pred_seq = [id2label[p] for p in pred_paths[i]]
                label_indices = labels[i][crf_mask[i]].tolist()
                label_seq = [id2label[l] for l in label_indices]
                all_preds.append(pred_seq)
                all_labels.append(label_seq)

    print(classification_report(all_labels, all_preds, zero_division=0))

    results = {
        "model": "BERT+CRF",
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
