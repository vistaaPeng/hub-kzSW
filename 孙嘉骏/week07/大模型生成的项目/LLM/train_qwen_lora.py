"""
任务四：Qwen2-0.5B-Instruct LoRA微调进行NER
使用LoRA对Qwen模型进行参数高效微调
"""
import json
import os
import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from tqdm import tqdm

try:
    from peft import LoraConfig, get_peft_model, TaskType
except ImportError:
    print("请安装peft库: pip install peft")
    exit(1)

# ============ 配置 ============
class Config:
    train_path = "data/peoples_daily/train.json"
    val_path = "data/peoples_daily/validation.json"
    test_path = "data/peoples_daily/test.json"
    label_names_path = "data/peoples_daily/label_names.json"

    model_path = "pretrain_models/Qwen2-0.5B-Instruct"

    max_input_length = 256
    batch_size = 4
    learning_rate = 2e-4
    num_epochs = 3
    warmup_ratio = 0.1
    weight_decay = 0.01
    gradient_accumulation_steps = 4

    # LoRA参数
    lora_r = 16
    lora_alpha = 32
    lora_dropout = 0.1
    lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    output_dir = "LLM/output_qwen_lora"
    seed = 42

config = Config()

def set_seed(seed):
    import numpy as np
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

entity_types = list(set(l[2:] for l in label_names if l != "O"))
print(f"实体类型: {entity_types}")

# ============ 加载数据 ============
def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ============ 获取完整文本 ============
def get_text(text):
    """将text字段转为完整句子。如果是字符列表则拼接，否则直接返回。"""
    if isinstance(text, list):
        return "".join(text)
    return text

# ============ BIO标签转实体 ============
def bio_to_entities(text, labels):
    """将BIO标签序列转换为实体列表。text可以是列表或字符串。"""
    full_text = get_text(text)
    entities = []
    i = 0
    while i < len(labels):
        if labels[i].startswith("B-"):
            entity_type = labels[i][2:]
            start = i
            i += 1
            while i < len(labels) and labels[i] == f"I-{entity_type}":
                i += 1
            end = i
            entity_text = full_text[start:end]
            entities.append({"type": entity_type, "text": entity_text})
        else:
            i += 1
    return entities

def entities_to_str(entities):
    if not entities:
        return "无实体"
    result = []
    for ent in entities:
        result.append(f"{ent['type']}: {ent['text']}")
    return "; ".join(result)

# ============ 构建训练样本 ============
def build_instruction(text):
    return f"""你是一个命名实体识别(NER)专家。请从给定的文本中识别出以下类型的命名实体：
- PER: 人名
- ORG: 组织机构名
- LOC: 地名

请按以下格式输出，每个实体用"类型: 实体文本"表示，多个实体用分号分隔。如果没有识别到实体，输出"无实体"。

文本：{text}

识别结果："""

# ============ 数据集 ============
class NERInstructionDataset(Dataset):
    def __init__(self, data, tokenizer, max_length, is_train=True):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_train = is_train
        self.processed_data = self._process_data()

    def _process_data(self):
        processed = []
        for item in tqdm(self.data, desc="Processing data"):
            raw_text = item["text"]
            text = get_text(raw_text)  # 拼接为完整句子
            labels = item["labels"]
            entities = bio_to_entities(raw_text, labels)
            entities_str = entities_to_str(entities)

            instruction = build_instruction(text)  # 使用完整句子构建指令

            if self.is_train:
                # 训练时使用chat模板
                messages = [
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": entities_str}
                ]
            else:
                messages = [
                    {"role": "user", "content": instruction}
                ]

            processed.append({
                "messages": messages,
                "text": text,  # 存储完整句子
                "gold_labels": labels,
                "entities_str": entities_str
            })
        return processed

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        item = self.processed_data[idx]

        if self.is_train:
            # 使用chat模板编码
            full_text = self.tokenizer.apply_chat_template(
                item["messages"], tokenize=False, add_generation_prompt=False
            )
            # 只编码assistant部分的token作为标签
            input_text = self.tokenizer.apply_chat_template(
                item["messages"][:1], tokenize=False, add_generation_prompt=True
            )

            input_ids = self.tokenizer(full_text, max_length=self.max_length, truncation=True, padding="max_length", return_tensors="pt")

            # 创建标签：只对assistant回复部分计算loss
            input_ids_tensor = input_ids["input_ids"].squeeze()
            attention_mask_tensor = input_ids["attention_mask"].squeeze()

            # 找到assistant回复的起始位置
            input_part = self.tokenizer(input_text, max_length=self.max_length, truncation=True, padding=False)
            input_len = len(input_part["input_ids"])

            labels_tensor = input_ids_tensor.clone()
            # 将input部分和padding部分的标签设为-100
            labels_tensor[:input_len] = -100
            labels_tensor[attention_mask_tensor == 0] = -100

            return {
                "input_ids": input_ids_tensor,
                "attention_mask": attention_mask_tensor,
                "labels": labels_tensor
            }
        else:
            return item

# ============ 解析模型输出 ============
def parse_output(text):
    entities = []
    pattern = r"(PER|ORG|LOC)\s*[:：]\s*([^;；]+)"
    matches = re.findall(pattern, text)
    for match in matches:
        entity_type = match[0]
        entity_text = match[1].strip()
        if entity_text and entity_text != "无实体":
            entities.append({"type": entity_type, "text": entity_text})
    return entities

def entities_to_bio(text, entities):
    labels = ["O"] * len(text)
    for ent in entities:
        entity_text = ent["text"]
        entity_type = ent["type"]
        start = text.find(entity_text)
        while start != -1:
            end = start + len(entity_text)
            overlap = False
            for i in range(start, end):
                if labels[i] != "O":
                    overlap = True
                    break
            if not overlap:
                labels[start] = f"B-{entity_type}"
                for i in range(start + 1, end):
                    labels[i] = f"I-{entity_type}"
            start = text.find(entity_text, start + 1)
    return labels

# ============ 评估函数 ============
def evaluate(model, tokenizer, test_data, device):
    model.eval()
    all_preds = []
    all_labels = []

    for item in tqdm(test_data, desc="Evaluating"):
        text = get_text(item["text"])  # 拼接为完整句子
        gold_labels = item["labels"]

        instruction = build_instruction(text)  # 使用完整句子构建指令
        messages = [{"role": "user", "content": instruction}]
        text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text_input, return_tensors="pt", max_length=config.max_input_length, truncation=True).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
            )

        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)

        pred_entities = parse_output(response)
        pred_labels = entities_to_bio(text, pred_entities)

        min_len = min(len(pred_labels), len(gold_labels))
        all_preds.append(pred_labels[:min_len])
        all_labels.append(gold_labels[:min_len])

    f1 = f1_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)

    return {"f1": f1, "precision": precision, "recall": recall, "report": report}

# ============ 训练函数 ============
def train():
    print("加载Qwen模型和tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )
    if not torch.cuda.is_available():
        model.to(device)

    # 配置LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 加载数据
    train_data = load_data(config.train_path)
    val_data = load_data(config.val_path)
    test_data = load_data(config.test_path)

    print(f"训练集: {len(train_data)}, 验证集: {len(val_data)}, 测试集: {len(test_data)}")

    # 创建数据集
    train_dataset = NERInstructionDataset(train_data, tokenizer, config.max_input_length, is_train=True)
    val_dataset = NERInstructionDataset(val_data, tokenizer, config.max_input_length, is_train=False)
    test_dataset = NERInstructionDataset(test_data, tokenizer, config.max_input_length, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    total_steps = (len(train_loader) // config.gradient_accumulation_steps) * config.num_epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_f1 = 0
    os.makedirs(config.output_dir, exist_ok=True)

    # 训练循环
    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0
        optimizer.zero_grad()

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        for step, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss / config.gradient_accumulation_steps
            total_loss += loss.item() * config.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if (step + 1) % config.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            progress_bar.set_postfix({"loss": f"{loss.item() * config.gradient_accumulation_steps:.4f}"})

        avg_loss = total_loss / len(train_loader)

        # 验证
        val_metrics = evaluate(model, tokenizer, val_data, device)
        print(f"
Epoch {epoch+1}: Train Loss={avg_loss:.4f}, Val F1={val_metrics['f1']:.4f}, "
              f"Precision={val_metrics['precision']:.4f}, Recall={val_metrics['recall']:.4f}")

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            model.save_pretrained(os.path.join(config.output_dir, "best_model"))
            tokenizer.save_pretrained(os.path.join(config.output_dir, "best_model"))
            print(f"  -> 保存最佳模型 (F1={best_f1:.4f})")

    # 测试集评估
    print("
加载最佳模型进行测试集评估...")
    from peft import PeftModel
    base_model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )
    best_model = PeftModel.from_pretrained(base_model, os.path.join(config.output_dir, "best_model"))
    if not torch.cuda.is_available():
        best_model.to(device)

    test_metrics = evaluate(best_model, tokenizer, test_data, device)
    print(f"
测试集结果:")
    print(f"  F1: {test_metrics['f1']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall: {test_metrics['recall']:.4f}")

    # 详细报告
    all_preds = []
    all_labels_list = []
    for item in tqdm(test_data, desc="Final evaluation"):
        text = get_text(item["text"])  # 拼接为完整句子
        gold_labels = item["labels"]
        instruction = build_instruction(text)  # 使用完整句子构建指令
        messages = [{"role": "user", "content": instruction}]
        text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text_input, return_tensors="pt", max_length=config.max_input_length, truncation=True).to(device)
        with torch.no_grad():
            outputs = best_model.generate(**inputs, max_new_tokens=256, do_sample=False)
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        pred_entities = parse_output(response)
        pred_labels = entities_to_bio(text, pred_entities)
        min_len = min(len(pred_labels), len(gold_labels))
        all_preds.append(pred_labels[:min_len])
        all_labels_list.append(gold_labels[:min_len])

    print(classification_report(all_labels_list, all_preds, zero_division=0))

    results = {
        "model": "Qwen-LoRA",
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
