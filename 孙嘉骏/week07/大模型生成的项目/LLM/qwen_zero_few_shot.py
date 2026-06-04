"""
任务三：Qwen2-0.5B-Instruct Zero Shot 和 Few Shot 进行NER
使用Qwen预训练模型进行零样本和少样本NER
"""
import json
import os
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from tqdm import tqdm

# ============ 配置 ============
class Config:
    test_path = "data/peoples_daily/test.json"
    train_path = "data/peoples_daily/train.json"
    label_names_path = "data/peoples_daily/label_names.json"

    model_path = "pretrain_models/Qwen2-0.5B-Instruct"

    max_input_length = 512
    max_new_tokens = 512
    batch_size = 1

    # Few shot参数
    num_few_shot = 5  # few shot示例数量

    output_dir = "LLM/output_qwen_prompt"
    seed = 42

config = Config()
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

# ============ 将BIO标签转换为实体列表 ============
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
            entities.append({"type": entity_type, "text": entity_text, "start": start, "end": end})
        else:
            i += 1
    return entities

# ============ 将实体列表转换为字符串格式 ============
def entities_to_str(entities):
    """将实体列表转换为字符串格式，用于prompt"""
    if not entities:
        return "无实体"
    result = []
    for ent in entities:
        result.append(f"{ent["type"]}: {ent["text"]}")
    return "; ".join(result)

# ============ 构建Prompt ============
def build_zero_shot_prompt(text):
    """构建零样本prompt"""
    prompt = f"""你是一个命名实体识别(NER)专家。请从给定的文本中识别出以下类型的命名实体：
- PER: 人名
- ORG: 组织机构名
- LOC: 地名

请按以下格式输出，每个实体用"类型: 实体文本"表示，多个实体用分号分隔。如果没有识别到实体，输出"无实体"。

文本：{text}

识别结果："""
    return prompt

def build_few_shot_prompt(text, examples):
    """构建少样本prompt"""
    example_str = ""
    for ex in examples:
        example_str += f"\n文本：{ex["text"]}\n识别结果：{ex["entities_str"]}\n"

    prompt = f"""你是一个命名实体识别(NER)专家。请从给定的文本中识别出以下类型的命名实体：
- PER: 人名
- ORG: 组织机构名
- LOC: 地名

请按以下格式输出，每个实体用"类型: 实体文本"表示，多个实体用分号分隔。如果没有识别到实体，输出"无实体"。

以下是一些示例：
{example_str}
文本：{text}
识别结果："""
    return prompt

# ============ 解析模型输出 ============
def parse_output(text):
    """解析模型输出，提取实体"""
    entities = []
    # 匹配 "类型: 实体" 格式
    pattern = r"(PER|ORG|LOC)\s*[:：]\s*([^;；\n]+)"
    matches = re.findall(pattern, text)
    for match in matches:
        entity_type = match[0]
        entity_text = match[1].strip()
        if entity_text and entity_text != "无实体":
            entities.append({"type": entity_type, "text": entity_text})
    return entities

# ============ 将实体列表转换为BIO标签 ============
def entities_to_bio(text, entities, label2id):
    """将实体列表转换为BIO标签序列"""
    labels = ["O"] * len(text)
    for ent in entities:
        entity_text = ent["text"]
        entity_type = ent["type"]
        # 在文本中查找实体位置
        start = text.find(entity_text)
        while start != -1:
            end = start + len(entity_text)
            # 检查是否与已有实体重叠
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
def evaluate_ner(model, tokenizer, test_data, mode="zero_shot", few_shot_examples=None):
    """评估NER性能"""
    model.eval()
    all_preds = []
    all_labels = []

    label2id = {label: i for i, label in enumerate(label_names)}

    for item in tqdm(test_data, desc=f"Evaluating {mode}"):
        raw_text = item["text"]
        text = get_text(raw_text)  # 拼接为完整句子
        gold_labels = item["labels"]

        # 构建prompt（使用完整句子）
        if mode == "zero_shot":
            prompt = build_zero_shot_prompt(text)
        else:
            prompt = build_few_shot_prompt(text, few_shot_examples)

        # 生成
        messages = [{"role": "user", "content": prompt}]
        text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text_input, return_tensors="pt", max_length=config.max_input_length, truncation=True).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
            )

        # 解码输出
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # 解析实体
        pred_entities = parse_output(response)

        # 转换为BIO标签（使用完整句子）
        pred_labels = entities_to_bio(text, pred_entities, label2id)

        # 对齐长度
        min_len = min(len(pred_labels), len(gold_labels))
        pred_seq = pred_labels[:min_len]
        label_seq = gold_labels[:min_len]

        all_preds.append(pred_seq)
        all_labels.append(label_seq)

    # 计算指标
    f1 = f1_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)

    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "report": report
    }

# ============ 主函数 ============
def main():
    # 加载模型
    print("加载Qwen模型...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )
    if not torch.cuda.is_available():
        model.to(device)
    model.eval()

    # 加载测试数据
    test_data = load_data(config.test_path)
    print(f"测试集样本数: {len(test_data)}")

    # 为了快速评估，可以限制测试样本数量（取消注释下行即可）
    # test_data = test_data[:200]

    os.makedirs(config.output_dir, exist_ok=True)

    # ============ Zero Shot 评估 ============
    print("\n" + "=" * 60)
    print("Zero Shot 评估")
    print("=" * 60)
    zero_shot_metrics = evaluate_ner(model, tokenizer, test_data, mode="zero_shot")
    print(f"\nZero Shot 结果:")
    print(f"  F1: {zero_shot_metrics['f1']:.4f}")
    print(f"  Precision: {zero_shot_metrics['precision']:.4f}")
    print(f"  Recall: {zero_shot_metrics['recall']:.4f}")

    # 保存结果
    zero_results = {
        "model": "Qwen-ZeroShot",
        "test_f1": zero_shot_metrics["f1"],
        "test_precision": zero_shot_metrics["precision"],
        "test_recall": zero_shot_metrics["recall"],
        "report": zero_shot_metrics["report"]
    }
    with open(os.path.join(config.output_dir, "results_zero_shot.json"), "w", encoding="utf-8") as f:
        json.dump(zero_results, f, ensure_ascii=False, indent=2)

    # ============ Few Shot 评估 ============
    print("\n" + "=" * 60)
    print("Few Shot 评估")
    print("=" * 60)

    # 从训练集中选取few shot示例
    train_data = load_data(config.train_path)

    # 选择包含不同实体类型的示例
    few_shot_examples = []
    type_count = {t: 0 for t in entity_types}

    for item in train_data:
        entities = bio_to_entities(item["text"], item["labels"])
        if entities:
            entities_str = entities_to_str(entities)
            few_shot_examples.append({
                "text": get_text(item["text"]),  # 拼接为完整句子
                "entities_str": entities_str
            })
            for ent in entities:
                type_count[ent["type"]] = type_count.get(ent["type"], 0) + 1
            if len(few_shot_examples) >= config.num_few_shot:
                break

    print(f"Few shot示例数: {len(few_shot_examples)}")
    for ex in few_shot_examples:
        print(f"  文本: {ex['text'][:50]}... 实体: {ex['entities_str']}")

    few_shot_metrics = evaluate_ner(model, tokenizer, test_data, mode="few_shot", few_shot_examples=few_shot_examples)
    print(f"\nFew Shot 结果:")
    print(f"  F1: {few_shot_metrics['f1']:.4f}")
    print(f"  Precision: {few_shot_metrics['precision']:.4f}")
    print(f"  Recall: {few_shot_metrics['recall']:.4f}")

    # 保存结果
    few_results = {
        "model": "Qwen-FewShot",
        "test_f1": few_shot_metrics["f1"],
        "test_precision": few_shot_metrics["precision"],
        "test_recall": few_shot_metrics["recall"],
        "report": few_shot_metrics["report"]
    }
    with open(os.path.join(config.output_dir, "results_few_shot.json"), "w", encoding="utf-8") as f:
        json.dump(few_results, f, ensure_ascii=False, indent=2)

    return zero_results, few_results

if __name__ == "__main__":
    main()
