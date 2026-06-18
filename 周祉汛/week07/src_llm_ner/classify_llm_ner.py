"""
LLM Zero-Shot NER：使用 Qwen2-0.5B-Instruct，要求输出 JSON 实体列表。
评估实体级 F1。
"""

import argparse
import json
import random
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "ner"
MODEL_PATH = ROOT.parent.parent / "pretrain_models" / "Qwen2-0.5B-Instruct"
OUTPUT_DIR = ROOT / "outputs_ner"

# 从 label_map.json 动态获取标签类型
def get_entity_types(label_map_path):
    with open(label_map_path, "r", encoding="utf-8") as f:
        lm = json.load(f)
    types = set()
    for label in lm["id2label"].values():
        if label != "O":
            if label.startswith("B-"):
                types.add(label[2:])
    return list(types)

SYSTEM_PROMPT_TEMPLATE = """
你是一个命名实体识别（NER）助手。请从给定的新闻标题中抽取实体，以 JSON 列表形式输出。
实体类型包括：{types}。
示例输出：[{{"text": "阿里巴巴", "type": "ORG"}}, {{"text": "杭州", "type": "LOC"}}]
只输出 JSON，不要有其他内容。
"""

def build_prompt(text, entity_types):
    return f"新闻标题：{text}\n实体列表："

def load_model(model_path, device):
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        dtype=torch.float16 if device.type == "cuda" else torch.float32,
        device_map="auto" if device.type == "cuda" else None,
        trust_remote_code=True,
    )
    if device.type != "cuda":
        model = model.to(device)
    model.eval()
    return model, tokenizer

def parse_prediction(raw_output: str):
    """从模型输出中提取 JSON 实体列表"""
    try:
        start = raw_output.find('[')
        end = raw_output.rfind(']') + 1
        if start == -1 or end == 0:
            return []
        json_str = raw_output[start:end]
        entities = json.loads(json_str)
        if isinstance(entities, list):
            return entities
    except:
        pass
    return []

def evaluate_entity_f1(pred_entities, gold_entities):
    """计算实体级别的精确率、召回率、F1（基于严格匹配）"""
    # 将实体列表转换为 (text, type) 集合
    pred_set = set((e["text"], e["type"]) for e in pred_entities)
    gold_set = set((e["text"], e["type"]) for e in gold_entities)
    tp = len(pred_set & gold_set)
    prec = tp / len(pred_set) if pred_set else 0
    rec = tp / len(gold_set) if gold_set else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    return prec, rec, f1

def load_gold_entities(text, label_ids, id2label):
    """将 BIO 标签序列转换为实体列表"""
    # 与 predict.py 中的 extract_entities 逻辑相同
    tokens = list(text)
    entities = []
    current = None
    for i, label_id in enumerate(label_ids):
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
    return [{"text": "".join(e["tokens"]), "type": e["type"]} for e in entities]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=str(MODEL_PATH))
    parser.add_argument("--data_dir", default=str(DATA_DIR))
    parser.add_argument("--num_samples", default=200, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)

    # 加载数据
    with open(data_dir / "val.json", "r", encoding="utf-8") as f:
        val_data = json.load(f)
    with open(data_dir / "label_map.json", "r", encoding="utf-8") as f:
        label_map = json.load(f)
    id2label = {int(k): v for k, v in label_map["id2label"].items()}
    entity_types = get_entity_types(data_dir / "label_map.json")
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(types="、".join(entity_types))

    random.seed(args.seed)
    n = 5 if args.demo else args.num_samples
    samples = random.sample(val_data, min(n, len(val_data)))

    model, tokenizer = load_model(args.model_path, device)

    total_prec, total_rec, total_f1 = 0.0, 0.0, 0.0
    total_samples = 0
    results = []

    t0 = time.time()
    for i, item in enumerate(samples):
        text = item["text"]
        char_labels = item["labels"]
        gold_entities = load_gold_entities(text, char_labels, id2label)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": build_prompt(text, entity_types)},
        ]
        encoding = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt", return_dict=True
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        prompt_len = input_ids.shape[-1]

        with torch.no_grad():
            output_ids = model.generate(
                input_ids, attention_mask=attention_mask,
                max_new_tokens=256, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = output_ids[0][prompt_len:]
        raw_output = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        pred_entities = parse_prediction(raw_output)

        prec, rec, f1 = evaluate_entity_f1(pred_entities, gold_entities)
        total_prec += prec
        total_rec += rec
        total_f1 += f1
        total_samples += 1

        results.append({
            "text": text,
            "gold_entities": gold_entities,
            "pred_entities": pred_entities,
            "raw_output": raw_output,
            "f1": f1,
        })
        print(f"[{i+1}/{len(samples)}] F1={f1:.3f} | {text[:50]}")

    avg_prec = total_prec / total_samples
    avg_rec = total_rec / total_samples
    avg_f1 = total_f1 / total_samples
    elapsed = time.time() - t0

    print(f"\nZero-Shot NER 结果（{len(samples)} 条）")
    print(f"平均实体级 Precision: {avg_prec:.4f}")
    print(f"平均实体级 Recall:    {avg_rec:.4f}")
    print(f"平均实体级 F1:        {avg_f1:.4f}")
    print(f"总耗时 {elapsed:.1f}s, 平均 {elapsed/len(samples):.2f}s/条")

    out_path = OUTPUT_DIR / "llm_zero_shot_ner_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"结果已保存 → {out_path}")

if __name__ == "__main__":
    main()