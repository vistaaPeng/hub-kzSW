"""
使用大模型 API 做 NER：zero-shot vs few-shot 对比（适配新数据集：PER/ORG/LOC）

新数据集标签：
  O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC

使用方式：
  python llm_ner.py
  python llm_ner.py --n_samples 50 --model qwen-max

依赖：
  pip install openai
  export DASHSCOPE_API_KEY="sk-xxx"
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import time
import random
import argparse
import re
from pathlib import Path
from collections import defaultdict

from openai import OpenAI

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "peoples_daily"
LOG_DIR = ROOT / "outputs" / "logs"

# 新的实体类型（仅三类）
ENTITY_TYPE_ZH = {
    "name": "人名",
    "organization": "组织机构",
    "location": "地点",
}
ENTITY_TYPES_EN = ["name", "organization", "location"]

# 标签映射（BIO -> 英文类型）
TAG_TO_TYPE = {
    "PER": "name",
    "ORG": "organization",
    "LOC": "location",
}
PREFIX_B = "B-"
PREFIX_I = "I-"


def build_client() -> OpenAI:
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise EnvironmentError("请设置环境变量 DASHSCOPE_API_KEY")
    return OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )


def gold_spans_from_record(record: dict) -> set[tuple[str, str, int, int]]:
    """
    从新格式数据中提取 gold spans。
    输入：{"tokens": [...], "ner_tags": [...]}
    输出：{(surface, type, start, end)}，其中 start/end 为字符索引（token 索引）。
    """
    tokens = record.get("tokens", [])
    tags = record.get("ner_tags", [])
    if not tokens or not tags or len(tokens) != len(tags):
        return set()

    spans = set()
    i = 0
    n = len(tokens)
    while i < n:
        tag = tags[i]
        # 跳过非实体
        if tag == "O" or not tag.startswith(PREFIX_B):
            i += 1
            continue

        # 实体开始，提取类型（PER/ORG/LOC）
        bio_type = tag[len(PREFIX_B):]   # 例如 "PER"
        entity_type = TAG_TO_TYPE.get(bio_type)
        if entity_type is None:
            i += 1
            continue

        start = i
        entity_tokens = [tokens[i]]
        i += 1
        # 继续收集 I- 标签
        while i < n and tags[i] == f"{PREFIX_I}{bio_type}":
            entity_tokens.append(tokens[i])
            i += 1
        end = i - 1
        surface = "".join(entity_tokens)   # tokens 已是字符列表
        spans.add((surface, entity_type, start, end))

    return spans


def pred_spans_from_response(text: str, response_text: str) -> set[tuple[str, str, int, int]]:
    """从 LLM 输出中解析 span，格式：{(surface, type, start, end)}。"""
    json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
    if not json_match:
        return set()

    try:
        obj = json.loads(json_match.group())
    except json.JSONDecodeError:
        return set()

    entities = obj.get("entities", [])
    if not isinstance(entities, list):
        return set()

    spans = set()
    for ent in entities:
        if not isinstance(ent, dict):
            continue
        surface = str(ent.get("text", "")).strip()
        etype = str(ent.get("type", "")).strip()
        if not surface or etype not in ENTITY_TYPES_EN:
            continue
        idx = text.find(surface)
        if idx == -1:
            continue
        spans.add((surface, etype, idx, idx + len(surface) - 1))

    return spans


def compute_span_f1(all_golds: list[set], all_preds: list[set]) -> dict:
    """计算 span-level 精确率、召回率、F1。"""
    tp = sum(len(g & p) for g, p in zip(all_golds, all_preds))
    pred_total = sum(len(p) for p in all_preds)
    gold_total = sum(len(g) for g in all_golds)
    p = tp / pred_total if pred_total else 0.0
    r = tp / gold_total if gold_total else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": p, "recall": r, "f1": f1, "tp": tp, "pred_total": pred_total, "gold_total": gold_total}


# 更新的 System Prompt（只识别三类实体）
SYSTEM_PROMPT = """你是一个命名实体识别（NER）专家，专门处理中文文本。
请从用户输入的文本中识别以下3类实体，并以 JSON 格式输出结果：
- name：人名
- organization：组织机构名称（如公司、政府机构、球队等）
- location：地点（如城市、景点、地址等）

输出格式（严格遵守，不要包含其他文字）：
{"entities": [{"text": "实体文本", "type": "实体类型英文名"}, ...]}

如果没有实体，输出：{"entities": []}"""

# 更新的 Few-shot 示例（只使用三类实体）
FEW_SHOT_EXAMPLES = [
    {
        "text": "浙商银行企业信贷部叶老桂博士则从另一个角度举了个例子",
        "output": '{"entities": [{"text": "浙商银行", "type": "organization"}, {"text": "叶老桂", "type": "name"}]}'
    },
    {
        "text": "《白鹿原》改编自陕西作家陈忠实的同名小说",
        "output": '{"entities": [{"text": "陕西", "type": "location"}, {"text": "陈忠实", "type": "name"}]}'
    },
    {
        "text": "华为技术有限公司总裁任正非在深圳接受了媒体采访",
        "output": '{"entities": [{"text": "华为技术有限公司", "type": "organization"}, {"text": "任正非", "type": "name"}, {"text": "深圳", "type": "location"}]}'
    },
]


def zero_shot_prompt(text: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]


def few_shot_prompt(text: str) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": ex["text"]})
        messages.append({"role": "assistant", "content": ex["output"]})
    messages.append({"role": "user", "content": text})
    return messages


def call_api(client: OpenAI, messages: list[dict], model: str) -> str:
    """调用 LLM API，返回文本输出，带简单重试。"""
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=512,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                print(f"  API 调用失败：{e}")
                return ""
    return ""


def sample_records(n: int, seed: int = 42) -> list[dict]:
    """从验证集中采样 n 条，尽量覆盖所有实体类型（基于实际标签）。"""
    with open(DATA_DIR / "validation.json", "r", encoding="utf-8") as f:
        records = json.load(f)

    random.seed(seed)

    # 统计每条记录包含的实体类型（通过 ner_tags）
    record_types = []
    for rec in records:
        types = set()
        for tag in rec.get("ner_tags", []):
            if tag != "O" and tag.startswith(PREFIX_B):
                bio_type = tag[2:]   # 去掉 "B-"
                etype = TAG_TO_TYPE.get(bio_type)
                if etype:
                    types.add(etype)
        record_types.append(types)

    # 按类型分组（索引）
    by_type = defaultdict(list)
    for idx, types in enumerate(record_types):
        for t in types:
            by_type[t].append(idx)

    selected_indices = set()
    selected_records = []

    # 每类先取 per_type 条
    per_type = max(1, n // len(ENTITY_TYPES_EN))
    for etype in ENTITY_TYPES_EN:
        candidates = [i for i in by_type.get(etype, []) if i not in selected_indices]
        chosen = random.sample(candidates, min(per_type, len(candidates)))
        for i in chosen:
            if len(selected_records) < n:
                selected_indices.add(i)
                selected_records.append(records[i])

    # 补足到 n 条
    remaining = [i for i in range(len(records)) if i not in selected_indices]
    random.shuffle(remaining)
    for i in remaining:
        if len(selected_records) >= n:
            break
        selected_records.append(records[i])

    return selected_records[:n]


def main():
    args = parse_args()

    client = build_client()
    records = sample_records(args.n_samples)
    print(f"采样 {len(records)} 条验证集样本")

    zero_shot_golds = []
    zero_shot_preds = []
    few_shot_golds = []
    few_shot_preds = []

    detail_records = []

    for i, record in enumerate(records, 1):
        # 将 tokens 拼接成文本字符串
        text = "".join(record.get("tokens", []))
        gold = gold_spans_from_record(record)

        # Zero-shot
        zs_resp = call_api(client, zero_shot_prompt(text), args.model)
        zs_pred = pred_spans_from_response(text, zs_resp)

        # Few-shot
        fs_resp = call_api(client, few_shot_prompt(text), args.model)
        fs_pred = pred_spans_from_response(text, fs_resp)

        zero_shot_golds.append(gold)
        zero_shot_preds.append(zs_pred)
        few_shot_golds.append(gold)
        few_shot_preds.append(fs_pred)

        detail_records.append({
            "text": text,
            "gold": [{"text": s, "type": t} for s, t, _, _ in gold],
            "zero_shot": [{"text": s, "type": t} for s, t, _, _ in zs_pred],
            "few_shot": [{"text": s, "type": t} for s, t, _, _ in fs_pred],
        })

        if i % 10 == 0 or i == len(records):
            print(f"  已处理 {i}/{len(records)} 条")

    zs_metrics = compute_span_f1(zero_shot_golds, zero_shot_preds)
    fs_metrics = compute_span_f1(few_shot_golds, few_shot_preds)

    print("\n" + "=" * 60)
    print(f"LLM NER 对比结果（模型：{args.model}，样本：{len(records)} 条）")
    print("=" * 60)
    print(f"{'方案':<20} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 52)
    print(f"{'Zero-shot':<20} {zs_metrics['precision']:>10.4f} {zs_metrics['recall']:>10.4f} {zs_metrics['f1']:>10.4f}")
    print(f"{'Few-shot (3例)':<20} {fs_metrics['precision']:>10.4f} {fs_metrics['recall']:>10.4f} {fs_metrics['f1']:>10.4f}")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "model": args.model,
        "n_samples": len(records),
        "zero_shot": zs_metrics,
        "few_shot": fs_metrics,
        "detail": detail_records,
    }

    def _to_python(v):
        return v.item() if hasattr(v, "item") else v

    result["zero_shot"] = {k: _to_python(v) for k, v in result["zero_shot"].items()}
    result["few_shot"] = {k: _to_python(v) for k, v in result["few_shot"].items()}

    out_path = LOG_DIR / "eval_llm.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nLLM 评估结果已保存 → {out_path}")
    print("\n下一步：python compare_results.py")


def parse_args():
    parser = argparse.ArgumentParser(description="LLM zero-shot/few-shot NER 对比")
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--model", type=str, default="qwen-plus")
    return parser.parse_args()


if __name__ == "__main__":
    main()
