"""
使用大模型 API 做 NER：zero-shot vs few-shot 对比

教学重点：
  1. LLM 做 NER 的 prompt 设计
     - zero-shot：只靠任务描述，无样例
     - few-shot：给 3 个标注示例，引导格式对齐
  2. 结构化输出解析（JSON提取 + 容错处理）
  3. LLM 的 span 级别 F1 计算（与 BERT 保持可比性）
  4. 成本控制：只采样 100 条，不跑完整验证集

使用方式：
  python llm_ner.py
  python llm_ner.py --n_samples 50 --model deepseek-chat

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
LOG_DIR = ROOT / "outputs" / "logs"

ENTITY_TYPES = {
    "cluener2020": {
        "address": "地址", "book": "书名", "company": "公司",
        "game": "游戏", "government": "政府机构", "movie": "影视作品",
        "name": "人名", "organization": "组织机构", "position": "职位",
        "scene": "景点/场所",
    },
    "peoples_daily": {
        "PER": "人名", "ORG": "组织机构", "LOC": "地名",
    },
}


def get_data_dir(dataset: str) -> Path:
    return ROOT / "data" / ("cluener" if dataset == "cluener2020" else dataset)


def get_entity_types(dataset: str) -> list[str]:
    return list(ENTITY_TYPES[dataset].keys())


def build_system_prompt(dataset: str) -> str:
    """根据数据集生成 NER system prompt。"""
    type_map = ENTITY_TYPES[dataset]
    type_desc = "、".join(f"{k}：{v}" for k, v in type_map.items())
    return f"""你是一个命名实体识别（NER）专家，专门处理中文文本。
请从用户输入的文本中识别以下实体，并以 JSON 格式输出结果：
{type_desc}

输出格式（严格遵守，不要包含其他文字）：
{{"entities": [{{"text": "实体文本", "type": "实体类型英文名"}}]}}

如果没有实体，输出：{{"entities": []}}"""


# Few-shot 示例（仅 cluener2020，peoples_daily 使用空列表）
FEW_SHOT_EXAMPLES_CLUENER = [
    {
        "text": "浙商银行企业信贷部叶老桂博士则从另一个角度举了个例子",
        "output": '{"entities": [{"text": "浙商银行", "type": "company"}, {"text": "叶老桂", "type": "name"}]}'
    },
    {
        "text": "《白鹿原》改编自陕西作家陈忠实的同名小说",
        "output": '{"entities": [{"text": "白鹿原", "type": "movie"}, {"text": "陕西", "type": "address"}, {"text": "陈忠实", "type": "name"}, {"text": "白鹿原", "type": "book"}]}'
    },
    {
        "text": "华为技术有限公司总裁任正非在深圳接受了媒体采访",
        "output": '{"entities": [{"text": "华为技术有限公司", "type": "company"}, {"text": "总裁", "type": "position"}, {"text": "任正非", "type": "name"}, {"text": "深圳", "type": "address"}]}'
    },
]


def build_client() -> OpenAI:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise EnvironmentError("请设置环境变量 DEEPSEEK_API_KEY")
    return OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
    )
    
def build_local_client(base_url: str = "http://localhost:1234/v1") -> OpenAI:
    return OpenAI(base_url=base_url, api_key="not-needed")


def gold_spans_from_record(record: dict, dataset: str = "cluener2020") -> set[tuple[str, str, int, int]]:
    """提取 gold spans，格式：{(text, type, start, end)}。支持 span 和 BIO 两种格式。"""
    spans = set()
    if dataset == "peoples_daily":
        tokens = record["tokens"]
        tags = record["ner_tags"]
        i = 0
        while i < len(tags):
            if tags[i].startswith("B-"):
                etype = tags[i][2:]
                j = i + 1
                while j < len(tags) and tags[j] == f"I-{etype}":
                    j += 1
                text = "".join(tokens[i:j])
                spans.add((text, etype, i, j - 1))
                i = j
            else:
                i += 1
    else:
        for etype, surfaces in (record.get("label") or {}).items():
            for surface, positions in surfaces.items():
                for start, end in positions:
                    spans.add((surface, etype, start, end))
    return spans


def pred_spans_from_response(text: str, response_text: str, dataset: str) -> set[tuple[str, str, int, int]]:
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
    entity_types = get_entity_types(dataset)
    spans = set()
    for ent in entities:
        if not isinstance(ent, dict):
            continue
        surface = str(ent.get("text", "")).strip()
        etype = str(ent.get("type", "")).strip()
        if not surface or etype not in entity_types:
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


def zero_shot_prompt(text: str, dataset: str) -> list[dict]:
    return [
        {"role": "system", "content": build_system_prompt(dataset)},
        {"role": "user", "content": text},
    ]


def few_shot_prompt(text: str, dataset: str) -> list[dict]:
    messages = [{"role": "system", "content": build_system_prompt(dataset)}]
    # few-shot 示例仅用于 cluener2020
    if dataset == "cluener2020":
        for ex in FEW_SHOT_EXAMPLES_CLUENER:
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


def sample_records(n: int, dataset: str, seed: int = 42) -> list[dict]:
    """从验证集中采样 n 条，尽量覆盖所有实体类型。"""
    data_dir = get_data_dir(dataset)
    with open(data_dir / "validation.json", "r", encoding="utf-8") as f:
        records = json.load(f)

    random.seed(seed)
    entity_types = get_entity_types(dataset)
    by_type = defaultdict(list)
    for r in records:
        if dataset == "peoples_daily":
            for t in r.get("ner_tags", []):
                if t.startswith("B-"):
                    by_type[t[2:]].append(r)
        else:
            for etype in (r.get("label") or {}):
                by_type[etype].append(r)

    selected = set()
    selected_list = []

    per_type = max(1, n // len(entity_types))
    for etype in entity_types:
        candidates = [r for r in by_type[etype] if id(r) not in selected]
        chosen = random.sample(candidates, min(per_type, len(candidates)))
        for r in chosen:
            if len(selected_list) < n and id(r) not in selected:
                selected.add(id(r))
                selected_list.append(r)

    remaining = [r for r in records if id(r) not in selected]
    random.shuffle(remaining)
    for r in remaining:
        if len(selected_list) >= n:
            break
        selected_list.append(r)

    return selected_list[:n]


def main():
    args = parse_args()
    dataset = args.dataset

    if args.local:
        client = build_local_client(args.base_url)
    else:
        client = build_client()
    records = sample_records(args.n_samples, dataset)
    print(f"采样 {len(records)} 条验证集样本 ({dataset})")

    zero_shot_golds = []
    zero_shot_preds = []
    few_shot_golds = []
    few_shot_preds = []

    detail_records = []

    for i, record in enumerate(records, 1):
        text = record.get("text", "".join(record.get("tokens", [])))
        gold = gold_spans_from_record(record, dataset)

        # Zero-shot
        zs_resp = call_api(client, zero_shot_prompt(text, dataset), args.model)
        if args.verbose:
            print(f"\n{'='*40}")
            print(f"[{i}] ZERO-SHOT 原始返回:")
            print(zs_resp[:500])
            if len(zs_resp) > 500:
                print(f"... (共 {len(zs_resp)} 字符)")
            print(f"{'='*40}")
        zs_pred = pred_spans_from_response(text, zs_resp, dataset)

        # Few-shot
        fs_resp = call_api(client, few_shot_prompt(text, dataset), args.model)
        if args.verbose:
            print(f"\n{'='*40}")
            print(f"[{i}] FEW-SHOT 原始返回:")
            print(fs_resp[:500])
            if len(fs_resp) > 500:
                print(f"... (共 {len(fs_resp)} 字符)")
            print(f"{'='*40}")
        fs_pred = pred_spans_from_response(text, fs_resp, dataset)

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

    # 确保数值可 JSON 序列化
    def _to_python(v):
        return v.item() if hasattr(v, "item") else v

    result["zero_shot"] = {k: _to_python(v) for k, v in result["zero_shot"].items()}
    result["few_shot"] = {k: _to_python(v) for k, v in result["few_shot"].items()}

    out_path = LOG_DIR / f"eval_llm_{dataset}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nLLM 评估结果已保存 → {out_path}")
    print("\n下一步：python compare_results.py")


def parse_args():
    parser = argparse.ArgumentParser(description="LLM zero-shot/few-shot NER 对比")
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--dataset", type=str, default="cluener2020",
                        choices=["cluener2020", "peoples_daily"])
    parser.add_argument("--model", type=str, default="deepseek-chat")
    parser.add_argument("--local", action="store_true",
                        help="使用本地 LM Studio API（localhost:1234）")
    parser.add_argument("--base_url", type=str, default="http://localhost:1234/v1",
                        help="本地 API 地址（与 --local 配合使用）")
    parser.add_argument("--verbose", action="store_true",
                        help="打印大模型原始返回")
    return parser.parse_args()


if __name__ == "__main__":
    main()
