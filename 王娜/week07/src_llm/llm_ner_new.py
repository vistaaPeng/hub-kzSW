"""
使用大模型 API（OpenAI SDK 兼容模式）做 NER（peoples_daily 数据集），
支持 zero-shot 和 few-shot 对比，支持选择调用 Kimi 或 DeepSeek 的 API。

使用方式：
  python llm_ner_new.py --platform kimi --api_key sk-xxxxx --n_samples 100
  python llm_ner_new.py --platform deepseek --api_key sk-xxxxx --n_samples 100
"""

import os
import sys

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

PLATFORM_CONFIG = {
    "kimi": {
        "base_url": "https://api.moonshot.cn/v1",
        "default_model": "moonshot-v1-8k",
        "env_key": "MOONSHOT_API_KEY",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "default_model": "deepseek-chat",
        "env_key": "DEEPSEEK_API_KEY",
    },
}

ENTITY_TYPE_ZH = {
    "PER": "人名",
    "ORG": "机构",
    "LOC": "地名",
}
ENTITY_TYPES_EN = list(ENTITY_TYPE_ZH.keys())


def build_client(platform: str, api_key: str | None = None) -> OpenAI:
    cfg = PLATFORM_CONFIG[platform]
    key = api_key or os.getenv(cfg["env_key"])
    if not key:
        print(f"\n[ERROR] 找不到 {platform} 的 API Key", flush=True)
        print(f"环境变量名：{cfg['env_key']}", flush=True)
        print("解决方法：", flush=True)
        print(f"  1. set {cfg['env_key']}=sk-xxxxx", flush=True)
        print(f"  2. python llm_ner_new.py --platform {platform} --api_key sk-xxxxx", flush=True)
        sys.exit(1)
    return OpenAI(api_key=key, base_url=cfg["base_url"])


def gold_spans_from_record(record: dict) -> set[tuple[str, str, int, int]]:
    """从 BIO 格式（tokens + ner_tags）提取 span，格式：{(surface, type, start, end)}。"""
    tokens = record["tokens"]
    ner_tags = record["ner_tags"]
    spans = set()
    i = 0
    n = len(ner_tags)
    while i < n:
        tag = ner_tags[i]
        if tag.startswith("B-"):
            etype = tag[2:]
            start = i
            i += 1
            while i < n and ner_tags[i] == f"I-{etype}":
                i += 1
            end = i - 1
            surface = "".join(tokens[start:end + 1])
            spans.add((surface, etype, start, end))
        else:
            i += 1
    return spans


def pred_spans_from_response(text: str, response_text: str) -> set[tuple[str, str, int, int]]:
    """从 LLM 的 JSON 输出解析 span，格式：{(surface, type, start, end)}。"""
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


SYSTEM_PROMPT = (
    "你是一个命名实体识别（NER）专家，专门处理中文文本。\n"
    "请从用户输入的文本中识别以下3类实体，并以 JSON 格式输出结果：\n"
    "- PER：人名\n"
    "- ORG：机构名\n"
    "- LOC：地名\n\n"
    '输出格式（严格遵守，不要包含其他文字）：\n'
    '{"entities": [{"text": "实体文本", "type": "实体类型英文名"}, ...]}\n\n'
    "注意：text 字段请直接输出中文字符，不要使用 Unicode 转义（如 \\uXXXX）。\n"
    '如果没有实体，输出：{"entities": []}'
)

FEW_SHOT_EXAMPLES = [
    {
        "text": "海钓比赛地点在厦门与金门之间的海域。",
        "output": json.dumps({"entities": [{"text": "厦门", "type": "LOC"}, {"text": "金门", "type": "LOC"}]}, ensure_ascii=False),
    },
    {
        "text": "这座依山傍水的博物馆由国内一流的设计团队设计。",
        "output": json.dumps({"entities": []}, ensure_ascii=False),
    },
    {
        "text": "李强总理在中关村科技园考察了华为技术有限公司。",
        "output": json.dumps({"entities": [{"text": "李强", "type": "PER"}, {"text": "中关村科技园", "type": "ORG"}, {"text": "华为技术有限公司", "type": "ORG"}]}, ensure_ascii=False),
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
    """调用 LLM API，返回文本输出，带 3 次重试和 60 秒超时。"""
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=512,
                timeout=60,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            print(f"  API 调用失败（attempt {attempt + 1}/3）：{e}", flush=True)
            if attempt < 2:
                time.sleep(2 ** attempt)
    return ""


def sample_records(n: int, seed: int = 42) -> list[dict]:
    """从验证集中采样 n 条，尽量覆盖所有实体类型。"""
    with open(DATA_DIR / "validation.json", "r", encoding="utf-8") as f:
        records = json.load(f)
    random.seed(seed)
    by_type = defaultdict(list)
    for r in records:
        spans = gold_spans_from_record(r)
        etypes_in_record = {s[1] for s in spans}
        for etype in etypes_in_record:
            by_type[etype].append(r)
    selected = set()
    selected_list = []
    per_type = max(1, n // len(ENTITY_TYPES_EN))
    for etype in ENTITY_TYPES_EN:
        candidates = [r for r in by_type[etype] if id(r) not in selected]
        if candidates:
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
    print("\n" + "=" * 60, flush=True)
    print("[llm_ner_new] 脚本启动 —— LLM NER（peoples_daily）Zero-shot vs Few-shot", flush=True)
    print("=" * 60, flush=True)

    args = parse_args()
    model = args.model if args.model is not None else PLATFORM_CONFIG[args.platform]["default_model"]
    print(f"[llm_ner_new] 平台：{args.platform}，模型：{model}，样本数：{args.n_samples}", flush=True)

    client = build_client(args.platform, args.api_key)
    print("[llm_ner_new] API 客户端创建成功", flush=True)

    records = sample_records(args.n_samples)
    print(f"[llm_ner_new] 采样 {len(records)} 条验证集样本", flush=True)

    zero_shot_golds = []
    zero_shot_preds = []
    few_shot_golds = []
    few_shot_preds = []
    detail_records = []

    for i, record in enumerate(records, 1):
        text = "".join(record["tokens"])
        gold = gold_spans_from_record(record)

        print(f"\n[llm_ner_new] 正在处理第 {i}/{len(records)} 条...", flush=True)
        print(f"  文本：{text[:40]}...", flush=True)

        zs_resp = call_api(client, zero_shot_prompt(text), model)
        print(f"  Zero-shot 返回：{zs_resp[:80]}...", flush=True)
        zs_pred = pred_spans_from_response(text, zs_resp)

        fs_resp = call_api(client, few_shot_prompt(text), model)
        print(f"  Few-shot 返回：{fs_resp[:80]}...", flush=True)
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

    zs_metrics = compute_span_f1(zero_shot_golds, zero_shot_preds)
    fs_metrics = compute_span_f1(few_shot_golds, few_shot_preds)

    print("\n" + "=" * 60, flush=True)
    print(f"LLM NER 对比结果（平台：{args.platform}，模型：{model}，样本：{len(records)} 条）", flush=True)
    print("=" * 60, flush=True)
    print(f"{'方案':<20} {'Precision':>10} {'Recall':>10} {'F1':>10}", flush=True)
    print("-" * 52, flush=True)
    print(f"{'Zero-shot':<20} {zs_metrics['precision']:>10.4f} {zs_metrics['recall']:>10.4f} {zs_metrics['f1']:>10.4f}", flush=True)
    print(f"{'Few-shot (3例)':<20} {fs_metrics['precision']:>10.4f} {fs_metrics['recall']:>10.4f} {fs_metrics['f1']:>10.4f}", flush=True)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "platform": args.platform,
        "model": model,
        "n_samples": len(records),
        "zero_shot": zs_metrics,
        "few_shot": fs_metrics,
        "detail": detail_records,
    }

    def _to_python(v):
        return v.item() if hasattr(v, "item") else v

    result["zero_shot"] = {k: _to_python(v) for k, v in result["zero_shot"].items()}
    result["few_shot"] = {k: _to_python(v) for k, v in result["few_shot"].items()}

    out_path = LOG_DIR / "eval_llm_peoples_daily_deepseek.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nLLM 评估结果已保存 → {out_path}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="LLM zero-shot/few-shot NER 对比（peoples_daily）")
    parser.add_argument("--platform", type=str, default="kimi", choices=["kimi", "deepseek"], help="选择 API 平台（默认：kimi）")
    parser.add_argument("--api_key", type=str, default=None, help="直接传入 API Key（替代环境变量）")
    parser.add_argument("--model", type=str, default=None, help="模型名称（默认使用平台配置的 default_model）")
    parser.add_argument("--n_samples", type=int, default=100, help="采样条数（默认：100）")
    return parser.parse_args()


if __name__ == "__main__":
    main()