import json
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


Sample = Dict[str, List[str]]


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_labels(data_dir: Path) -> List[str]:
    return load_json(data_dir / "label_names.json")


def load_samples(data_dir: Path, split: str, limit: int | None = None) -> List[Sample]:
    samples = load_json(data_dir / f"{split}.json")
    if limit is not None and limit > 0:
        return samples[:limit]
    return samples


def sample_text(tokens: Sequence[str]) -> str:
    return "".join(tokens)


def format_tagged_example(sample: Sample) -> str:
    items = [
        {"token": token, "tag": tag}
        for token, tag in zip(sample["tokens"], sample["ner_tags"])
    ]
    return json.dumps(items, ensure_ascii=False)


def build_messages(
    tokens: Sequence[str],
    labels: Sequence[str],
    examples: Sequence[Sample] | None = None,
) -> List[Dict[str, str]]:
    label_text = ", ".join(labels)
    system = (
        "你是一个中文命名实体识别序列标注助手。"
        "请严格为用户给出的 tokens 中每一个 token 输出一个 BIO 标签。"
        f"合法标签只有: {label_text}。"
        "输出必须是 JSON 数组，长度必须与 tokens 完全一致；"
        "每个元素格式为 {\"token\":\"原token\",\"tag\":\"标签\"}，不要输出解释。"
    )
    messages = [{"role": "system", "content": system}]

    for example in examples or []:
        messages.append(
            {
                "role": "user",
                "content": json.dumps(
                    {"tokens": example["tokens"], "text": sample_text(example["tokens"])},
                    ensure_ascii=False,
                ),
            }
        )
        messages.append({"role": "assistant", "content": format_tagged_example(example)})

    messages.append(
        {
            "role": "user",
            "content": json.dumps(
                {"tokens": list(tokens), "text": sample_text(tokens)},
                ensure_ascii=False,
            ),
        }
    )
    return messages


def build_sft_record(sample: Sample, labels: Sequence[str], examples: Sequence[Sample] | None = None) -> Dict[str, str]:
    messages = build_messages(sample["tokens"], labels, examples)
    prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages[:-1])
    prompt = f"{prompt}\nuser: {messages[-1]['content']}\nassistant: "
    return {"prompt": prompt, "response": format_tagged_example(sample)}


def choose_demonstrations(samples: Sequence[Sample], k: int, seed: int) -> List[Sample]:
    if k <= 0:
        return []
    rng = random.Random(seed)
    pool = list(samples)
    rng.shuffle(pool)
    return pool[:k]


def parse_pred_tags(raw_text: str, tokens: Sequence[str], labels: Sequence[str]) -> List[str]:
    text = raw_text.strip()
    match = re.search(r"\[[\s\S]*\]", text)
    if match:
        text = match.group(0)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = []

    tags: List[str] = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                tags.append(str(item.get("tag", "O")))
            elif isinstance(item, str):
                tags.append(item)

    label_set = set(labels)
    cleaned = [tag if tag in label_set else "O" for tag in tags]
    if len(cleaned) < len(tokens):
        cleaned.extend(["O"] * (len(tokens) - len(cleaned)))
    return cleaned[: len(tokens)]


def bio_spans(tags: Sequence[str]) -> set[tuple[str, int, int]]:
    spans = set()
    ent_type = None
    start = 0
    for idx, tag in enumerate(list(tags) + ["O"]):
        if tag.startswith("B-"):
            if ent_type is not None:
                spans.add((ent_type, start, idx - 1))
            ent_type = tag[2:]
            start = idx
        elif tag.startswith("I-") and ent_type == tag[2:]:
            continue
        else:
            if ent_type is not None:
                spans.add((ent_type, start, idx - 1))
                ent_type = None
    return spans


def compute_metrics(y_true: Iterable[Sequence[str]], y_pred: Iterable[Sequence[str]]) -> Dict[str, float]:
    true_sets = [bio_spans(seq) for seq in y_true]
    pred_sets = [bio_spans(seq) for seq in y_pred]

    tp = sum(len(t & p) for t, p in zip(true_sets, pred_sets))
    pred_total = sum(len(p) for p in pred_sets)
    true_total = sum(len(t) for t in true_sets)

    precision = tp / pred_total if pred_total else 0.0
    recall = tp / true_total if true_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}
