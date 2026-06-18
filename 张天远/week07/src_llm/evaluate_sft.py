"""
加载 SFT checkpoint（LoRA / 全量微调），在验证集上评估 NER entity-level F1，
与 BERT+CRF 和 LLM API（zero/few-shot）多方对比

教学重点：
  1. 生成式 NER 的评估方式：JSON 解析 → span-level F1（与 llm_ner.py 一致）
  2. LoRA adapter 自动识别：目录含 adapter_config.json → LoRA，否则 → 全量
  3. 与 BERT+CRF 的对比：生成式 vs 序列标注，各有什么优劣

使用方式：
  python evaluate_sft.py                              # 评估 LoRA 模型（默认）
  python evaluate_sft.py --ckpt_dir ../outputs/sft_full_ckpt  # 评估全量微调模型
  python evaluate_sft.py --n_samples 50 --demo        # 5 条示例快速演示

依赖：
  pip install torch transformers peft
"""

import os
import argparse
import json
import random
import re
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

ROOT        = Path(__file__).parent.parent
DATA_DIR    = ROOT / "data" / "cluener"
LOG_DIR     = ROOT / "outputs" / "logs"

# 数据集 → entity_types 映射（与 train_sft_qlora.py 同步）
DATASET_ENTITY_TYPES = {
    "cluener2020": [
        "address", "book", "company", "game", "government",
        "movie", "name", "organization", "position", "scene",
    ],
    "peoples_daily": ["PER", "ORG", "LOC"],
}

def get_entity_types(dataset: str) -> list[str]:
    return DATASET_ENTITY_TYPES.get(dataset, DATASET_ENTITY_TYPES["cluener2020"])

def build_system_prompt(dataset: str) -> str:
    types = get_entity_types(dataset)
    type_desc = "、".join(types)
    return (
        "你是一个命名实体识别助手。从文本中识别命名实体，以 JSON 格式输出。\n"
        f"实体类型（英文标识）：{type_desc}\n"
        '输出格式（严格遵守，不输出其他内容）：{"entities": [{"text": "实体文本", "type": "实体类型"}]}\n'
        '无实体时输出：{"entities": []}'
    )


# ══════════════════════════════════════════════════════════════════════════════
# 模型加载
# ══════════════════════════════════════════════════════════════════════════════

def load_model(model_path: str, ckpt_dir: str, device: torch.device):
    """加载 SFT checkpoint，自动识别 LoRA/全量/QLoRA。"""
    ckpt_path = Path(ckpt_dir)
    is_lora   = (ckpt_path / "adapter_config.json").exists()

    # 处理模型路径：本地路径用 resolve()，HF 模型名直接用字符串
    model_src = model_path
    if Path(model_path).exists():
        model_src = str(Path(model_path).resolve())

    if is_lora:
        if not PEFT_AVAILABLE:
            raise ImportError("加载 LoRA adapter 需要 peft 库：pip install peft>=0.14.0")
        print(f"检测到 LoRA adapter，加载 base model: {model_src}")
        tokenizer  = AutoTokenizer.from_pretrained(model_src, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            model_src,
            dtype=torch.float16 if device.type == "cuda" else torch.float32,
            device_map="auto" if device.type == "cuda" else None,
            trust_remote_code=True,
        )
        print(f"加载 LoRA adapter: {ckpt_dir}")
        model = PeftModel.from_pretrained(base_model, str(ckpt_path))
        model = model.merge_and_unload()
    else:
        print(f"检测到全量微调 checkpoint，直接加载: {ckpt_dir}")
        tokenizer = AutoTokenizer.from_pretrained(
            str(ckpt_path), trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            str(ckpt_path),
            dtype=torch.float16 if device.type == "cuda" else torch.float32,
            device_map="auto" if device.type == "cuda" else None,
            trust_remote_code=True,
        )

    if device.type != "cuda":
        model = model.to(device)
    model.eval()
    ckpt_type = "LoRA adapter 已合并" if is_lora else "全量微调模型"
    print(f"模型加载完成（{ckpt_type}）\n")
    return model, tokenizer


# ══════════════════════════════════════════════════════════════════════════════
# 推理与解析
# ══════════════════════════════════════════════════════════════════════════════

def generate_ner(text: str, model, tokenizer, device: torch.device,
                 system_prompt: str, max_new_tokens: int = 256) -> str:
    """生成 NER JSON 输出。system_prompt 根据数据集动态生成。"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": text},
    ]
    encoding = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_tensors="pt", return_dict=True,
    )
    input_ids      = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    prompt_len     = input_ids.shape[-1]

    with torch.no_grad():
        output_ids = model.generate(
            input_ids, attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0][prompt_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def gold_spans_from_record(record: dict, dataset: str = "cluener2020") -> set[tuple[str, str, int, int]]:
    """提取 gold spans，格式 (text, type, start, end)。兼容 cluener2020 (span) 和 peoples_daily (BIO)。"""
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


def pred_spans_from_output(text: str, raw_output: str, entity_types: list[str]) -> set[tuple[str, str, int, int]]:
    """从 SFT 生成的 JSON 中提取 spans，用 text.find() 近似定位。entity_types 根据数据集动态传入。"""
    json_match = re.search(r"\{.*\}", raw_output, re.DOTALL)
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
        etype   = str(ent.get("type", "")).strip()
        if not surface or etype not in entity_types:
            continue
        idx = text.find(surface)
        if idx == -1:
            continue
        spans.add((surface, etype, idx, idx + len(surface) - 1))
    return spans


def compute_span_f1(all_golds: list[set], all_preds: list[set]) -> dict:
    """计算 span-level precision / recall / F1。与 llm_ner.py 的同名函数完全一致。"""
    tp         = sum(len(g & p) for g, p in zip(all_golds, all_preds))
    pred_total = sum(len(p) for p in all_preds)
    gold_total = sum(len(g) for g in all_golds)
    p  = tp / pred_total if pred_total else 0.0
    r  = tp / gold_total if gold_total else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": p, "recall": r, "f1": f1,
            "tp": tp, "pred_total": pred_total, "gold_total": gold_total}


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="LLM SFT NER 评估")
    parser.add_argument("--model_path",  default=None,
                        help="HF 模型名或本地路径（默认 auto：LoRA 适配器目录向上推理）")
    parser.add_argument("--ckpt_dir",    default=None,
                        help="checkpoint 目录；含 adapter_config.json → LoRA，否则 → 全量")
    parser.add_argument("--dataset",     default="cluener2020",
                        choices=["cluener2020", "peoples_daily"])
    parser.add_argument("--data_dir",    default=None)
    parser.add_argument("--n_samples",   default=100, type=int,
                        help="验证集采样数（与 llm_ner.py 默认 100 条对齐）")
    parser.add_argument("--seed",        default=42,  type=int)
    parser.add_argument("--demo",        action="store_true",
                        help="只跑 5 条示例，快速演示")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 默认路径
    data_dir = Path(args.data_dir) if args.data_dir else (ROOT / "data" / ("cluener" if args.dataset == "cluener2020" else args.dataset))
    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else (ROOT / "outputs" / "sft_adapter")

    if not ckpt_dir.exists():
        print(f"[错误] checkpoint 目录不存在：{ckpt_dir}")
        print("请先运行 train_sft.py 或 train_sft_qlora.py 完成训练。")
        return

    entity_types = get_entity_types(args.dataset)
    system_prompt = build_system_prompt(args.dataset)
    print(f"数据集: {args.dataset} | 实体类型: {entity_types}")

    # ── 加载数据 ──────────────────────────────────────────────────────────────
    with open(data_dir / "validation.json", encoding="utf-8") as f:
        val_data = json.load(f)

    random.seed(args.seed)
    n = 5 if args.demo else args.n_samples
    samples = random.sample(val_data, min(n, len(val_data)))
    print(f"评估样本数: {len(samples)}\n")

    # ── 加载模型 ──────────────────────────────────────────────────────────────
    model_path = args.model_path or "Qwen/Qwen2.5-0.5B-Instruct"  # 默认 0.5B
    model, tokenizer = load_model(model_path, str(ckpt_dir), device)

    # ── 推理 ──────────────────────────────────────────────────────────────────
    all_golds, all_preds = [], []
    detail_records = []
    parse_fail = 0
    t0 = time.time()

    for i, record in enumerate(samples, 1):
        text  = record.get("text", "".join(record.get("tokens", [])))
        g_set = gold_spans_from_record(record, args.dataset)
        raw   = generate_ner(text, model, tokenizer, device, system_prompt)
        p_set = pred_spans_from_output(text, raw, entity_types)

        # JSON 解析失败：输出里没有合法的 {entities: [...]} 结构
        if not re.search(r"\{.*entities.*\}", raw, re.DOTALL):
            parse_fail += 1

        all_golds.append(g_set)
        all_preds.append(p_set)
        detail_records.append({
            "text": text,
            "gold":  [{"text": s, "type": t} for s, t, *_ in g_set],
            "pred":  [{"text": s, "type": t} for s, t, *_ in p_set],
            "raw_output": raw,
        })

        # 打印进度（取前3个 gold span 显示）
        tp_here  = len(g_set & p_set)
        status   = "✓" if g_set == p_set else ("~" if tp_here > 0 else "✗")
        gold_str = ",".join(f"{s}({t})" for s, t, *_ in list(g_set)[:3])
        print(f"[{i:3d}/{len(samples)}] {status}  gold:{gold_str or '无'}  |  {text[:30]}")

    elapsed = time.time() - t0
    metrics = compute_span_f1(all_golds, all_preds)

    # ── 读取已有结果做多方对比 ─────────────────────────────────────────────────
    bert_crf_f1   = "?"
    llm_zero_f1   = "?"
    llm_few_f1    = "?"
    crf_log_path  = LOG_DIR / "eval_crf_validation.json"
    llm_log_path  = LOG_DIR / "eval_llm.json"

    if crf_log_path.exists():
        with open(crf_log_path, encoding="utf-8") as f:
            crf_data = json.load(f)
        bert_crf_f1 = f"{crf_data.get('entity_f1', crf_data.get('f1', '?')):.4f}"

    if llm_log_path.exists():
        with open(llm_log_path, encoding="utf-8") as f:
            llm_data = json.load(f)
        llm_zero_f1 = f"{llm_data['zero_shot']['f1']:.4f}"
        llm_few_f1  = f"{llm_data['few_shot']['f1']:.4f}"

    print(f"\n{'='*65}")
    print(f"LLM SFT NER 评估结果")
    print(f"{'='*65}")
    print(f"  样本数      : {len(samples)}")
    print(f"  Precision   : {metrics['precision']:.4f}")
    print(f"  Recall      : {metrics['recall']:.4f}")
    print(f"  F1          : {metrics['f1']:.4f}")
    print(f"  JSON 解析失败: {parse_fail} 条 ({parse_fail/len(samples)*100:.1f}%)")
    print(f"  总耗时      : {elapsed:.1f}s，均值 {elapsed/len(samples):.2f}s/条")

    print(f"""
多方对比（验证集随机采样，seed=42）
  ┌──────────────────────────────────────────┬──────────┐
  │ 方法                                     │  F1      │
  ├──────────────────────────────────────────┼──────────┤
  │ BERT + CRF（全量数据，3 epoch）           │ {bert_crf_f1:<8} │
  │ Qwen API zero-shot（100 条）              │ {llm_zero_f1:<8} │
  │ Qwen API few-shot（100 条，3 例）         │ {llm_few_f1:<8} │
  │ Qwen2-0.5B SFT（LoRA，{len(samples)} 条样本）  │ {metrics['f1']:.4f}   │
  └──────────────────────────────────────────┴──────────┘

评估标准说明：
  SFT（本脚本）和 LLM API（llm_ner.py）均使用 span F1：
    text.find() 近似定位 → (text, type, start, end) 4元组 → 与 gold 做集合交集
  BERT+CRF（evaluate.py）使用 seqeval：
    BIO 解码出精确 token 边界 → 严格位置匹配（标准比上面略严，差异主要在多次出现的实体）

思考题：
  1. SFT 本地小模型 vs LLM API few-shot，谁的 F1 更高？为什么？
  2. NER 的 JSON 输出比分类的单词输出难控制，体现在哪里（parse_fail 数）？
  3. BERT+CRF 保证零非法序列，生成式 NER 有这个保证吗？如何处理？
  4. 如果给 SFT 模型也提供 few-shot 示例（系统提示里加样例），F1 会提升吗？
""")

    # ── 保存结果 ──────────────────────────────────────────────────────────────
    model_tag = Path(args.model_path or "sft").name
    # 从 ckpt_dir 提取区分标签：sft_adapter → 用父目录名；否则用目录名本身
    ckpt_path = Path(args.ckpt_dir) if args.ckpt_dir else Path("sft")
    ckpt_tag = ckpt_path.parent.name if ckpt_path.name == "sft_adapter" else ckpt_path.name
    out_path = LOG_DIR / f"eval_{model_tag}_{ckpt_tag}_{args.dataset}.json"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "metrics": {k: (v if isinstance(v, (int, float)) else v)
                        for k, v in metrics.items()},
            "n_samples": len(samples), "parse_fail": parse_fail,
            "detail": detail_records,
        }, f, ensure_ascii=False, indent=2)
    print(f"结果已保存 → {out_path}")


if __name__ == "__main__":
    main()
