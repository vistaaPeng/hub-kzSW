"""LLM SFT 评估 + 三方对比"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from common.config import DATA_DIR, OUTPUT_DIR, ADAPTER_DIR, QWEN_MODEL_NAME, LABEL_NAMES, ZERO_SHOT_DEFAULTS
from common.utils import get_device

SYSTEM_PROMPT = (
    "你是一个新闻标题分类助手。请将给定的新闻标题分类到以下类别之一，"
    "只输出类别名称，不要输出任何其他内容。\n"
    "可选类别：" + "、".join(LABEL_NAMES)
)


def classify_one(text, model, tokenizer, device):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"新闻标题：{text}\n类别："},
    ]
    input_ids = torch.tensor(
        [tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)],
        device=device
    )
    with torch.no_grad():
        output_ids = model.generate(
            input_ids, max_new_tokens=8, do_sample=False, pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0][input_ids.shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def parse_prediction(raw):
    for name in LABEL_NAMES:
        if name in raw:
            return name
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=ZERO_SHOT_DEFAULTS["num_samples"])
    parser.add_argument("--seed", type=int, default=ZERO_SHOT_DEFAULTS["seed"])
    args = parser.parse_args()

    device = get_device()
    print(f"设备: {device}")

    if not ADAPTER_DIR.exists():
        print(f"[错误] adapter 不存在：{ADAPTER_DIR}，请先运行 llm_sft/train.py")
        return

    # Data
    with open(DATA_DIR / "val.json", encoding="utf-8") as f:
        val_data = json.load(f)
    with open(DATA_DIR / "label_map.json", encoding="utf-8") as f:
        id2name = {int(k): v for k, v in json.load(f)["id2name"].items()}

    random.seed(args.seed)
    samples = random.sample(val_data, min(args.num_samples, len(val_data)))
    print(f"评估样本: {len(samples)}")

    # Model
    print(f"加载 base: {QWEN_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_NAME, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL_NAME, torch_dtype=torch.float32, trust_remote_code=True,
    )
    print(f"加载 adapter: {ADAPTER_DIR}")
    model = PeftModel.from_pretrained(base_model, str(ADAPTER_DIR))
    model = model.merge_and_unload().to(device)
    model.eval()

    # Eval
    correct = total = unparseable = 0
    t0 = time.time()

    for i, item in enumerate(samples):
        true_name = id2name[item["label"]]
        raw = classify_one(item["sentence"], model, tokenizer, device)
        pred_name = parse_prediction(raw)
        is_correct = (pred_name == true_name)
        if pred_name is None:
            unparseable += 1
        if is_correct:
            correct += 1
        total += 1
        status = "✓" if is_correct else ("?" if pred_name is None else "✗")
        print(f"[{i+1:3d}/{len(samples)}] {status} 真实:{true_name:4s} 预测:{str(pred_name):4s} | {item['sentence'][:30]}")

    elapsed = time.time() - t0
    acc = correct / total
    print(f"\nLLM SFT: acc={acc:.4f} ({correct}/{total})  unparseable={unparseable}  耗时={elapsed:.0f}s")

    # 三方对比
    print(f"\n{'='*60}")
    print(f"{'方案':<25} {'准确率':>8}")
    print(f"{'-'*60}")

    # BERT
    bert_train = OUTPUT_DIR / "train_log_cls.json"
    bert_acc_str = "（未训练）"
    if bert_train.exists():
        with open(bert_train, encoding="utf-8") as f:
            bert_log = json.load(f)
        bert_acc_str = f"{max(e['val_acc'] for e in bert_log):.4f}"

    # Zero-shot
    zs_path = OUTPUT_DIR / "llm_zero_shot_results.json"
    zs_acc_str = "（未运行）"
    if zs_path.exists():
        with open(zs_path, encoding="utf-8") as f:
            zs = json.load(f)
        zs_acc_str = f"{zs['accuracy']:.4f}"

    print(f"{'BERT fine-tune':<25} {bert_acc_str:>8}")
    print(f"{'LLM zero-shot':<25} {zs_acc_str:>8}")
    print(f"{'LLM SFT (LoRA)':<25} {acc:>8.4f}")
    print(f"{'='*60}")

    # Save
    out_path = OUTPUT_DIR / "llm_sft_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"accuracy": acc, "total": total, "correct": correct, "unparseable": unparseable}, f, ensure_ascii=False, indent=2)
    print(f"结果 → {out_path}")


if __name__ == "__main__":
    main()
