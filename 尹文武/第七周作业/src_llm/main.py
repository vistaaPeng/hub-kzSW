import argparse
import json
import os
import random
from pathlib import Path
from typing import List

import torch
from tqdm.auto import tqdm

from data_utils import (
    build_messages,
    build_sft_record,
    choose_demonstrations,
    compute_metrics,
    load_labels,
    load_samples,
    parse_pred_tags,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "src_llm"


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def call_openai_compatible(messages, args) -> str:
    from openai import OpenAI

    client = OpenAI(
        api_key=args.api_key or os.getenv("OPENAI_API_KEY"),
        base_url=args.base_url or os.getenv("OPENAI_BASE_URL"),
    )
    resp = client.chat.completions.create(
        model=args.api_model,
        messages=messages,
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
    )
    return resp.choices[0].message.content or "[]"


def api_predict(samples, labels, examples, args):
    y_true, y_pred, rows = [], [], []
    for sample in tqdm(samples, desc=f"{args.mode} inference"):
        messages = build_messages(sample["tokens"], labels, examples)
        raw = call_openai_compatible(messages, args)
        pred_tags = parse_pred_tags(raw, sample["tokens"], labels)
        y_true.append(sample["ner_tags"])
        y_pred.append(pred_tags)
        rows.append({"tokens": sample["tokens"], "gold": sample["ner_tags"], "pred": pred_tags, "raw": raw})
    return y_true, y_pred, rows


def build_training_dataset(train_samples, labels, args):
    from datasets import Dataset

    demonstrations = choose_demonstrations(train_samples, args.num_shots, args.seed)
    records = [
        build_sft_record(sample, labels, demonstrations)
        for sample in train_samples[: args.train_limit or None]
    ]
    return Dataset.from_list(records)


def train_adapter(train_samples, labels, args) -> None:
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    if not args.model_name_or_path:
        raise ValueError("--model_name_or_path is required for lora/qlora modes")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = None
    if args.mode == "qlora":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=quant_config,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )
    if args.mode == "qlora":
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules.split(",") if args.target_modules else None,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = build_training_dataset(train_samples, labels, args)

    def tokenize(record):
        text = record["prompt"] + record["response"] + tokenizer.eos_token
        encoded = tokenizer(text, truncation=True, max_length=args.max_length)
        encoded["labels"] = encoded["input_ids"].copy()
        return encoded

    tokenized = dataset.map(tokenize, remove_columns=dataset.column_names)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=torch.cuda.is_available(),
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))


def load_adapter_model(args):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = args.model_name_or_path
    adapter = args.adapter_path or args.output_dir
    if not base:
        raise ValueError("--model_name_or_path is required when validating an adapter")

    tokenizer = AutoTokenizer.from_pretrained(adapter, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tokenizer


def local_generate(model, tokenizer, messages, args) -> str:
    prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant: "
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_length).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=args.temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True)


def adapter_predict(samples, labels, examples, args):
    model, tokenizer = load_adapter_model(args)
    y_true, y_pred, rows = [], [], []
    for sample in tqdm(samples, desc="adapter validation"):
        raw = local_generate(model, tokenizer, build_messages(sample["tokens"], labels, examples), args)
        pred_tags = parse_pred_tags(raw, sample["tokens"], labels)
        y_true.append(sample["ner_tags"])
        y_pred.append(pred_tags)
        rows.append({"tokens": sample["tokens"], "gold": sample["ner_tags"], "pred": pred_tags, "raw": raw})
    return y_true, y_pred, rows


def save_results(args, metrics, rows) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{args.mode}_{args.eval_split}_predictions.json", "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "predictions": rows}, f, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description="LLM-based NER sequence labeling")
    parser.add_argument("--mode", choices=["one_shot", "few_shot", "lora", "qlora", "validate_adapter"], default="few_shot")
    parser.add_argument("--data_dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--eval_split", choices=["train", "validation", "test"], default="validation")
    parser.add_argument("--eval_limit", type=int, default=50)
    parser.add_argument("--train_limit", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--api_model", default=os.getenv("OPENAI_MODEL", "deepseek-v4-flash"))
    parser.add_argument("--base_url", default=os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com"))
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--num_shots", type=int, default=5)

    parser.add_argument("--model_name_or_path", default=None)
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", default="q_proj,k_proj,v_proj,o_proj")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--skip_eval_after_train", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    labels = load_labels(args.data_dir)
    train_samples = load_samples(args.data_dir, "train")
    eval_samples = load_samples(args.data_dir, args.eval_split, args.eval_limit)

    if args.mode == "one_shot":
        examples = choose_demonstrations(train_samples, 1, args.seed)
        y_true, y_pred, rows = api_predict(eval_samples, labels, examples, args)
    elif args.mode == "few_shot":
        examples = choose_demonstrations(train_samples, args.num_shots, args.seed)
        y_true, y_pred, rows = api_predict(eval_samples, labels, examples, args)
    elif args.mode in {"lora", "qlora"}:
        train_adapter(train_samples, labels, args)
        if args.skip_eval_after_train:
            return
        examples = choose_demonstrations(train_samples, args.num_shots, args.seed)
        y_true, y_pred, rows = adapter_predict(eval_samples, labels, examples, args)
    else:
        examples = choose_demonstrations(train_samples, args.num_shots, args.seed)
        y_true, y_pred, rows = adapter_predict(eval_samples, labels, examples, args)

    metrics = compute_metrics(y_true, y_pred)
    save_results(args, metrics, rows)


if __name__ == "__main__":
    main()
