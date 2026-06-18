from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from model import BiEncoder, CrossEncoder
from text_matching_utils import build_dataloader, compute_metrics, ensure_dir, get_tokenizer, load_samples, save_json


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = ROOT / "data" / "bq_corpus" / "test.jsonl"
DEFAULT_CHECKPOINT = ROOT / "outputs" / "checkpoints" / "best_model.pt"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained text matching model.")
    parser.add_argument("--data_path", type=str, default=str(DEFAULT_DATASET))
    parser.add_argument("--model_path", type=str, default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--model_name_or_path", type=str, default=str(ROOT.parent.parent / "pretrained_models" / "bert-base-chinese"))
    parser.add_argument("--output_dir", type=str, default=str(ROOT / "outputs"))
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_path = Path(args.data_path)
    model_path = Path(args.model_path)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir / "reports")

    samples = load_samples(data_path)
    if not samples:
        raise ValueError(f"No valid evaluation samples found in {data_path}")

    checkpoint = torch.load(model_path, map_location=device)
    model_type = checkpoint.get("model_type", "biencoder")
    tokenizer = get_tokenizer(args.model_name_or_path)

    if model_type == "crossencoder":
        model = CrossEncoder(args.model_name_or_path).to(device)
    else:
        model = BiEncoder(args.model_name_or_path).to(device)

    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    loader = build_dataloader(
        samples,
        tokenizer,
        batch_size=16,
        max_length=128,
        cross_encoder=(model_type == "crossencoder"),
    )

    preds = []
    truths = []
    with torch.no_grad():
        for batch in loader:
            if model_type == "crossencoder":
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                logits = model(input_ids, attention_mask)
                pred = (torch.sigmoid(logits) > 0.5).long()
            else:
                input_ids_1 = batch["input_ids_1"].to(device)
                attention_mask_1 = batch["attention_mask_1"].to(device)
                input_ids_2 = batch["input_ids_2"].to(device)
                attention_mask_2 = batch["attention_mask_2"].to(device)
                emb1, emb2 = model(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2)
                pred = (torch.nn.functional.cosine_similarity(emb1, emb2) > 0.5).long()

            preds.extend(pred.detach().cpu().tolist())
            truths.extend(batch["labels"].detach().cpu().tolist())

    metrics = compute_metrics(truths, preds)

    report = {
        "dataset": str(data_path),
        "model_path": str(model_path),
        "metrics": metrics,
        "sample_count": len(samples),
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))
    save_json(report, output_dir / "reports" / "evaluation_report.json")


if __name__ == "__main__":
    main()
