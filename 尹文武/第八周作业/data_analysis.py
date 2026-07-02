from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from text_matching_utils import ensure_dir, load_samples, save_json


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT / "outputs"
DEFAULT_FIGURE_DIR = DEFAULT_OUTPUT_DIR / "figures"
DEFAULT_LOG_DIR = DEFAULT_OUTPUT_DIR / "logs"


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze text matching datasets.")
    parser.add_argument("--data_path", type=str, default=str(ROOT / "data" / "bq_corpus" / "train.jsonl"))
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def safe_len(text):
    return len(str(text)) if text is not None else 0


def analyze_dataset(samples, output_dir: Path):
    ensure_dir(output_dir)
    ensure_dir(output_dir / "figures")
    ensure_dir(output_dir / "logs")

    counts = len(samples)
    labels = [sample["label"] for sample in samples]
    label_counter = Counter(labels)
    text1_lengths = [safe_len(sample["text1"]) for sample in samples]
    text2_lengths = [safe_len(sample["text2"]) for sample in samples]
    length_diffs = [abs(a - b) for a, b in zip(text1_lengths, text2_lengths)]

    missing = {
        "text1_missing": sum(1 for s in samples if not s.get("text1")),
        "text2_missing": sum(1 for s in samples if not s.get("text2")),
        "label_missing": sum(1 for s in samples if s.get("label") is None),
    }

    seen = {}
    duplicates = 0
    for sample in samples:
        key = (sample.get("text1"), sample.get("text2"), sample.get("label"))
        seen[key] = seen.get(key, 0) + 1
    duplicates = sum(v - 1 for v in seen.values() if v > 1)

    stats = {
        "sample_count": counts,
        "label_distribution": dict(sorted(label_counter.items())),
        "positive_ratio": float(sum(1 for x in labels if x == 1) / counts) if counts else 0.0,
        "negative_ratio": float(sum(1 for x in labels if x == 0) / counts) if counts else 0.0,
        "text1_length_stats": {
            "mean": float(np.mean(text1_lengths)) if text1_lengths else 0.0,
            "median": float(np.median(text1_lengths)) if text1_lengths else 0.0,
            "max": int(max(text1_lengths)) if text1_lengths else 0,
            "min": int(min(text1_lengths)) if text1_lengths else 0,
            "p95": float(np.percentile(text1_lengths, 95)) if text1_lengths else 0.0,
        },
        "text2_length_stats": {
            "mean": float(np.mean(text2_lengths)) if text2_lengths else 0.0,
            "median": float(np.median(text2_lengths)) if text2_lengths else 0.0,
            "max": int(max(text2_lengths)) if text2_lengths else 0,
            "min": int(min(text2_lengths)) if text2_lengths else 0,
            "p95": float(np.percentile(text2_lengths, 95)) if text2_lengths else 0.0,
        },
        "length_diff_stats": {
            "mean": float(np.mean(length_diffs)) if length_diffs else 0.0,
            "median": float(np.median(length_diffs)) if length_diffs else 0.0,
            "max": int(max(length_diffs)) if length_diffs else 0,
            "p95": float(np.percentile(length_diffs, 95)) if length_diffs else 0.0,
        },
        "missing_values": missing,
        "duplicate_samples": duplicates,
    }

    log_path = output_dir / "logs" / "data_analysis.log"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(stats, ensure_ascii=False, indent=2))

    # label distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    labels_plot = [0, 1]
    counts_plot = [label_counter.get(0, 0), label_counter.get(1, 0)]
    ax.bar(labels_plot, counts_plot, color=["#e57373", "#64b5f6"])
    ax.set_xticks(labels_plot)
    ax.set_xticklabels(["negative", "positive"])
    ax.set_ylabel("count")
    ax.set_title("Label distribution")
    fig.tight_layout()
    fig.savefig(output_dir / "figures" / "label_distribution.png", dpi=150)
    plt.close(fig)

    # text length distribution
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(text1_lengths, bins=30, alpha=0.5, label="text1")
    ax.hist(text2_lengths, bins=30, alpha=0.5, label="text2")
    ax.set_xlabel("character length")
    ax.set_ylabel("frequency")
    ax.set_title("Text length distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "figures" / "text_length_distribution.png", dpi=150)
    plt.close(fig)

    # length diff distribution
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(length_diffs, bins=30, color="#8e44ad", alpha=0.7)
    ax.set_xlabel("abs(len(text1)-len(text2))")
    ax.set_ylabel("frequency")
    ax.set_title("Length difference distribution")
    fig.tight_layout()
    fig.savefig(output_dir / "figures" / "length_difference_distribution.png", dpi=150)
    plt.close(fig)

    return stats


def main():
    args = parse_args()
    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)

    samples = load_samples(data_path)
    if not samples:
        raise ValueError(f"No valid samples were found in {data_path}")

    stats = analyze_dataset(samples, output_dir)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
