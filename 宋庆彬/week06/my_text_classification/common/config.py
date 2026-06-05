"""统一配置：路径 + 超参数 + 标签定义"""

from pathlib import Path

# ── 项目根目录 ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
CKPT_DIR = OUTPUT_DIR / "checkpoints"
ADAPTER_DIR = OUTPUT_DIR / "sft_adapter"

# ── 预训练模型（HF 名称或本地路径）──────────────────────────────────────────
BERT_MODEL_NAME = "bert-base-chinese"
QWEN_MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"

# ── TNEWS 15 类标签 ─────────────────────────────────────────────────────────
LABEL_NAMES = [
    "故事", "文化", "娱乐", "体育", "财经",
    "房产", "汽车", "教育", "科技", "军事",
    "旅游", "国际", "证券", "农业", "电竞",
]
NUM_LABELS = len(LABEL_NAMES)

# ── BERT Fine-tune 默认超参（速度优先）───────────────────────────────────────
BERT_DEFAULTS = {
    "num_train": 10000,
    "epochs": 2,
    "batch_size": 32,
    "max_length": 64,
    "lr": 2e-5,
    "head_lr_mult": 5.0,
    "dropout": 0.1,
    "warmup_ratio": 0.1,
    "grad_accum": 1,
    "pool": "cls",
    "early_stopping_patience": 2,
    "val_subset": 2000,
}

# ── LLM Zero-shot 默认超参 ───────────────────────────────────────────────────
ZERO_SHOT_DEFAULTS = {
    "num_samples": 100,
    "max_new_tokens": 8,
    "seed": 42,
}

# ── LLM SFT LoRA 默认超参（速度优先）─────────────────────────────────────────
SFT_DEFAULTS = {
    "num_train": 2000,
    "epochs": 2,
    "batch_size": 2,
    "max_length": 128,
    "grad_accum": 4,
    "lr": 2e-4,
    "lora_r": 4,
    "lora_alpha": 8,
    "lora_dropout": 0.05,
    "val_subset": 200,
    "seed": 42,
}
