"""
BERT 新闻分类 — 配置文件
"""
import os

# HuggingFace 缓存路径 (必须在 import transformers 之前设置)
os.environ["HF_HOME"] = "M:\\huggingface_cache"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"  # 关闭 Windows 软链警告

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CNEWS_DIR = os.path.join(os.path.dirname(BASE_DIR), "cnews")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
MODEL_DIR = os.path.join(BASE_DIR, "checkpoints")
LOG_DIR = os.path.join(BASE_DIR, "logs")

TRAIN_FILE = os.path.join(CNEWS_DIR, "cnews.train.txt")
VAL_FILE = os.path.join(CNEWS_DIR, "cnews.val.txt")
TEST_FILE = os.path.join(CNEWS_DIR, "cnews.test.txt")

TRAIN_X = os.path.join(PROCESSED_DIR, "train_input_ids.pt")
TRAIN_MASK = os.path.join(PROCESSED_DIR, "train_attention_mask.pt")
TRAIN_Y = os.path.join(PROCESSED_DIR, "train_labels.pt")
VAL_X = os.path.join(PROCESSED_DIR, "val_input_ids.pt")
VAL_MASK = os.path.join(PROCESSED_DIR, "val_attention_mask.pt")
VAL_Y = os.path.join(PROCESSED_DIR, "val_labels.pt")
TEST_X = os.path.join(PROCESSED_DIR, "test_input_ids.pt")
TEST_MASK = os.path.join(PROCESSED_DIR, "test_attention_mask.pt")
TEST_Y = os.path.join(PROCESSED_DIR, "test_labels.pt")
LABEL_MAP_FILE = os.path.join(PROCESSED_DIR, "label_map.json")

# ---- BERT 参数 ----
MODEL_NAME = "bert-base-chinese"
MAX_LEN = 256                       # BERT 最大 512，256 覆盖大部分中文新闻
NUM_CLASSES = 10

# ---- 训练参数 (1080Ti 11GB) ----
BATCH_SIZE = 16                     # BERT 单卡不能太大
GRADIENT_ACCUMULATION = 4           # 梯度累积，等效 batch = 16 * 4 = 64
EPOCHS = 5                          # BERT 微调通常 3-5 轮
LEARNING_RATE = 2e-5                # BERT 标准微调学习率
WARMUP_RATIO = 0.1                  # warmup 步数比例
WEIGHT_DECAY = 0.01                 # BERT 标准权重衰减
EARLY_STOP_PATIENCE = 2
MAX_GRAD_NORM = 1.0

# ---- DataLoader ----
NUM_WORKERS = 0
PIN_MEMORY = True
USE_AMP = False                     # 1080 Ti 无 Tensor Cores

for d in [PROCESSED_DIR, MODEL_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)
