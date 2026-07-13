import torch

# 设备配置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 模型超参数
D_MODEL = 128
N_LAYER = 3
N_HEAD = 4
D_FF = 256
SEQ_LEN = 64
DROPOUT = 0.1

# 训练超参数
BATCH_SIZE = 16
EPOCHS = 200
LR = 1e-4
SAVE_PATH = "./checkpoints/lm_model.pth"

# 生成参数
MAX_GEN_LEN = 50