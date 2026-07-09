"""跨平台路径与显存配置（Windows / WSL2 自动识别）"""
import os
import platform

_WIN_MODEL = r"E:\DeepLearning\week9\pretrain_models\Qwen2-0.5B-Instruct"
_WSL_MODEL = "/mnt/e/DeepLearning/week9/pretrain_models/Qwen2-0.5B-Instruct"

if os.environ.get("VLLM_MODEL_PATH"):
    MODEL_PATH = os.environ["VLLM_MODEL_PATH"]
elif platform.system() == "Windows":
    MODEL_PATH = _WIN_MODEL
else:
    MODEL_PATH = _WSL_MODEL

# GTX 1060 6GB 显存较紧
GPU_MEMORY_UTILIZATION = float(os.environ.get("GPU_MEM_UTIL", "0.45"))
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "1024"))
SERVED_MODEL_NAME = "qwen2-0.5b"
SERVER_PORT = 8000
