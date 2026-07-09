import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from huggingface_hub import hf_hub_download

repo = "Qwen/Qwen2-0.5B-Instruct-GGUF"
target = "qwen2-0_5b-instruct-q4_k_m.gguf"
out_dir = "/mnt/e/DeepLearning/week9/pretrain_models/Qwen2-0.5B-GGUF"
print("downloading:", target)
p = hf_hub_download(repo, target, local_dir=out_dir)
print("done:", p)
