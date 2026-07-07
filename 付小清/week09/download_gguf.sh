#!/bin/bash
set -e
source ~/vllm_env/bin/activate
export HF_ENDPOINT=https://hf-mirror.com
python << 'PY'
from huggingface_hub import hf_hub_download
p = hf_hub_download(
    repo_id="Qwen/Qwen2-0.5B-Instruct-GGUF",
    filename="qwen2-0.5b-instruct-q4_k_m.gguf",
    local_dir="/mnt/e/DeepLearning/week9/pretrain_models/Qwen2-0.5B-GGUF",
)
print("done:", p)
PY
