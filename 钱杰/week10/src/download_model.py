import os
import requests
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
MODEL_DIR = BASE_DIR / "model_cache" / "bert-base-chinese"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FILES = [
    "config.json",
    "tokenizer_config.json",
    "vocab.txt",
    "pytorch_model.bin",
    "special_tokens_map.json",
]

BASE_URL = "https://huggingface.co/bert-base-chinese/resolve/main/"


def download_file(filename):
    url = BASE_URL + filename
    filepath = MODEL_DIR / filename
    
    if filepath.exists():
        print(f"跳过 {filename}（已存在）")
        return
    
    print(f"下载 {filename}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0
    
    with open(filepath, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    progress = (downloaded / total_size) * 100
                    print(f"\r进度: {progress:.1f}%", end='')
    
    print(f"\n完成: {filename}")


def main():
    for filename in FILES:
        try:
            download_file(filename)
        except Exception as e:
            print(f"下载 {filename} 失败: {e}")
    
    print("\n模型下载完成！")


if __name__ == "__main__":
    main()