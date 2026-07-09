"""
Ollama vs Transformers 吞吐对比（GTX 1060 可用）

前置条件：
  1. 已安装 Ollama：https://ollama.com
  2. 已拉取模型：ollama pull qwen2:0.5b
  3. Ollama 服务运行中（安装后通常自动启动）

使用方式：
  conda activate py312
  cd work9/src
  python bench_ollama.py
  python bench_ollama.py --skip-transformers   # 仅测 Ollama（transformers 已有数据时）
"""

import argparse
import gc
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import requests
import torch

from config import MODEL_PATH

N_PROMPTS = 50
MAX_NEW_TOKENS = 100
BATCH_SIZE = 8
OLLAMA_MODEL = "qwen2:0.5b"
OLLAMA_URL = "http://localhost:11434/api/generate"

SHORT_QUESTIONS = [
    "什么是股票？", "什么是基金？", "什么是ETF？", "什么是债券？", "什么是期权？",
    "什么是熊市？", "什么是牛市？", "什么是PE？", "什么是ROE？", "什么是毛利率？",
]
MEDIUM_QUESTIONS = [
    "解释一下价值投资和趋势投资的区别。",
    "什么情况下应该止损？",
    "为什么会出现股市崩盘？",
    "沪深300和中证500有什么区别？",
    "什么是量化交易？",
    "基金定投的优势是什么？",
    "股票回购对股价有什么影响？",
    "可转债有哪些特点？",
    "如何判断一家公司是否值得投资？",
    "什么是做市商制度？",
]
LONG_QUESTIONS = [
    "请详细介绍一下巴菲特的投资理念及其核心原则，并举例说明。",
    "解释下现金流折现（DCF）估值法的基本步骤、使用的参数以及它的局限性。",
    "比较A股和美股在交易制度、监管环境、投资者结构等方面的主要差异。",
    "什么是技术分析？它和基本面分析有什么区别？两种方法各自的适用场景是什么？",
    "详细解释资产配置的核心思想，常见的几种配置模型，以及如何根据个人风险偏好调整。",
]
PROMPTS = (SHORT_QUESTIONS * 3 + MEDIUM_QUESTIONS * 1 + LONG_QUESTIONS * 2)[:N_PROMPTS]


def check_ollama():
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        r.raise_for_status()
        names = [m["name"] for m in r.json().get("models", [])]
        if not any(OLLAMA_MODEL in n for n in names):
            print(f"错误：未找到模型 {OLLAMA_MODEL}，请先执行: ollama pull {OLLAMA_MODEL}")
            print(f"已安装模型: {names}")
            sys.exit(1)
        print(f"Ollama 就绪，模型: {names}")
    except requests.exceptions.ConnectionError:
        print("错误：Ollama 未运行。请启动 Ollama 应用或执行 ollama serve")
        sys.exit(1)


def bench_ollama_serial(prompts: list[str]) -> dict:
    print("\n[C1] Ollama 串行（逐条 API 请求）...")
    total_tokens = 0
    t0 = time.time()
    for i, q in enumerate(prompts):
        r = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL,
            "prompt": q,
            "stream": False,
            "options": {"num_predict": MAX_NEW_TOKENS, "temperature": 0},
        }, timeout=120)
        r.raise_for_status()
        data = r.json()
        total_tokens += data.get("eval_count", 0)
        if (i + 1) % 10 == 0:
            print(f"    进度 {i+1}/{len(prompts)}")
    dt = time.time() - t0
    return {"time": dt, "gen_tokens": total_tokens,
            "qps": len(prompts) / dt, "tps": total_tokens / dt}


def bench_ollama_concurrent(prompts: list[str], workers: int = 4) -> dict:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    print(f"\n[C2] Ollama 并发={workers}（模拟多请求调度）...")

    def one_request(q: str) -> int:
        r = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL,
            "prompt": q,
            "stream": False,
            "options": {"num_predict": MAX_NEW_TOKENS, "temperature": 0},
        }, timeout=120)
        r.raise_for_status()
        return r.json().get("eval_count", 0)

    total_tokens = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(one_request, q): i for i, q in enumerate(prompts)}
        done = 0
        for f in as_completed(futures):
            total_tokens += f.result()
            done += 1
            if done % 10 == 0:
                print(f"    进度 {done}/{len(prompts)}")
    dt = time.time() - t0
    return {"time": dt, "gen_tokens": total_tokens,
            "qps": len(prompts) / dt, "tps": total_tokens / dt}


def bench_transformers(prompts: list[str]) -> dict:
    print("\n" + "=" * 70)
    print(f"  加载 transformers | 模型: {MODEL_PATH}")
    print("=" * 70)
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, device_map="cuda",
    )
    model.eval()

    def make_prompt(q: str) -> str:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": q}],
            tokenize=False, add_generation_prompt=True,
        )

    chat_prompts = [make_prompt(q) for q in prompts]

    print("\n[A] transformers 串行...")
    total_a = 0
    t0 = time.time()
    for i, p in enumerate(chat_prompts):
        inputs = tokenizer(p, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                 do_sample=False, pad_token_id=tokenizer.pad_token_id)
        total_a += len(out[0]) - len(inputs["input_ids"][0])
        if (i + 1) % 10 == 0:
            print(f"    进度 {i+1}/{len(chat_prompts)}")
    dt_a = time.time() - t0

    print(f"\n[B] transformers batch={BATCH_SIZE}...")
    tokenizer.padding_side = "left"
    total_b = 0
    t0 = time.time()
    for i in range(0, len(chat_prompts), BATCH_SIZE):
        batch = chat_prompts[i:i + BATCH_SIZE]
        enc = tokenizer(batch, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS,
                                 do_sample=False, pad_token_id=tokenizer.pad_token_id)
        gen = out[:, enc["input_ids"].shape[1]:]
        for row in gen:
            total_b += (row != tokenizer.pad_token_id).sum().item()
    dt_b = time.time() - t0

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "serial": {"time": dt_a, "gen_tokens": total_a, "qps": len(prompts)/dt_a, "tps": total_a/dt_a},
        "batch": {"time": dt_b, "gen_tokens": total_b, "qps": len(prompts)/dt_b, "tps": total_b/dt_b},
    }


def plot_results(results: dict, out_path: str, gpu_name: str):
    modes = [
        "transformers\nserial", f"transformers\nbatch={BATCH_SIZE}",
        "Ollama\nserial", "Ollama\nconcurrent=4",
    ]
    keys = ["serial", "batch", "ollama_serial", "ollama_concurrent"]
    times = [results[k]["time"] for k in keys]
    qps = [results[k]["qps"] for k in keys]
    tps = [results[k]["tps"] for k in keys]
    colors = ["#aab7c4", "#82b1ff", "#ffb74d", "#ff8a65"]
    best_qps = max(qps)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, vals, ylabel, title in zip(
        axes, [times, qps, tps],
        ["Time (seconds)", "QPS (requests/sec)", "Tokens / sec"],
        [f"Total Time ({N_PROMPTS} requests)", "Requests Per Second", "Generation Throughput"],
    ):
        bars = ax.bar(modes, vals, color=colors)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        for b, v in zip(bars, vals):
            fmt = f"{v:.1f}s" if "Time" in title else (f"{v:.0f}" if v > 100 else f"{v:.2f}")
            ax.text(b.get_x() + b.get_width()/2, v, fmt, ha="center", va="bottom", fontsize=9)

    plt.suptitle(f"Ollama vs Transformers ({gpu_name}, {OLLAMA_MODEL})", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n柱状图: {out_path}")


def print_summary(results: dict):
    names = {
        "serial": "[A] transformers 串行",
        "batch": f"[B] transformers batch={BATCH_SIZE}",
        "ollama_serial": "[C1] Ollama 串行",
        "ollama_concurrent": "[C2] Ollama 并发=4",
    }
    best = max(results[k]["qps"] for k in names)
    print("\n" + "=" * 70)
    print("  结果汇总")
    print("=" * 70)
    print(f"{'模式':<32}{'总耗时':<12}{'QPS':<10}{'tokens/s':<12}{'相对最快':<10}")
    print("-" * 80)
    for k in names:
        r = results[k]
        print(f"{names[k]:<30}{r['time']:>6.2f}s     {r['qps']:>5.2f}     {r['tps']:>6.0f}      {r['qps']/best:>5.2f}x")

    o_best = max(results["ollama_serial"]["qps"], results["ollama_concurrent"]["qps"])
    print(f"\n  Ollama 相对 transformers 串行: {o_best/results['serial']['qps']:.1f}x")
    print(f"  Ollama 相对 transformers batch: {o_best/results['batch']['qps']:.1f}x")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-transformers", action="store_true")
    args = parser.parse_args()

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print("=" * 70)
    print(f"  Ollama vs Transformers | GPU: {gpu_name}")
    print(f"  {N_PROMPTS} prompts x max {MAX_NEW_TOKENS} tokens")
    print("=" * 70)

    check_ollama()

    if args.skip_transformers:
        out_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
        prev = os.path.join(out_dir, "throughput_results.json")
        if os.path.isfile(prev):
            with open(prev, encoding="utf-8") as f:
                tf = json.load(f)["results"]
            tf = {"serial": tf["serial"], "batch": tf["batch"]}
            print("复用已有 transformers 数据")
        else:
            print("错误：无 transformers 数据，去掉 --skip-transformers")
            sys.exit(1)
    else:
        tf = bench_transformers(PROMPTS)

    o_serial = bench_ollama_serial(PROMPTS)
    o_conc = bench_ollama_concurrent(PROMPTS, workers=4)

    results = {**tf, "ollama_serial": o_serial, "ollama_concurrent": o_conc}
    print_summary(results)

    out_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
    os.makedirs(out_dir, exist_ok=True)
    meta = {
        "gpu": gpu_name,
        "ollama_model": OLLAMA_MODEL,
        "n_prompts": N_PROMPTS,
        "max_new_tokens": MAX_NEW_TOKENS,
        "results": results,
    }
    json_path = os.path.join(out_dir, "ollama_comparison_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"\nJSON: {json_path}")

    plot_results(results, os.path.join(out_dir, "ollama_comparison.png"), gpu_name)


if __name__ == "__main__":
    main()
