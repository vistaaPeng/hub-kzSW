"""
吞吐对比：transformers 串行 / transformers batch=8 / vLLM 批处理

使用方式：
  conda activate py312
  cd work9/src
  python bench_throughput.py              # 完整三路对比（需 vLLM + CUDA sm_70+）
  python bench_throughput.py --skip-vllm  # 仅跑 transformers baseline（Windows 可用）
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
import torch

from config import MODEL_PATH, GPU_MEMORY_UTILIZATION, MAX_MODEL_LEN

N_PROMPTS = 50
MAX_NEW_TOKENS = 100
BATCH_SIZE = 8

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


def bench_transformers(prompts: list[str]) -> dict:
    print("\n" + "=" * 70)
    print(f"  加载 transformers | 模型: {MODEL_PATH}")
    print("=" * 70)
    from transformers import AutoTokenizer, AutoModelForCausalLM

    if not os.path.isdir(MODEL_PATH):
        print(f"错误：模型目录不存在 {MODEL_PATH}")
        sys.exit(1)

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

    print("\n[A] transformers 串行（一次一条）...")
    total_tokens_a = 0
    t0 = time.time()
    for i, p in enumerate(chat_prompts):
        inputs = tokenizer(p, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False, pad_token_id=tokenizer.pad_token_id,
            )
        gen_ids = out[0, inputs["input_ids"].shape[1]:]
        total_tokens_a += len(gen_ids)
        if (i + 1) % 10 == 0:
            print(f"    进度 {i+1}/{len(chat_prompts)}")
    dt_a = time.time() - t0

    print(f"\n[B] transformers batch={BATCH_SIZE}（手动 padding）...")
    tokenizer.padding_side = "left"
    total_tokens_b = 0
    t0 = time.time()
    for i in range(0, len(chat_prompts), BATCH_SIZE):
        batch = chat_prompts[i:i + BATCH_SIZE]
        enc = tokenizer(batch, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False, pad_token_id=tokenizer.pad_token_id,
            )
        gen_ids = out[:, enc["input_ids"].shape[1]:]
        for row in gen_ids:
            total_tokens_b += (row != tokenizer.pad_token_id).sum().item()
        print(f"    进度 batch {i//BATCH_SIZE + 1}/{(len(chat_prompts)+BATCH_SIZE-1)//BATCH_SIZE}")
    dt_b = time.time() - t0

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "serial": {"time": dt_a, "gen_tokens": total_tokens_a,
                   "qps": len(prompts) / dt_a, "tps": total_tokens_a / dt_a},
        "batch": {"time": dt_b, "gen_tokens": total_tokens_b,
                  "qps": len(prompts) / dt_b, "tps": total_tokens_b / dt_b},
    }


def bench_vllm(prompts: list[str]) -> dict:
    print("\n" + "=" * 70)
    print(f"  加载 vLLM | 模型: {MODEL_PATH}")
    print("=" * 70)
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=MODEL_PATH,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        dtype="float16",
        enforce_eager=True,
    )
    tokenizer = llm.get_tokenizer()
    chat_prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": q}],
            tokenize=False, add_generation_prompt=True,
        )
        for q in prompts
    ]

    print("\n[C] vLLM 批处理（continuous batching）...")
    sampling = SamplingParams(temperature=0, max_tokens=MAX_NEW_TOKENS)
    t0 = time.time()
    outputs = llm.generate(chat_prompts, sampling)
    dt_c = time.time() - t0
    total_tokens_c = sum(len(o.outputs[0].token_ids) for o in outputs)

    del llm
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "vllm": {"time": dt_c, "gen_tokens": total_tokens_c,
                 "qps": len(prompts) / dt_c, "tps": total_tokens_c / dt_c},
    }


def plot_results(r: dict, out_path: str, gpu_name: str):
    modes = ["transformers\nserial", f"transformers\nbatch={BATCH_SIZE}", "vLLM\ncontinuous\nbatching"]
    times = [r["serial"]["time"], r["batch"]["time"], r["vllm"]["time"]]
    qps = [r["serial"]["qps"], r["batch"]["qps"], r["vllm"]["qps"]]
    tps = [r["serial"]["tps"], r["batch"]["tps"], r["vllm"]["tps"]]
    colors = ["#aab7c4", "#82b1ff", "#69f0ae"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, vals, ylabel, title in zip(
        axes,
        [times, qps, tps],
        ["Time (seconds)", "QPS (requests/sec)", "Tokens / sec (generated)"],
        [f"Total Time ({N_PROMPTS} requests)", "Requests Per Second", "Generation Throughput"],
    ):
        bars = ax.bar(modes, vals, color=colors)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        for b, v in zip(bars, vals):
            fmt = f"{v:.1f}s" if "Time" in title else (f"{v:.0f}" if v > 100 else f"{v:.1f}")
            ax.text(b.get_x() + b.get_width() / 2, v, fmt, ha="center", va="bottom")

    plt.suptitle(f"vLLM vs Transformers Throughput ({gpu_name})", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n柱状图已保存：{out_path}")


def print_summary(results: dict):
    print("\n" + "=" * 70)
    print("  结果汇总")
    print("=" * 70)
    print(f"{'模式':<30}{'总耗时':<12}{'QPS':<10}{'tokens/s':<12}{'相对vLLM':<10}")
    print("-" * 80)
    base = results["vllm"]["qps"]
    names = {"serial": "[A] transformers 串行",
             "batch": f"[B] transformers batch={BATCH_SIZE}",
             "vllm": "[C] vLLM 批处理"}
    for k in ["serial", "batch", "vllm"]:
        r = results[k]
        print(f"{names[k]:<28}{r['time']:>6.2f}s     "
              f"{r['qps']:>5.2f}     {r['tps']:>6.0f}      {r['qps']/base:>5.2f}x")

    print("\n  核心结论：")
    print(f"    vLLM 相对 transformers 串行加速：{results['vllm']['qps']/results['serial']['qps']:.1f}x")
    print(f"    vLLM 相对 transformers batch:    {results['vllm']['qps']/results['batch']['qps']:.1f}x")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-vllm", action="store_true", help="跳过 vLLM（Windows 无 vLLM 时用）")
    args = parser.parse_args()

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print("=" * 70)
    print(f"  Throughput Benchmark | GPU: {gpu_name}")
    print(f"  {N_PROMPTS} prompts x max {MAX_NEW_TOKENS} new tokens")
    print("=" * 70)

    tf = bench_transformers(PROMPTS)

    if args.skip_vllm:
        print("\n[跳过 vLLM] 使用课程参考数据（RTX 4060 8GB / WSL2）")
        vl = {"vllm": {
            "time": 1.15, "gen_tokens": 3492,
            "qps": 43.57, "tps": 3043.0,
            "_note": "课程参考数据，本机需在 WSL2 中运行完整 benchmark",
        }}
    else:
        try:
            vl = bench_vllm(PROMPTS)
        except Exception as e:
            print(f"\nvLLM 运行失败: {e}")
            print("提示：vLLM 需 Linux/WSL2，且 GPU 计算能力 >= 7.0（Volta 及以上）")
            print("GTX 1060 (sm_61) 不满足官方要求，请使用 --skip-vllm 或换 WSL2 + 更高算力 GPU")
            sys.exit(1)

    results = {**tf, **vl}
    print_summary(results)

    out_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
    os.makedirs(out_dir, exist_ok=True)
    meta = {
        "gpu": gpu_name,
        "model_path": MODEL_PATH,
        "n_prompts": N_PROMPTS,
        "max_new_tokens": MAX_NEW_TOKENS,
        "batch_size": BATCH_SIZE,
        "results": results,
    }
    json_path = os.path.join(out_dir, "throughput_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"\nJSON 结果：{json_path}")

    plot_results(results, os.path.join(out_dir, "throughput_comparison.png"), gpu_name)


if __name__ == "__main__":
    main()
