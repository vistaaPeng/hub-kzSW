"""
跨数据集方法对比脚本（简化版）

对比不同文本匹配方法在多个数据集上的效果：
  - 数据集：bq_corpus, lcqmc
  - 方法：BiEncoder (CosineEmbeddingLoss), BiEncoder (TripletLoss), CrossEncoder

使用方式：
  python compare_datasets.py --datasets bq_corpus lcqmc
  python compare_datasets.py --datasets bq_corpus lcqmc --epochs 3 --num_hidden_layers 4
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import subprocess
import time
from pathlib import Path

# ── 默认路径 ──────────────────────────────────────────────────────────────
try:
    SCRIPT_DIR = Path(__file__).parent.resolve()
    ROOT       = SCRIPT_DIR.parent
except NameError:
    ROOT       = Path("D:/workspace/AA/w8/文本匹配项目").resolve()
    SCRIPT_DIR = ROOT / "src"

DATA_DIR   = ROOT / "data"
BERT_PATH  = ROOT / "bert-base-chinese" / "bert-base-chinese"
OUTPUT_DIR = ROOT / "outputs"
CKPT_DIR   = OUTPUT_DIR / "checkpoints"
LOG_DIR    = OUTPUT_DIR / "logs"

# ── 支持的数据集和方法 ────────────────────────────────────────────────────
DATASETS = {
    "bq_corpus": {
        "name": "BQ Corpus",
        "desc": "银行领域问答匹配",
    },
    "lcqmc": {
        "name": "LCQMC",
        "desc": "大规模中文问句匹配",
    },
}

METHODS = [
    {"key": "biencoder_cosine", "label": "BiEncoder (Cosine)", "loss": "cosine"},
    {"key": "biencoder_triplet", "label": "BiEncoder (Triplet)", "loss": "triplet"},
    {"key": "crossencoder", "label": "CrossEncoder", "loss": None},
]

# ── 训练单个方法在单个数据集上 ────────────────────────────────────────────

def train_method(dataset, method, args):
    """训练指定方法在指定数据集上"""
    data_path = DATA_DIR / dataset
    if not data_path.exists():
        print(f"  [ERROR] 数据集不存在: {data_path}")
        return False
    
    ckpt_name = f"{dataset}_{method['key']}_best.pt"
    ckpt_path = CKPT_DIR / ckpt_name
    
    if args.skip_existing and ckpt_path.exists():
        print(f"  [SKIP] checkpoint 已存在: {ckpt_name}")
        return True
    
    cmd = []
    if method["key"].startswith("biencoder"):
        cmd = [
            "python", "train_biencoder.py",
            "--data_dir", str(data_path),
            "--loss", method["loss"],
            "--num_hidden_layers", str(args.num_hidden_layers),
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--lr", str(args.lr),
            "--margin", str(args.margin),
        ]
    else:  # crossencoder
        cmd = [
            "python", "train_crossencoder.py",
            "--data_dir", str(data_path),
            "--num_hidden_layers", str(args.num_hidden_layers),
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--lr", str(args.lr),
        ]
    
    print(f"  开始训练: {' '.join(cmd)}")
    t0 = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            cwd=ROOT / "src",
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=args.timeout,
        )
        elapsed = time.time() - t0
        
        if result.returncode == 0:
            print(f"  ✓ 训练完成 ({elapsed:.1f}s)")
            # 重命名 checkpoint 为带数据集前缀的名称
            if method["key"].startswith("biencoder"):
                old_ckpt = CKPT_DIR / f"biencoder_{method['loss']}_best.pt"
            else:
                old_ckpt = CKPT_DIR / f"crossencoder_best.pt"
            if old_ckpt.exists():
                old_ckpt.rename(ckpt_path)
                print(f"  ✓ 重命名 checkpoint: {old_ckpt.name} -> {ckpt_name}")
            else:
                print(f"  ⚠️  checkpoint 不存在: {old_ckpt}")
            return True
        else:
            print(f"  ✗ 训练失败 (exit code: {result.returncode})")
            stderr_content = result.stderr[:500] if result.stderr else "None"
            print(f"  stderr: {stderr_content}")
            return False
    except subprocess.TimeoutExpired:
        print(f"  ✗ 训练超时 ({args.timeout}s)")
        return False
    except Exception as e:
        print(f"  ✗ 训练异常: {e}")
        return False

# ── 评估单个方法在单个数据集上 ────────────────────────────────────────────

def evaluate_method(dataset, method, args):
    """评估指定方法在指定数据集上"""
    data_path = DATA_DIR / dataset
    ckpt_name = f"{dataset}_{method['key']}_best.pt"
    ckpt_path = CKPT_DIR / ckpt_name
    
    if not ckpt_path.exists():
        print(f"  [SKIP] checkpoint 不存在: {ckpt_name}")
        return None
    
    cmd = [
        "python", "evaluate.py",
        "--model_type", "biencoder" if method["key"].startswith("biencoder") else "crossencoder",
        "--ckpt", str(ckpt_path),
        "--data_dir", str(data_path),
        "--batch_size", str(args.batch_size),
    ]
    
    print(f"  开始评估: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=ROOT / "src",
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
        )
        
        if result.returncode == 0:
            # 解析评估结果
            output = result.stdout
            metrics = {}
            for line in output.split("\n"):
                if "Accuracy:" in line:
                    parts = line.split(":")
                    if len(parts) > 1:
                        metrics["accuracy"] = float(parts[1].strip())
                if "F1" in line and ":" in line and "threshold" not in line.lower():
                    parts = line.split(":")
                    if len(parts) > 1:
                        try:
                            metrics["f1"] = float(parts[1].strip())
                        except:
                            pass
                if "最优阈值:" in line:
                    parts = line.split(":")
                    if len(parts) > 1:
                        metrics["threshold"] = float(parts[1].strip())
            metrics["output"] = output
            return metrics
        else:
            print(f"  ✗ 评估失败 (exit code: {result.returncode})")
            return None
    except Exception as e:
        print(f"  ✗ 评估异常: {e}")
        return None

# ── 主流程 ────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="跨数据集文本匹配方法对比")
    parser.add_argument("--datasets", nargs="+", default=["bq_corpus", "lcqmc"],
                        choices=["bq_corpus", "lcqmc"],
                        help="要对比的数据集")
    parser.add_argument("--methods", nargs="+", default=[m["key"] for m in METHODS],
                        choices=[m["key"] for m in METHODS],
                        help="要对比的方法")
    parser.add_argument("--num_hidden_layers", default=4, type=int)
    parser.add_argument("--epochs", default=3, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--lr", default=2e-5, type=float)
    parser.add_argument("--margin", default=0.3, type=float)
    parser.add_argument("--skip_existing", action="store_true",
                        help="跳过已存在的 checkpoint")
    parser.add_argument("--timeout", default=3600, type=int,
                        help="单方法训练超时时间（秒）")
    parser.add_argument("--eval_only", action="store_true",
                        help="仅评估，不训练")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"{'='*70}")
    print("跨数据集文本匹配方法对比")
    print(f"{'='*70}")
    print(f"数据集: {', '.join([DATASETS[d]['name'] for d in args.datasets])}")
    print(f"方法: {', '.join([m['label'] for m in METHODS if m['key'] in args.methods])}")
    print(f"参数: {args.num_hidden_layers} 层 BERT, {args.epochs} epochs, batch_size={args.batch_size}")
    print(f"{'='*70}")
    
    # 创建输出目录
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    # 筛选要运行的方法
    selected_methods = [m for m in METHODS if m["key"] in args.methods]
    
    # ── 训练阶段 ──────────────────────────────────────────────────────────
    if not args.eval_only:
        for dataset in args.datasets:
            print(f"\n{'─'*60}")
            print(f"训练 [{DATASETS[dataset]['name']}]")
            print(f"{'─'*60}")
            
            for method in selected_methods:
                print(f"\n  方法: {method['label']}")
                success = train_method(dataset, method, args)
                if not success:
                    print(f"  ⚠️  训练失败，跳过评估")
    
    # ── 评估阶段 ──────────────────────────────────────────────────────────
    results = {}
    for dataset in args.datasets:
        results[dataset] = {}
        print(f"\n{'─'*60}")
        print(f"评估 [{DATASETS[dataset]['name']}]")
        print(f"{'─'*60}")
        
        for method in selected_methods:
            print(f"\n  方法: {method['label']}")
            metrics = evaluate_method(dataset, method, args)
            if metrics:
                results[dataset][method["key"]] = metrics
                print(f"    Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")
                print(f"    F1: {metrics.get('f1', 'N/A'):.4f}")
                if "threshold" in metrics:
                    print(f"    Threshold: {metrics['threshold']:.2f}")
    
    # ── 汇总对比表 ────────────────────────────────────────────────────────
    print(f"\n{'='*75}")
    print(f"跨数据集方法对比汇总")
    print(f"{'='*75}")
    
    header = f"{'方法':<22}"
    for dataset in args.datasets:
        header += f" | {DATASETS[dataset]['name']:^24}"
    print(header)
    
    separator = f"{'-'*22}"
    for _ in args.datasets:
        separator += f" | {'-'*24}"
    print(separator)
    
    for method in selected_methods:
        row = f"{method['label']:<22}"
        for dataset in args.datasets:
            metrics = results[dataset].get(method["key"], {})
            acc = metrics.get("accuracy", 0)
            f1 = metrics.get("f1", 0)
            row += f" |  Acc={acc:.4f} F1={f1:.4f}"
        print(row)
    
    # ── 分析结论 ──────────────────────────────────────────────────────────
    print(f"\n{'─'*75}")
    print("分析结论")
    print(f"{'─'*75}")
    
    for dataset in args.datasets:
        print(f"\n【{DATASETS[dataset]['name']}】")
        ds_results = results[dataset]
        
        if not ds_results:
            print("  无可用评估结果")
            continue
        
        best_acc_method = max(ds_results.keys(), key=lambda k: ds_results[k].get("accuracy", 0))
        best_f1_method = max(ds_results.keys(), key=lambda k: ds_results[k].get("f1", 0))
        
        best_acc = ds_results[best_acc_method].get("accuracy", 0)
        best_f1 = ds_results[best_f1_method].get("f1", 0)
        
        acc_method_name = [m['label'] for m in METHODS if m['key'] == best_acc_method][0]
        f1_method_name = [m['label'] for m in METHODS if m['key'] == best_f1_method][0]
        
        print(f"  最高 Accuracy: {acc_method_name} ({best_acc:.4f})")
        print(f"  最高 F1: {f1_method_name} ({best_f1:.4f})")
    
    if len(args.datasets) >= 2:
        print("\n【跨数据集差异】")
        for method in selected_methods:
            acc1 = results[args.datasets[0]].get(method["key"], {}).get("accuracy", 0)
            acc2 = results[args.datasets[1]].get(method["key"], {}).get("accuracy", 0)
            f1_1 = results[args.datasets[0]].get(method["key"], {}).get("f1", 0)
            f1_2 = results[args.datasets[1]].get(method["key"], {}).get("f1", 0)
            
            acc_diff = abs(acc1 - acc2)
            f1_diff = abs(f1_1 - f1_2)
            print(f"  {method['label']}: Accuracy差异={acc_diff:.4f}, F1差异={f1_diff:.4f}")
    
    # ── 保存日志 ──────────────────────────────────────────────────────────
    log_path = LOG_DIR / "cross_dataset_comparison.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n对比日志已保存 → {log_path}")
    print(f"\n{'='*75}")
    print("对比完成！")
    print(f"{'='*75}")

if __name__ == "__main__":
    main()
