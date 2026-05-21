"""
BERT 测试脚本：评估 + 交互推理
"""
import os
import sys
import json
import logging

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)

from config import *
from preprocess import BertDataset
from train import BertClassifier


def setup_logging():
    os.makedirs(LOG_DIR, exist_ok=True)
    log_file = os.path.join(LOG_DIR, "bert_test_result.log")
    logger = logging.getLogger("bert_test")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)
    return logger


def load_model(device):
    ckpt_path = os.path.join(MODEL_DIR, "best_model.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"未找到模型: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = BertClassifier(ckpt["model_name"], ckpt["num_classes"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt["idx2label"], ckpt


@torch.no_grad()
def evaluate(model, test_loader, device, idx2label, logger):
    model.eval()
    all_preds, all_labels = [], []
    for x, mask, y in test_loader:
        x, mask, y = x.to(device), mask.to(device), y.to(device)
        logits = model(x, mask)
        preds = logits.argmax(dim=-1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(y.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    label_names = [idx2label[str(i)] for i in range(len(idx2label))]

    report = classification_report(all_labels, all_preds, target_names=label_names, digits=4)

    return {
        "accuracy": acc,
        "precision_macro": precision_score(all_labels, all_preds, average="macro"),
        "recall_macro": recall_score(all_labels, all_preds, average="macro"),
        "f1_macro": f1_score(all_labels, all_preds, average="macro"),
        "f1_weighted": f1_score(all_labels, all_preds, average="weighted"),
        "report": report,
        "preds": all_preds, "labels": all_labels,
    }


def predict_single(text, model, tokenizer, device, idx2label):
    encoded = tokenizer(
        text, max_length=MAX_LEN, padding="max_length",
        truncation=True, return_tensors="pt",
    )
    x = encoded["input_ids"].to(device)
    mask = encoded["attention_mask"].to(device)
    with torch.no_grad():
        logits = model(x, mask)
        probs = F.softmax(logits, dim=-1).squeeze()
        pred_idx = logits.argmax(dim=-1).item()
    return idx2label[str(pred_idx)], probs


def main():
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("BERT 新闻分类 — 测试评估")
    logger.info("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"设备: {device}")

    logger.info("\n加载模型...")
    model, idx2label, ckpt = load_model(device)
    logger.info(f"模型: {ckpt['model_name']}, epoch {ckpt['epoch']}, Val F1: {ckpt['val_f1']:.4f}")

    logger.info("\n加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    logger.info("加载测试数据...")
    test_ds = BertDataset(TEST_X, TEST_MASK, TEST_Y)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    logger.info(f"测试样本: {len(test_ds)}")

    results = evaluate(model, test_loader, device, idx2label, logger)

    logger.info("\n" + "=" * 60)
    logger.info("测试结果")
    logger.info("=" * 60)
    logger.info(f"Accuracy:        {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    logger.info(f"Precision (Macro): {results['precision_macro']:.4f}")
    logger.info(f"Recall (Macro):    {results['recall_macro']:.4f}")
    logger.info(f"F1 (Macro):        {results['f1_macro']:.4f}")
    logger.info(f"F1 (Weighted):     {results['f1_weighted']:.4f}")
    logger.info("\n--- 分类报告 ---")
    logger.info(results["report"])

    logger.info("--- 各类别准确率 ---")
    cm = confusion_matrix(results["labels"], results["preds"])
    for i in range(len(idx2label)):
        name = idx2label[str(i)]
        correct = cm[i, i]
        total = cm[i].sum()
        logger.info(f"  {name:4s}: {correct:4d}/{total:4d} = {correct/total*100:.2f}%")

    # Demo
    logger.info("\n--- 单条预测演示 ---")
    demos = [
        "北京时间今天凌晨，NBA总决赛第六场在洛杉矶斯台普斯中心进行，湖人队主场击败凯尔特人，将总比分扳成3比3平。",
        "近日，中国科学家在量子计算领域取得重大突破，成功实现了量子比特的长时间稳定操作。",
        "央行今日宣布下调存款准备金率0.5个百分点，旨在释放流动性，支持实体经济发展。",
    ]
    for text in demos:
        label, probs = predict_single(text, model, tokenizer, device, idx2label)
        top3 = probs.topk(3).indices.tolist()
        top3_str = ", ".join(f"{idx2label[str(i)]}({probs[i].item():.3f})" for i in top3)
        logger.info(f"  {text[:50]}...")
        logger.info(f"    Top3: {top3_str}")

    logger.info("\n评估完成！")

    # 交互推理
    print("\n" + "=" * 60)
    print("BERT 交互式推理模式")
    print(f"可选类别: {', '.join(idx2label[str(i)] for i in range(len(idx2label)))}")
    print("输入 'quit' / 'exit' 退出")
    print("=" * 60)

    while True:
        try:
            text = input("\n请输入新闻文本: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出。")
            break
        if text.lower() in ("quit", "exit", "q"):
            print("退出。")
            break
        if not text:
            continue
        if len(text) < 10:
            print("文本太短。")
            continue

        label, probs = predict_single(text, model, tokenizer, device, idx2label)
        sorted_idx = probs.argsort(descending=True).tolist()
        print(f"\n  预测结果: {label}")
        print("  各类别概率:")
        for i in sorted_idx:
            name = idx2label[str(i)]
            p = probs[i].item()
            bar = "#" * int(p * 40)
            print(f"    {name:4s} | {bar}{' ' * max(0, 40 - len(bar))} | {p:.3f}")


if __name__ == "__main__":
    main()
