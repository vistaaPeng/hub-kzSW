import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report

from dataset import NERDataset, tokenizer
from model import build_model

# =========================
# 配置路径
# =========================
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
PRETRAINED_DIR = ROOT.parent.parent / "pretrained_models" / "bert-base-chinese"
TEST_FILE = DATA_DIR / "test.json"
LABEL_FILE = DATA_DIR / "label_names.json"
SAVE_FILE = ROOT / "outputs" / "prediction_errors.txt"

# 确保输出目录存在
SAVE_FILE.parent.mkdir(parents=True, exist_ok=True)

# =========================
# 标签
# =========================
with open(LABEL_FILE, "r", encoding="utf-8") as f:
    LABELS = json.load(f)

LABEL2ID = {label: idx for idx, label in enumerate(LABELS)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}
NUM_LABELS = len(LABELS)

# =========================
# 数据加载
# =========================
test_dataset = NERDataset(TEST_FILE, LABEL2ID, tokenizer)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# =========================
# 模型加载
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
ckpt_path = ROOT.parent.parent / "best_model.pt"

# 自动检测是否使用 CRF
checkpoint = torch.load(ckpt_path, map_location=device)
use_crf = any("crf." in key for key in checkpoint.keys())
print(f"检测到模型使用 CRF: {use_crf}")

# 构建对应架构的模型
model = build_model(
    bert_model_path=str(PRETRAINED_DIR),
    num_labels=NUM_LABELS,
    use_crf=use_crf
)
model.to(device)
model.load_state_dict(checkpoint)
print(f"成功加载模型: {ckpt_path}")

# =========================
# 测试函数
# =========================
def evaluate_with_examples(model, dataloader, device, save_file="prediction_errors.txt"):
    model.eval()

    all_true_labels = []
    all_pred_labels = []

    error_count = 0
    total_samples = 0

    with open(save_file, "w", encoding="utf-8") as writer:
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"]

                outputs = model(input_ids, attention_mask, labels=None)

                if model.use_crf:
                    # CRF 模式：preds 是 list of lists
                    batch_preds = outputs["preds"]
                else:
                    # Softmax 模式：preds 是 tensor
                    batch_preds = outputs["preds"].cpu().tolist()

                for i in range(len(labels)):
                    total_samples += 1
                    label_ids = labels[i].tolist()

                    if model.use_crf:
                        # CRF 解码返回的是已解码的标签序列
                        pred_ids = batch_preds[i]
                        # 可能需要对齐（因为 CRF 的 pred 可能已经去掉了 padding）
                        # 这里需要根据你的具体情况调整
                    else:
                        pred_ids = batch_preds[i]

                    token_ids = input_ids[i].cpu().tolist()
                    tokens = tokenizer.convert_ids_to_tokens(token_ids)

                    cur_true = []
                    cur_pred = []
                    rows = []
                    has_error = False


                    # 处理 CRF 和非 CRF 模式的差异
                    if model.use_crf:
                        # CRF 返回的 preds 只包含有效位置的预测
                        pred_idx = 0
                        for token_idx, (token, l) in enumerate(zip(tokens, label_ids)):
                            if l == -100:
                                # 跳过特殊 token，但 [CLS] 在 CRF 中也有预测
                                if token_idx == 0 and pred_idx < len(pred_ids):
                                    pred_idx += 1  # 跳过 [CLS] 的预测
                                continue
                            true_tag = ID2LABEL[l]
                            if pred_idx < len(pred_ids):
                                pred_tag = ID2LABEL[pred_ids[pred_idx]]
                                pred_idx += 1
                            else:
                                pred_tag = "O"  # fallback

                            cur_true.append(true_tag)
                            cur_pred.append(pred_tag)
                            if true_tag != pred_tag:
                                has_error = True
                            rows.append((token, true_tag, pred_tag))

                            # print(f"Sample tokens: {tokens[:10]}")
                            # print(f"True labels: {label_ids[:10]}")
                            # print(f"Pred labels: {pred_ids[:10] if isinstance(pred_ids, list) else pred_ids[:10]}")
                            # print(f"CRF pred length: {len(pred_ids)}, valid labels: {sum(1 for l in label_ids if l != -100)}")
                    else:
                        # 非 CRF 模式
                        for token, l, p in zip(tokens, label_ids, pred_ids):
                            if l == -100:
                                continue
                            true_tag = ID2LABEL[l]
                            pred_tag = ID2LABEL[p]
                            cur_true.append(true_tag)
                            cur_pred.append(pred_tag)
                            if true_tag != pred_tag:
                                has_error = True
                            rows.append((token, true_tag, pred_tag))

                    all_true_labels.append(cur_true)
                    all_pred_labels.append(cur_pred)

                    if not has_error:
                        continue

                    error_count += 1
                    text = "".join([r[0] for r in rows if r[0] not in ["[CLS]", "[SEP]", "[PAD]", "#", "##"]])
                    writer.write("="*80 + "\n")
                    writer.write(f"ERROR SAMPLE #{error_count} (Dataset Index={total_samples})\n")
                    writer.write("="*80 + "\n\n")
                    writer.write(f"文本:\n{text}\n\n")
                    writer.write("Token".ljust(15) + "True".ljust(12) + "Pred\n")
                    writer.write("-"*50 + "\n")
                    for token, t, p in rows:
                        marker = " <<<" if t != p else ""
                        writer.write(f"{token:<15}{t:<12}{p}{marker}\n")
                    writer.write("\n\n")

    precision = precision_score(all_true_labels, all_pred_labels)
    recall = recall_score(all_true_labels, all_pred_labels)
    f1 = f1_score(all_true_labels, all_pred_labels)
    report = classification_report(all_true_labels, all_pred_labels, digits=4)

    print(f"\n错误样本数: {error_count}/{total_samples}")
    print(f"错误样本已保存到: {save_file}")

    return precision, recall, f1, report

# =========================
# 执行评估
# =========================
precision, recall, f1, report = evaluate_with_examples(
    model, test_loader, device, save_file=str(SAVE_FILE)
)
print(f"Test Results | P: {precision:.4f} | R: {recall:.4f} | F1: {f1:.4f}")
print(report)
print(f"详细预测结果已保存到 {SAVE_FILE}")
