"""验证 DIY BERT + 分类头与原版 pipeline 一致 — 使用微调权重 + 真实测试数据"""
import os
os.environ["HF_HOME"] = "M:/huggingface_cache"

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from MyBert import MyBert
from load_weights import load_hf_bert, load_weights_from_checkpoint

PROCESSED_DIR = "../bert_news_classify/processed"
CHECKPOINT_PATH = "../bert_news_classify/checkpoints/best_model.pt"


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    device = get_device()
    print("=" * 60)
    print("DIY BERT Classification Pipeline Verification")
    print(f"Device: {device}")
    print("=" * 60)

    print("\nLoading models...")
    hf_bert, config = load_hf_bert("bert-base-chinese", device=device)

    my_bert = MyBert(config)
    my_bert.eval().to(device)

    # 从 checkpoint 加载微调后的 BERT 权重 + classifier/dropout
    map_loc = device if device.type == "cuda" else "cpu"
    cls_w, cls_b = load_weights_from_checkpoint(my_bert, CHECKPOINT_PATH, map_location=map_loc)
    print("Fine-tuned weights loaded from checkpoint.")

    classifier = nn.Linear(config.hidden_size, 10).to(device)
    classifier.weight.data.copy_(cls_w.to(device))
    classifier.bias.data.copy_(cls_b.to(device))
    classifier.eval()

    print("Classifier loaded.")

    # 同时为 HF BERT 也加载微调后的 BERT 权重
    ckpt = torch.load(CHECKPOINT_PATH, map_location=map_loc, weights_only=False)
    hf_bert.load_state_dict(
        {k[5:]: v for k, v in ckpt["model_state_dict"].items() if k.startswith("bert.")},
        strict=False,
    )
    print("HF BERT fine-tuned weights loaded.")

    # Load real test data
    input_ids = torch.load(os.path.join(PROCESSED_DIR, "test_input_ids.pt"), weights_only=True)
    attn_mask = torch.load(os.path.join(PROCESSED_DIR, "test_attention_mask.pt"), weights_only=True)
    labels = torch.load(os.path.join(PROCESSED_DIR, "test_labels.pt"), weights_only=True)
    total = len(labels)
    n_samples = min(total, 200)
    idx = torch.randperm(total)[:n_samples]
    input_ids = input_ids[idx].to(device)
    attn_mask = attn_mask[idx].to(device)
    labels = labels[idx]
    print(f"Test data: {n_samples} randomly sampled from {total} total")

    # Inference (with dropout before classifier — matching BertClassifier.forward)
    with torch.no_grad():
        hf_out = hf_bert(input_ids, attention_mask=attn_mask)
        hf_logits = classifier(hf_out.pooler_output)

        diy_hidden, diy_pooled = my_bert(input_ids, attention_mask=attn_mask)

        diy_logits = classifier(diy_pooled)

    # Logits consistency
    logits_ok = torch.allclose(hf_logits.cpu(), diy_logits.cpu(), atol=1e-4)
    logits_diff = (hf_logits - diy_logits).abs().max().item()

    # Predictions
    hf_preds = hf_logits.argmax(dim=1).cpu()
    diy_preds = diy_logits.argmax(dim=1).cpu()
    pred_agree = (hf_preds == diy_preds).sum().item()
    pred_agree_rate = pred_agree / n_samples * 100

    # Accuracy (should be identical)
    hf_acc = accuracy_score(labels, hf_preds)
    diy_acc = accuracy_score(labels, diy_preds)

    print(f"\n  Samples:           {n_samples}")
    print(f"  Logits match:      {'PASS' if logits_ok else 'FAIL'} (max diff={logits_diff:.2e})")
    print(f"  Pred agreement:    {pred_agree}/{n_samples} ({pred_agree_rate:.1f}%)")
    print(f"  HF accuracy:       {hf_acc:.4f}")
    print(f"  DIY accuracy:      {diy_acc:.4f}")

    print("\n" + "=" * 60)
    all_pass = logits_ok and pred_agree == n_samples
    print(f"Result: {'ALL PASS' if all_pass else 'SOME FAILURES'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
