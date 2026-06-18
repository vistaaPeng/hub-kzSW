#coding:utf8
"""
使用训练好的 BERT NER 模型进行预测
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from pathlib import Path
import sys

# 获取当前文件所在目录
current_dir = Path(__file__).resolve().parent

# 如果 predict.py 在项目根目录，需要添加 src 到路径
if current_dir.name != "src":
    src_dir = current_dir / "src"
    sys.path.insert(0, str(src_dir))
    ROOT = current_dir
else:
    ROOT = current_dir.parent

from dataset import build_label_schema
from model import build_model

def predict_text(text, model, tokenizer, id2label, device):
    """对单个文本进行 NER 预测"""
    model.eval()
    
    # 对文本进行分词
    chars = list(text)
    
    # 编码
    encoding = tokenizer(
        chars,
        is_split_into_words=True,
        max_length=128,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    token_type_ids = encoding["token_type_ids"].to(device)
    
    # 预测
    with torch.no_grad():
        logits, _ = model(input_ids, attention_mask, token_type_ids)
    
    predictions = logits.argmax(dim=-1)[0].cpu().tolist()
    
    # 对齐到原始字符
    word_ids = encoding.word_ids(batch_index=0)
    char_labels = []
    prev_word_id = None
    
    for wid, pred in zip(word_ids, predictions):
        if wid is None or wid == prev_word_id:
            continue
        if wid < len(chars):
            char_labels.append((chars[wid], id2label[pred]))
        prev_word_id = wid
    
    return char_labels


def main():
    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 标签体系
    labels, label2id, id2label = build_label_schema()
    num_labels = len(labels)
    
    # 加载训练好的模型
    checkpoint_path = ROOT / "outputs" / "checkpoints" / "best_linear.pt"
    
    if not checkpoint_path.exists():
        print(f"❌ 模型文件不存在: {checkpoint_path}")
        print("请先运行 train.py 完成训练！")
        return
    
    # 加载 checkpoint（PyTorch 2.6+ 需要设置 weights_only=False）
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # 构建模型
    bert_path = ROOT / "bert-base-chinese"
    model = build_model(
        use_crf=False,  # 如果训练的是 CRF 模型，改为 True
        bert_path=str(bert_path),
        num_labels=num_labels,
        dropout=0.0  # 推理时不用 dropout
    )
    
    # 加载权重
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)
    
    # 加载 tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(bert_path))
    
    print("✅ 模型加载成功！")
    print(f"验证集 F1: {checkpoint.get('val_entity_f1', 'N/A')}")
    print("\n" + "=" * 50)
    print("开始预测（输入 quit 退出）")
    print("=" * 50)
    
    while True:
        text = input("\n请输入文本: ").strip()
        if text.lower() == "quit":
            break
        
        if not text:
            continue
        
        # 预测
        results = predict_text(text, model, tokenizer, id2label, device)
        
        # 打印结果
        print("\n识别结果:")
        for char, label in results:
            if label != "O":
                print(f"  {char} → {label}")
        
        if not any(label != "O" for _, label in results):
            print("  （未识别到命名实体）")


if __name__ == "__main__":
    main()
