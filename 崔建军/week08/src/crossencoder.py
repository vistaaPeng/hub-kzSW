"""
交互型文本匹配：CrossEncoder

教学重点：
  1. CrossEncoder架构 — 两句拼接后整体送入BERT，直接输出匹配概率
  2. 相比BiEncoder，CrossEncoder让两句在每一层都交互，表达能力更强
  3. 但无法预计算向量，每对句子都要完整过BERT，不适合大规模检索
  4. 典型用途：Reranker（对召回的Top-K候选精排）
  5. 输出是分类概率，直接取argmax即为预测标签，无需阈值搜索

使用方式：
  from crossencoder import CrossEncoder, build_crossencoder

依赖：
  pip install torch transformers
"""

import torch
import torch.nn as nn
import transformers
from transformers import BertConfig, BertModel


# ── CrossEncoder ──────────────────────────────────────────────────────────

class CrossEncoder(nn.Module):
    """
    交互型文本匹配：Cross-Encoder

    结构：
      BertModel([CLS] s1 [SEP] s2 [SEP]) → CLS 向量 → Dropout → Linear(H, 2) → logits

    对比 BiEncoder：
      优点：两句在每一层都交互，表达能力更强，精度更高
      缺点：无法预计算向量，每对句子都要完整过 BERT，不适合大规模检索
      典型用途：Reranker（对召回的 Top-K 候选精排）

    教学对比：
      - 相比BiEncoder，CrossEncoder精度更高但速度慢
      - 相比传统方法，CrossEncoder能学习深层语义交互
      - 输出是分类概率，评估更简单（无需阈值搜索）

    参数：
      bert_path         : 预训练权重路径
      dropout           : 分类头 Dropout 比例
      num_hidden_layers : 同 BiEncoder，限层数加速
    """

    def __init__(self, bert_path, dropout=0.1, num_hidden_layers=None):
        super().__init__()

        config = BertConfig.from_pretrained(bert_path)
        if num_hidden_layers is not None:
            config.num_hidden_layers = num_hidden_layers

        _prev = transformers.logging.get_verbosity()
        transformers.logging.set_verbosity_error()
        self.bert = BertModel.from_pretrained(bert_path, config=config)
        transformers.logging.set_verbosity(_prev)

        hidden_size  = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 2)

    def forward(self, input_ids, attention_mask, token_type_ids):
        """返回 logits [B, 2]，未经 softmax（CrossEntropyLoss 内部处理）"""
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        cls_vec = out.last_hidden_state[:, 0, :]  # [B, H]
        cls_vec = self.dropout(cls_vec)
        return self.classifier(cls_vec)            # [B, 2]


# ── 工厂函数 ──────────────────────────────────────────────────────────────

def build_crossencoder(bert_path, dropout=0.1, num_hidden_layers=None):
    """构建 CrossEncoder 并打印参数量"""
    model = CrossEncoder(bert_path, dropout=dropout,
                         num_hidden_layers=num_hidden_layers)
    _print_param_info(model, f"CrossEncoder (layers={num_hidden_layers or 12})")
    return model


def _print_param_info(model, name):
    """打印模型参数信息"""
    total = sum(p.numel() for p in model.parameters()) / 1e6
    bert  = sum(p.numel() for p in model.bert.parameters()) / 1e6
    print(f"模型: {name}")
    print(f"参数量: {total:.1f}M  (BERT 骨干: {bert:.1f}M)")


# ── 训练脚本示例（独立运行）──────────────────────────────────────────────

if __name__ == "__main__":
    """
    CrossEncoder训练示例

    教学重点：
      1. CrossEntropyLoss训练
      2. 直接输出分类概率，无需阈值搜索
      3. 模型保存与加载
    """
    import argparse
    from pathlib import Path
    from torch.utils.data import DataLoader
    from transformers import BertTokenizer
    from dataset import CrossEncoderDataset, build_crossencoder_loaders

    ROOT = Path(__file__).parent.parent
    DATA_DIR = ROOT / "data" / "lcqmc"
    BERT_PATH = ROOT.parent.parent.parent.parent / "pretrain_models" / "bert-base-chinese"
    OUTPUT_DIR = ROOT / "outputs"
    CKPT_DIR = OUTPUT_DIR / "checkpoints"

    def parse_args():
        parser = argparse.ArgumentParser(description="CrossEncoder训练")
        parser.add_argument("--num_hidden_layers", default=4, type=int)
        parser.add_argument("--max_length", default=128, type=int)
        parser.add_argument("--batch_size", default=32, type=int)
        parser.add_argument("--epochs", default=1, type=int)
        parser.add_argument("--lr", default=2e-5, type=float)
        parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
        return parser.parse_args()

    def train(args):
        """使用CrossEntropyLoss训练"""
        tokenizer = BertTokenizer.from_pretrained(str(BERT_PATH))
        model = build_crossencoder(
            str(BERT_PATH),
            num_hidden_layers=args.num_hidden_layers
        ).to(args.device)

        train_loader, val_loader, _ = build_crossencoder_loaders(
            str(DATA_DIR), tokenizer, args.max_length, args.batch_size
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        criterion = nn.CrossEntropyLoss()

        CKPT_DIR.mkdir(parents=True, exist_ok=True)

        for epoch in range(args.epochs):
            model.train()
            total_loss = 0

            for batch in train_loader:
                input_ids = batch["input_ids"].to(args.device)
                attention_mask = batch["attention_mask"].to(args.device)
                token_type_ids = batch["token_type_ids"].to(args.device)
                labels = batch["label"].to(args.device)

                logits = model(input_ids, attention_mask, token_type_ids)
                loss = criterion(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{args.epochs} - Loss: {avg_loss:.4f}")

        # 保存模型
        save_path = CKPT_DIR / "crossencoder_best.pt"
        torch.save({
            "state_dict": model.state_dict(),
            "args": {
                "num_hidden_layers": args.num_hidden_layers,
                "max_length": args.max_length,
            }
        }, save_path)
        print(f"模型已保存 → {save_path}")

    def main():
        args = parse_args()
        print(f"设备: {args.device}")
        train(args)

    main()