"""
表示型文本匹配：BiEncoder（双塔模型）

教学重点：
  1. BiEncoder架构 — 共享BERT骨干，对两句分别编码，计算余弦相似度
  2. 对应Sentence-BERT论文中的Siamese架构
  3. L2归一化 — encode()输出归一化向量后，余弦相似度等价于点积（更高效）
  4. num_hidden_layers — 限制BERT层数加速训练（4层约为全量的1/3时间）
  5. 支持两种Loss：
     - CosineEmbeddingLoss：直接用相似度与标签计算损失
     - TripletLoss：拉近(anchor, positive)，推远(anchor, negative)（对比学习）

使用方式：
  from biencoder import BiEncoder, build_biencoder

依赖：
  pip install torch transformers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import BertConfig, BertModel


# ── BiEncoder ─────────────────────────────────────────────────────────────

class BiEncoder(nn.Module):
    """
    表示型文本匹配：Siamese Bi-Encoder

    结构：
      shared BertModel → 池化 → Dropout → L2 归一化 → 句向量

    匹配方式：
      sim = cosine_similarity(encode(s1), encode(s2))
      sim ∈ [-1, 1]，越接近 1 越相似

    支持两种 Loss：
      CosineEmbeddingLoss — 直接用相似度与标签计算损失
      TripletLoss         — 拉近 (anchor, positive)，推远 (anchor, negative)

    教学对比：
      - 相比传统方法（编辑距离、TF-IDF），BiEncoder能学习深层语义表示
      - 相比CrossEncoder，BiEncoder可以预计算句向量，适合大规模检索
      - 但BiEncoder的两句编码过程独立，缺乏交互，精度可能低于CrossEncoder

    参数：
      bert_path         : 预训练权重路径（本地目录或 HuggingFace 模型名）
      pool              : 向量提取策略，'cls' / 'mean' / 'max'
                          mean 在句子相似度任务上通常优于 cls（Sentence-BERT 结论）
      dropout           : Dropout 比例
      num_hidden_layers : BERT Transformer 层数；None = 全量 12 层，
                          建议课堂快速验证用 4 层，留 12 层给学生自行实验
    """

    def __init__(self, bert_path, pool="mean", dropout=0.1, num_hidden_layers=None):
        super().__init__()
        assert pool in ("cls", "mean", "max"), f"pool 须为 cls/mean/max，收到: {pool}"

        config = BertConfig.from_pretrained(bert_path)
        if num_hidden_layers is not None:
            config.num_hidden_layers = num_hidden_layers

        _prev = transformers.logging.get_verbosity()
        transformers.logging.set_verbosity_error()
        self.bert = BertModel.from_pretrained(bert_path, config=config)
        transformers.logging.set_verbosity(_prev)

        self.pool    = pool
        self.dropout = nn.Dropout(dropout)

    def encode(self, input_ids, attention_mask, token_type_ids):
        """
        单句编码，返回 L2 归一化后的句向量 [B, H]

        L2 归一化后：cosine_sim(u, v) == dot(u, v)
        可用矩阵乘法批量计算所有两两相似度，适合向量检索场景（如 RAG）
        """
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        vec = self._pool(out.last_hidden_state, attention_mask)  # [B, H]
        vec = self.dropout(vec)
        return F.normalize(vec, p=2, dim=-1)

    def forward(self, batch_a, batch_b):
        """返回 (emb_a, emb_b)，各形状 [B, H]，可直接计算余弦相似度"""
        emb_a = self.encode(**batch_a)
        emb_b = self.encode(**batch_b)
        return emb_a, emb_b

    def _pool(self, last_hidden, attention_mask):
        """
        池化策略：
          - cls：使用[CLS]位置的向量（BERT原始做法）
          - mean：对所有token向量加权平均（Sentence-BERT推荐）
          - max：对所有token向量取最大值
        """
        if self.pool == "cls":
            return last_hidden[:, 0, :]

        mask = attention_mask.unsqueeze(-1).float()  # [B, L, 1]

        if self.pool == "mean":
            sum_h = (last_hidden * mask).sum(dim=1)
            count = mask.sum(dim=1).clamp(min=1e-9)
            return sum_h / count

        if self.pool == "max":
            masked = last_hidden + (1 - mask) * (-1e9)
            return masked.max(dim=1).values


# ── 工厂函数 ──────────────────────────────────────────────────────────────

def build_biencoder(bert_path, pool="mean", dropout=0.1, num_hidden_layers=None):
    """构建 BiEncoder 并打印参数量"""
    model = BiEncoder(bert_path, pool=pool, dropout=dropout,
                      num_hidden_layers=num_hidden_layers)
    _print_param_info(model, f"BiEncoder (pool={pool}, layers={num_hidden_layers or 12})")
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
    BiEncoder训练示例

    教学重点：
      1. CosineEmbeddingLoss vs TripletLoss对比
      2. 验证集阈值搜索
      3. 模型保存与加载
    """
    import argparse
    from pathlib import Path
    from torch.utils.data import DataLoader
    from transformers import BertTokenizer
    from dataset import PairDataset, TripletDataset, build_pair_loaders, build_triplet_loader

    ROOT = Path(__file__).parent.parent
    DATA_DIR = ROOT / "data" / "lcqmc"
    BERT_PATH = "bert-base-chinese"
    OUTPUT_DIR = ROOT / "outputs"
    CKPT_DIR = OUTPUT_DIR / "checkpoints"

    def parse_args():
        parser = argparse.ArgumentParser(description="BiEncoder训练")
        parser.add_argument("--loss_type", default="cosine", choices=["cosine", "triplet"])
        parser.add_argument("--pool", default="mean", choices=["cls", "mean", "max"])
        parser.add_argument("--num_hidden_layers", default=4, type=int)
        parser.add_argument("--max_length", default=64, type=int)
        parser.add_argument("--batch_size", default=32, type=int)
        parser.add_argument("--epochs", default=1, type=int)
        parser.add_argument("--lr", default=2e-5, type=float)
        parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
        return parser.parse_args()

    def train_cosine(args):
        """使用CosineEmbeddingLoss训练"""
        tokenizer = BertTokenizer.from_pretrained(str(BERT_PATH))
        model = build_biencoder(
            str(BERT_PATH),
            pool=args.pool,
            num_hidden_layers=args.num_hidden_layers
        ).to(args.device)

        train_loader, val_loader, _ = build_pair_loaders(
            str(DATA_DIR), tokenizer, args.max_length, args.batch_size
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        criterion = nn.CosineEmbeddingLoss(margin=0.5)

        CKPT_DIR.mkdir(parents=True, exist_ok=True)

        for epoch in range(args.epochs):
            model.train()
            total_loss = 0

            for batch in train_loader:
                batch_a = {
                    "input_ids": batch["input_ids_a"].to(args.device),
                    "attention_mask": batch["attention_mask_a"].to(args.device),
                    "token_type_ids": batch["token_type_ids_a"].to(args.device),
                }
                batch_b = {
                    "input_ids": batch["input_ids_b"].to(args.device),
                    "attention_mask": batch["attention_mask_b"].to(args.device),
                    "token_type_ids": batch["token_type_ids_b"].to(args.device),
                }
                labels = batch["label"].to(args.device)

                # CosineEmbeddingLoss需要标签为-1或+1
                targets = 2 * labels.float() - 1  # 0→-1, 1→+1

                emb_a, emb_b = model(batch_a, batch_b)
                loss = criterion(emb_a, emb_b, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{args.epochs} - Loss: {avg_loss:.4f}")

        # 保存模型
        save_path = CKPT_DIR / "biencoder_cosine_best.pt"
        torch.save({
            "state_dict": model.state_dict(),
            "args": {
                "pool": args.pool,
                "num_hidden_layers": args.num_hidden_layers,
                "max_length": args.max_length,
            }
        }, save_path)
        print(f"模型已保存 → {save_path}")

    def train_triplet(args):
        """使用TripletLoss训练（对比学习）"""
        tokenizer = BertTokenizer.from_pretrained(str(BERT_PATH))
        model = build_biencoder(
            str(BERT_PATH),
            pool=args.pool,
            num_hidden_layers=args.num_hidden_layers
        ).to(args.device)

        train_loader, val_loader = build_triplet_loader(
            str(DATA_DIR), tokenizer, args.max_length, args.batch_size
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        criterion = nn.TripletMarginLoss(margin=1.0)

        CKPT_DIR.mkdir(parents=True, exist_ok=True)

        for epoch in range(args.epochs):
            model.train()
            total_loss = 0

            for batch in train_loader:
                batch_a = {
                    "input_ids": batch["input_ids_a"].to(args.device),
                    "attention_mask": batch["attention_mask_a"].to(args.device),
                    "token_type_ids": batch["token_type_ids_a"].to(args.device),
                }
                batch_p = {
                    "input_ids": batch["input_ids_p"].to(args.device),
                    "attention_mask": batch["attention_mask_p"].to(args.device),
                    "token_type_ids": batch["token_type_ids_p"].to(args.device),
                }
                batch_n = {
                    "input_ids": batch["input_ids_n"].to(args.device),
                    "attention_mask": batch["attention_mask_n"].to(args.device),
                    "token_type_ids": batch["token_type_ids_n"].to(args.device),
                }

                emb_a = model.encode(**batch_a)
                emb_p = model.encode(**batch_p)
                emb_n = model.encode(**batch_n)

                loss = criterion(emb_a, emb_p, emb_n)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{args.epochs} - Loss: {avg_loss:.4f}")

        # 保存模型
        save_path = CKPT_DIR / "biencoder_triplet_best.pt"
        torch.save({
            "state_dict": model.state_dict(),
            "args": {
                "pool": args.pool,
                "num_hidden_layers": args.num_hidden_layers,
                "max_length": args.max_length,
            }
        }, save_path)
        print(f"模型已保存 → {save_path}")

    def main():
        args = parse_args()
        print(f"设备: {args.device}")
        print(f"Loss类型: {args.loss_type}")

        if args.loss_type == "cosine":
            train_cosine(args)
        else:
            train_triplet(args)

    main()