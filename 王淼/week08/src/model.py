"""
文本匹配模型定义

  1. BiEncoder（表示型）— 共享 BERT 骨干，对两句分别编码，计算余弦相似度
     对应 Sentence-BERT 论文中的 Siamese 架构
  2. CrossEncoder（交互型）— 两句拼接后整体送入 BERT，直接输出匹配概率
  3. L2 归一化 — encode() 输出归一化向量后，余弦相似度等价于点积（更高效）
  4. num_hidden_layers — 限制 BERT 层数加速训练（4 层约为全量的 1/3 时间）
     原理：从完整 12 层权重中只加载前 N 层，其余丢弃

使用方式：
  from model import BiEncoder, CrossEncoder, build_biencoder, build_crossencoder

依赖：
  pip install torch transformers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import BertConfig, BertModel

# ── BiEncoder（Siamese）──
class BiEncoder(nn.Module):
    def __init__(self,bert_path,pool="mean",dropout=0.1,num_hidden_layers=None):
        super().__init__()
        assert pool in ("cls", "mean", "max"), f"pool 须为 cls/mean/max，收到: {pool}"
        
        config = BertConfig.from_pretrained(bert_path)
        if num_hidden_layers is not None:
            config.num_hidden_layers = num_hidden_layers

        _prev = transformers.logging.get_verbosity()
        transformers.logging.set_verbosity_error()
        self.bert = BertModel.from_pretrained(bert_path, config=config)
        transformers.logging.set_verbosity(_prev)

        self.dropout = nn.Dropout(dropout)
        self.pool = pool
    
    def encode(self,input_ids,attention_mask,token_type_ids):
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        vec = self._pool(out.last_hidden_state, attention_mask)
        vec = self.dropout(vec)
        return F.normalize(vec, p=2, dim=1)
    
    def forward(self,batch_a,batch_b):
        emb_a = self.encode(**batch_a)
        emb_b = self.encode(**batch_b)
        return emb_a, emb_b
    
    def _pool(self, last_hidden, attention_mask):
        if self.pool == "cls":
            return last_hidden[:,0,:]
        
        mask = attention_mask.unsqueeze(-1).float()

        if self.pool == "mean":
            sum_hidden = (last_hidden * mask).sum(dim=1)
            count_tokens = mask.sum(dim=1).clamp(min=1e-9)
            return sum_hidden / count_tokens
        
        if self.pool == "max":
            masked = last_hidden + (1 - mask)*-1e9
            return masked.max(dim=1).values
        

# ── CrossEncoder ──
class CrossEncoder(nn.Module):
    def __init__(self,bert_path,dropout=0.1,num_hidden_layers=None):
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

    def forward(self,input_ids, attention_mask, token_type_ids):
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        cls_vec = out.last_hidden_state[:,0,:]
        cls_vec = self.dropout(cls_vec)
        return self.classifier(cls_vec)
    
# ── 工厂函数 ──────────────────────────────────────────────────────────────

def build_biencoder(bert_path, pool="mean", dropout=0.1, num_hidden_layers=None):
    """构建 BiEncoder 并打印参数量。"""
    model = BiEncoder(bert_path, pool=pool, dropout=dropout,
                      num_hidden_layers=num_hidden_layers)
    _print_param_info(model, f"BiEncoder (pool={pool}, layers={num_hidden_layers or 12})")
    return model


def build_crossencoder(bert_path, dropout=0.1, num_hidden_layers=None):
    """构建 CrossEncoder 并打印参数量。"""
    model = CrossEncoder(bert_path, dropout=dropout,
                         num_hidden_layers=num_hidden_layers)
    _print_param_info(model, f"CrossEncoder (layers={num_hidden_layers or 12})")
    return model


def _print_param_info(model, name):
    total = sum(p.numel() for p in model.parameters()) / 1e6
    bert  = sum(p.numel() for p in model.bert.parameters()) / 1e6
    print(f"模型: {name}")
    print(f"参数量: {total:.1f}M  (BERT 骨干: {bert:.1f}M)")
