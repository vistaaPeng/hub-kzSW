import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadSelfAttention(nn.Module):
    """多头自注意力层"""

    def __init__(self, hidden_size, num_attention_heads):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.output = nn.Linear(self.all_head_size, hidden_size)

    def transpose_for_scores(self, x):
        # x: [batch, seq_len, all_head_size] -> [batch, num_heads, seq_len, head_size]
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.num_attention_heads, self.attention_head_size)
        return x.permute(0, 2, 1, 3)

    def forward(self, x, attention_mask=None):
        # x: [batch, seq_len, hidden_size]
        q = self.transpose_for_scores(self.query(x))
        k = self.transpose_for_scores(self.key(x))
        v = self.transpose_for_scores(self.value(x))

        # q, k, v: [batch, num_heads, seq_len, head_size]
        # scores = q @ k^T / sqrt(head_size)
        #   q:          [batch, num_heads, seq_len, head_size]
        #   k^T:        [batch, num_heads, head_size, seq_len]
        #   q @ k^T:    [batch, num_heads, seq_len, seq_len]  每个token对所有token的相似度得分
        #   / sqrt(d):  缩放防止点积过大导致softmax梯度消失
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            scores = scores + attention_mask

        attn_weights = F.softmax(scores, dim=-1)
        # softmax(attn_weights) * v
        context = torch.matmul(attn_weights, v)  # [batch, num_heads, seq_len, head_size]

        # 拼接多头输出
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(context.size(0), context.size(1), self.all_head_size)

        return self.output(context)


class FeedForward(nn.Module):
    """前馈网络: hidden_size -> intermediate_size -> hidden_size"""

    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.dense1 = nn.Linear(hidden_size, intermediate_size)
        self.dense2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x):
        x = self.dense1(x)
        x = F.gelu(x)
        x = self.dense2(x)
        return x


class TransformerEncoderLayer(nn.Module):
    """单层Transformer Encoder: Self-Attention + FFN, 各带残差连接和LayerNorm"""

    def __init__(self, hidden_size, num_attention_heads, intermediate_size):
        super().__init__()
        self.attention = MultiHeadSelfAttention(hidden_size, num_attention_heads)
        self.attention_layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.ffn = FeedForward(hidden_size, intermediate_size)
        self.ffn_layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)

    def forward(self, x, attention_mask=None):
        # 自注意力 + 残差 + LayerNorm
        attn_output = self.attention(x, attention_mask)
        x = self.attention_layer_norm(x + attn_output)

        # 前馈网络 + 残差 + LayerNorm
        ff_output = self.ffn(x)
        x = self.ffn_layer_norm(x + ff_output)

        return x


class TransformerEmbedding(nn.Module):
    """词嵌入 + 位置嵌入 + 段落嵌入 + LayerNorm"""

    def __init__(self, vocab_size, hidden_size, max_position_embeddings=512, type_vocab_size=2):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)

    def forward(self, input_ids, token_type_ids=None):
        seq_len = input_ids.size(1)
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        embeddings = (
            self.word_embeddings(input_ids)
            + self.position_embeddings(position_ids)
            + self.token_type_embeddings(token_type_ids)
        )
        return self.layer_norm(embeddings)


class TransformerEncoder(nn.Module):
    """完整Transformer Encoder模型"""

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_attention_heads=12,
        intermediate_size=3072,
        num_hidden_layers=12,
        max_position_embeddings=512,
    ):
        super().__init__()
        self.embeddings = TransformerEmbedding(vocab_size, hidden_size, max_position_embeddings)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(hidden_size, num_attention_heads, intermediate_size)
            for _ in range(num_hidden_layers)
        ])
        self.pooler = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )

    def load_bert_weights(self, state_dict):
        """将HuggingFace BertModel的预训练权重加载到本模型中
        参数名映射关系:
            BertModel                              → 本模型
            embeddings.word_embeddings.weight      → embeddings.word_embeddings.weight
            embeddings.position_embeddings.weight  → embeddings.position_embeddings.weight
            embeddings.token_type_embeddings.weight → embeddings.token_type_embeddings.weight
            embeddings.LayerNorm.weight/bias       → embeddings.layer_norm.weight/bias
            encoder.layer.{i}.attention.self.query → layers.{i}.attention.query
            encoder.layer.{i}.attention.self.key   → layers.{i}.attention.key
            encoder.layer.{i}.attention.self.value → layers.{i}.attention.value
            encoder.layer.{i}.attention.output.dense → layers.{i}.attention.output
            encoder.layer.{i}.attention.output.LayerNorm → layers.{i}.attention_layer_norm
            encoder.layer.{i}.intermediate.dense   → layers.{i}.ffn.dense1
            encoder.layer.{i}.output.dense         → layers.{i}.ffn.dense2
            encoder.layer.{i}.output.LayerNorm     → layers.{i}.ffn_layer_norm
            pooler.dense                           → pooler.0 (nn.Sequential中的Linear层)
        """
        own_state = self.state_dict()
        mapping = {
            "embeddings.word_embeddings.weight": "embeddings.word_embeddings.weight",
            "embeddings.position_embeddings.weight": "embeddings.position_embeddings.weight",
            "embeddings.token_type_embeddings.weight": "embeddings.token_type_embeddings.weight",
            "embeddings.LayerNorm.weight": "embeddings.layer_norm.weight",
            "embeddings.LayerNorm.bias": "embeddings.layer_norm.bias",
            "pooler.dense.weight": "pooler.0.weight",
            "pooler.dense.bias": "pooler.0.bias",
        }
        # 每层Transformer的权重映射
        for i in range(len(self.layers)):
            mapping.update({
                f"encoder.layer.{i}.attention.self.query.weight": f"layers.{i}.attention.query.weight",
                f"encoder.layer.{i}.attention.self.query.bias": f"layers.{i}.attention.query.bias",
                f"encoder.layer.{i}.attention.self.key.weight": f"layers.{i}.attention.key.weight",
                f"encoder.layer.{i}.attention.self.key.bias": f"layers.{i}.attention.key.bias",
                f"encoder.layer.{i}.attention.self.value.weight": f"layers.{i}.attention.value.weight",
                f"encoder.layer.{i}.attention.self.value.bias": f"layers.{i}.attention.value.bias",
                f"encoder.layer.{i}.attention.output.dense.weight": f"layers.{i}.attention.output.weight",
                f"encoder.layer.{i}.attention.output.dense.bias": f"layers.{i}.attention.output.bias",
                f"encoder.layer.{i}.attention.output.LayerNorm.weight": f"layers.{i}.attention_layer_norm.weight",
                f"encoder.layer.{i}.attention.output.LayerNorm.bias": f"layers.{i}.attention_layer_norm.bias",
                f"encoder.layer.{i}.intermediate.dense.weight": f"layers.{i}.ffn.dense1.weight",
                f"encoder.layer.{i}.intermediate.dense.bias": f"layers.{i}.ffn.dense1.bias",
                f"encoder.layer.{i}.output.dense.weight": f"layers.{i}.ffn.dense2.weight",
                f"encoder.layer.{i}.output.dense.bias": f"layers.{i}.ffn.dense2.bias",
                f"encoder.layer.{i}.output.LayerNorm.weight": f"layers.{i}.ffn_layer_norm.weight",
                f"encoder.layer.{i}.output.LayerNorm.bias": f"layers.{i}.ffn_layer_norm.bias",
            })

        for bert_key, own_key in mapping.items():
            own_state[own_key].copy_(state_dict[bert_key])

    @classmethod
    def from_pretrained(cls, pretrained_path):
        """从预训练模型路径加载权重，验证自定义实现与HuggingFace输出一致"""
        from transformers import BertModel
        bert = BertModel.from_pretrained(pretrained_path, return_dict=False)
        state_dict = bert.state_dict()

        model = cls(
            vocab_size=bert.config.vocab_size,
            hidden_size=bert.config.hidden_size,
            num_attention_heads=bert.config.num_attention_heads,
            intermediate_size=bert.config.intermediate_size,
            num_hidden_layers=bert.config.num_hidden_layers,
            max_position_embeddings=bert.config.max_position_embeddings,
        )
        model.load_bert_weights(state_dict)
        model.eval()
        return model, bert

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        if attention_mask is not None:
            # 扩展维度: [batch, seq_len] -> [batch, 1, 1, seq_len]
            # 与注意力分数 [batch, num_heads, seq_len, seq_len] 广播
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            # 有效token(mask=1)变为0，padding位置(mask=0)变为-1e9(接近负无穷)
            # 加到注意力分数后，padding位置经softmax权重接近0，从而忽略padding token
            attention_mask = (1.0 - attention_mask.float()) * -1e9

        x = self.embeddings(input_ids, token_type_ids)

        for layer in self.layers:
            x = layer(x, attention_mask)

        pooler_output = self.pooler(x[:, 0])  # 取[CLS]位置的输出
        return x, pooler_output


if __name__ == "__main__":
    # ---- 随机初始化测试 ----
    model = TransformerEncoder(
        vocab_size=21128,
        hidden_size=768,
        num_attention_heads=12,
        intermediate_size=3072,
        num_hidden_layers=12,
    )
    print(f"参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    input_ids = torch.randint(0, 21128, (2, 6))
    sequence_output, pooler_output = model(input_ids)
    print(f"sequence_output: {sequence_output.shape}")  # [2, 6, 768]
    print(f"pooler_output:   {pooler_output.shape}")    # [2, 768]

    # ---- 加载预训练权重，对比HuggingFace输出 ----
    pretrained_path = r"E:\my\LLM\课程\week4语言模型\bert\bert-base-chinese"
    my_model, hf_model = TransformerEncoder.from_pretrained(pretrained_path)

    x = torch.LongTensor([[2450, 15486, 102, 2110]])  # 4个字的句子
    hf_seq, hf_pooler = hf_model(x)
    my_seq, my_pooler = my_model(x)

    print(f"\nsequence_output 最大误差: {(my_seq - hf_seq).abs().max().item():.2e}")
    print(f"pooler_output   最大误差: {(my_pooler - hf_pooler).abs().max().item():.2e}")
    print(f"输出一致: {torch.allclose(my_seq, hf_seq, atol=1e-4) and torch.allclose(my_pooler, hf_pooler, atol=1e-4)}")
