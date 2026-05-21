import math
import torch
from transformers import BertConfig


class BertEmbedding(torch.nn.Module):
    """
    Embedding 层：word + position + token_type → LayerNorm → Dropout

    参数量 (bert-base-chinese):
      word_embeddings:     V × H  = 21128 × 768 = 16,226,304
      position_embeddings: P × H  =   512 × 768 =    393,216
      token_type_embeddings:T × H  =     2 × 768 =      1,536
      LayerNorm:          2 × H                =      1,536
      ─────────────────────────────────────────────────────────
      合计: V×H + P×H + T×H + 2H           = 16,622,592
    """
    def __init__(self, config: BertConfig):
        super().__init__()
        self.word_embeddings = torch.nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = torch.nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = torch.nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = torch.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).unsqueeze(0), persistent=False)

    def forward(self, input_ids: torch.Tensor, token_type_ids: torch.Tensor = None):
        seq_length = input_ids.size(1)
        position_ids = self.position_ids[:, :seq_length]
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        word_emb = self.word_embeddings(input_ids)
        tok_emb = self.token_type_embeddings(token_type_ids)
        embeddings = word_emb + tok_emb
        embeddings = embeddings + self.position_embeddings(position_ids)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(torch.nn.Module):
    """
    多头自注意力：Q/K/V 投影 → 分头 → scaled dot-product → 合并

    参数量 (bert-base-chinese, H=768, A=12):
      Q: H × H + H = 768² + 768 = 590,592
      K: H × H + H = 590,592
      V: H × H + H = 590,592
      ─────────────────────────────────────
      合计: 3 × (H² + H)   = 1,771,776
    """
    def __init__(self, config: BertConfig):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(f"hidden_size {config.hidden_size} not divisible by num_heads {config.num_attention_heads}")
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = torch.nn.Linear(config.hidden_size, self.all_head_size)
        self.key = torch.nn.Linear(config.hidden_size, self.all_head_size)
        self.value = torch.nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = torch.nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor):
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor = None):
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_shape)
        return context_layer


class BertSublayerConnection(torch.nn.Module):
    """
    残差连接块：dense(input_size → H) → Dropout → LayerNorm(x + residual)
    Attention 和 FFN 子层各用一个，仅 input_size 不同。

    参数量 (bert-base-chinese, H=768):
      Attention 侧 (input_size=H):
        dense:     H × H + H =   768² +   768 =    590,592
        LayerNorm:     2 × H =     2 ×   768 =      1,536
        ────────────────────────────────────────────────────
        合计:     H² + 3H                   =    592,128

      FFN 侧 (input_size=I=3072):
        dense:     I × H + H =  3072 × 768 + 768 =  2,359,296
        LayerNorm:     2 × H =     2 ×   768       =      1,536
        ────────────────────────────────────────────────────
        合计:     I×H + H + 2H              =  2,361,600
    """
    def __init__(self, config: BertConfig, input_size: int):
        super().__init__()
        self.dense = torch.nn.Linear(input_size, config.hidden_size)
        self.LayerNorm = torch.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertFeedForward(torch.nn.Module):
    """
    FFN 中间层：dense(H → I) → GELU

    参数量 (bert-base-chinese, H=768, I=3072):
      dense: H × I + I = 768 × 3072 + 3072 = 2,362,368
    """
    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = torch.nn.Linear(config.hidden_size, config.intermediate_size)

    def forward(self, hidden_states: torch.Tensor):
        return torch.nn.functional.gelu(self.dense(hidden_states))


class BertEncoderLayer(torch.nn.Module):
    """
    单层 Encoder：SelfAttention → SublayerConnection → FeedForward → SublayerConnection

    参数量 (bert-base-chinese):
      BertSelfAttention:          3 × (H² + H)  =  1,771,776
      BertSublayerConnection(attn):  H² + 3H    =    592,128
      BertFeedForward:              H×I + I     =  2,362,368
      BertSublayerConnection(ffn):  I×H + H + 2H=  2,361,600
      ──────────────────────────────────────────────────────
      合计                                =  7,087,872
    """
    def __init__(self, config: BertConfig):
        super().__init__()
        self.attention = BertSelfAttention(config)
        self.attention_output = BertSublayerConnection(config, config.hidden_size)
        self.intermediate = BertFeedForward(config)
        self.output = BertSublayerConnection(config, config.intermediate_size)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor = None):
        attn_output = self.attention(hidden_states, attention_mask)
        attn_output = self.attention_output(attn_output, hidden_states)
        ffn_output = self.intermediate(attn_output)
        layer_output = self.output(ffn_output, attn_output)
        return layer_output


class BertEncoder(torch.nn.Module):
    """
    L 层 EncoderLayer 堆叠

    参数量 (bert-base-chinese, L=12):
      12 × 7,087,872 = 85,054,464
    """
    def __init__(self, config: BertConfig):
        super().__init__()
        self.layer = torch.nn.ModuleList([BertEncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor = None):
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
        return hidden_states


class BertPooler(torch.nn.Module):
    """
    Pooler：取 [CLS] token → dense → Tanh

    参数量 (bert-base-chinese):
      dense: H × H + H = 768² + 768 = 590,592
    """
    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = torch.nn.Tanh()

    def forward(self, hidden_states: torch.Tensor):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class MyBert(torch.nn.Module):
    """
    完整 BERT 模型

    总参数量 (bert-base-chinese):
      BertEmbedding:       16,622,592
      BertEncoder (×12):  85,054,464
      BertPooler:            590,592
      ────────────────────────────────
      合计:              102,267,648  ≈ 102.3M

    分布:
      Embedding:  16.3%
      Attention:  20.8%  (12 × 1,771,776 = 21,261,312)
      FFN:        55.5%  (12 × (2,362,368 + 2,361,600) = 56,687,616)
      Pooler:      0.6%
      LayerNorm:    0.1%  (12 × 4 × 1,536 + 1,536 = 75,264)
    """
    def __init__(self, config: BertConfig):
        super().__init__()
        self.embeddings = BertEmbedding(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None, token_type_ids: torch.Tensor = None):
        if attention_mask is not None:
            extended_attention_mask = (1.0 - attention_mask[:, None, None, :].to(torch.float32)) * -10000.0
        else:
            extended_attention_mask = None

        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoder_output = self.encoder(embedding_output, extended_attention_mask)
        pooled_output = self.pooler(encoder_output)
        return encoder_output, pooled_output
