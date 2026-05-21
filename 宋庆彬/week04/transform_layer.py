import torch
import torch.nn as nn
import math


# ==================== 1. Multi-Head Attention（Encoder/Decoder 共用）====================
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        # (batch, seq_len, d_model) → (batch, num_heads, seq_len, d_k)
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, V)

        # (batch, num_heads, seq_len, d_k) → (batch, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        output = self.W_o(attn_output)
        return output, attn_weights


# ==================== 2. Position-wise Feed-Forward（Encoder/Decoder 共用）====================
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))


# ==================== 3. Positional Encoding ====================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ==================== 4. Encoder Layer（2 个子层）====================
#  子层1: Multi-Head Self-Attention → Add & Norm
#  子层2: Feed-Forward → Add & Norm
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 子层1: Self-Attention
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # 子层2: Feed-Forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x


# ==================== 5. Decoder Layer（3 个子层）====================
#  子层1: Masked Multi-Head Self-Attention → Add & Norm
#  子层2: Multi-Head Cross-Attention → Add & Norm  (Q=Decoder, K/V=Encoder输出)
#  子层3: Feed-Forward → Add & Norm
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # 子层1: Masked Self-Attention（因果mask，只看当前位置之前）
        attn_output, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))

        # 子层2: Cross-Attention（Q=Decoder, K/V=Encoder输出）
        attn_output, _ = self.cross_attn(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))

        # 子层3: Feed-Forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))

        return x


# ==================== 6. Encoder ====================
class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_len=5000, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x


# ==================== 7. Decoder ====================
class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_len=5000, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.fc(x)


# ==================== 8. 完整 Transformer（Encoder + Decoder）====================
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, d_ff,
                 enc_layers, dec_layers, max_len=5000, dropout=0.1):
        super().__init__()
        self.encoder = TransformerEncoder(src_vocab_size, d_model, num_heads, d_ff,
                                          enc_layers, max_len, dropout)
        self.decoder = TransformerDecoder(tgt_vocab_size, d_model, num_heads, d_ff,
                                          dec_layers, max_len, dropout)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        encoder_output = self.encoder(src, src_mask)
        decoder_output = self.decoder(tgt, encoder_output, src_mask, tgt_mask)
        return decoder_output


# ==================== 9. 工具函数：生成 mask ====================
def generate_causal_mask(seq_len, device):
    """生成因果 mask（上三角为0），用于 Decoder 自注意力"""
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    return ~mask  # 0 的位置表示不可见


# ==================== 10. 测试与示例 ====================
if __name__ == '__main__':
    print("=" * 60)
    print("1. 维度正确性测试")
    print("=" * 60)

    batch, src_len, tgt_len, d_model, num_heads, d_ff = 2, 10, 8, 64, 8, 256

    x_enc = torch.randn(batch, src_len, d_model)
    x_dec = torch.randn(batch, tgt_len, d_model)

    # Encoder Layer：2 个子层
    enc_layer = TransformerEncoderLayer(d_model, num_heads, d_ff)
    out_enc = enc_layer(x_enc)
    print(f"EncoderLayer (2子层)     输入: {x_enc.shape} → 输出: {out_enc.shape}   ✓")

    # Decoder Layer：3 个子层（带 causal mask）
    dec_layer = TransformerDecoderLayer(d_model, num_heads, d_ff)
    causal_mask = torch.ones(batch, 1, tgt_len, tgt_len) * \
                  generate_causal_mask(tgt_len, x_dec.device).unsqueeze(0).unsqueeze(0)
    out_dec = dec_layer(x_dec, out_enc, tgt_mask=causal_mask)
    print(f"DecoderLayer (3子层)     输入: {x_dec.shape} → 输出: {out_dec.shape}   ✓")

    print()

    # --- 结构对比 ---
    print("=" * 60)
    print("2. 结构对比")
    print("=" * 60)
    print(f"""
    ┌─────────────────────────────────────────┐
    │  Encoder Layer（2 个子层）              │
    │    1. Self-Attention   → Add & Norm     │
    │    2. Feed-Forward     → Add & Norm     │
    ├─────────────────────────────────────────┤
    │  Decoder Layer（3 个子层）              │
    │    1. Masked Self-Attn → Add & Norm     │
    │    2. Cross-Attention  → Add & Norm     │
    │    3. Feed-Forward     → Add & Norm     │
    └─────────────────────────────────────────┘
    """)

    # --- seq2seq 复制任务 ---
    print("=" * 60)
    print("3. seq2seq 复制任务（Encoder + Decoder 完整测试）")
    print("=" * 60)

    import random
    from torch.utils.data import Dataset, DataLoader

    char_pool = list(
        "的一是在不了有和人这中大为上个国我以要他时来用们生到作地于出就分对成会可主发年动同工也能下过子说产种面而方后多定行学法所民得经十三之进着等部度家电力里如水化高自二理起小物现实加量都两体制机当使点从业本去把性好应开它合还因由其些然前外天政四日那社义事平形相全表间样与关各重新线内数正心反你明看原又么利比或但质气第向道命此变条只没结解问意建月公无系军很情者最立代想已通并提直题党程展五果料象员革位入常文总次品式活设及管特件长求老头基资边流路级少图山统接知较将组见计别她手角期根论运农指几九区强放决西被干做必战先回则任取据处队南给色光门即保治北造百规热领七海口东导器压志世金增争济阶油思术极交受联什认六共权收证改清己美再采转更单风切打白教速花带安场身车例真务具万每目至达走积示议声报斗完类八离华名确才科张信马节话米整空元况今集温传土许步群广石记需段研界拉林律叫且究观越织装影算低持音众书布复容儿须际商非验连断深难近矿千周委素技备半办青省列习响约支般史感劳便团往酸历市克何除消构府称太准精值号率族维划选标写存候毛亲快效斯院查江型眼王按格养易置派层片始却专状育厂京识适属圆包火住调满县局照参红细引听该铁价严"
    )

    all_chars = sorted(set(char_pool))
    char_to_idx = {ch: i + 1 for i, ch in enumerate(all_chars)}
    idx_to_char = {i + 1: ch for i, ch in enumerate(all_chars)}
    vocab_size = len(char_to_idx) + 2  # 0=padding, 1=BOS, 2..N=chars
    BOS_IDX = 1

    def generate_copy_sample(seq_len=5):
        chars = random.choices(char_pool, k=seq_len)
        src = ''.join(chars)
        return src, src  # 输入=输出（复制任务）

    def encode(s):
        return torch.tensor([char_to_idx[ch] for ch in s], dtype=torch.long)

    def decode(indices, skip_special=True):
        chars = []
        for i in indices:
            i = i.item() if isinstance(i, torch.Tensor) else i
            if i == 0:
                break
            if skip_special and i == BOS_IDX:
                continue
            chars.append(idx_to_char.get(i, '?'))
        return ''.join(chars)

    class CopyDataset(Dataset):
        def __init__(self, num_samples, seq_len=5):
            self.src, self.tgt = [], []
            for _ in range(num_samples):
                s, t = generate_copy_sample(seq_len)
                self.src.append(s)
                self.tgt.append(t)

        def __len__(self):
            return len(self.src)

        def __getitem__(self, idx):
            src = encode(self.src[idx])
            tgt = torch.cat([torch.tensor([BOS_IDX]), encode(self.tgt[idx])])
            return src, tgt

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    D_MODEL = 64
    NUM_HEADS = 4
    D_FF = 128
    ENC_LAYERS = 6           # 原论文：Encoder × 6
    DEC_LAYERS = 6           # 原论文：Decoder × 6
    BATCH_SIZE = 64
    EPOCHS = 15
    LR = 0.001
    SEQ_LEN = 5

    dataset = CopyDataset(3000, seq_len=SEQ_LEN)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        enc_layers=ENC_LAYERS,
        dec_layers=DEC_LAYERS,
        dropout=0.1
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    def train_epoch(model, loader, criterion, optimizer):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            tgt_mask = torch.ones(1, 1, tgt_input.size(1), tgt_input.size(1)).to(device) * \
                       generate_causal_mask(tgt_input.size(1), device).unsqueeze(0).unsqueeze(0)

            optimizer.zero_grad()
            outputs = model(src, tgt_input, tgt_mask=tgt_mask)
            loss = criterion(outputs.reshape(-1, vocab_size), tgt_output.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * src.size(0)
            _, predicted = torch.max(outputs, 2)
            correct += (predicted == tgt_output).sum().item()
            total += tgt_output.numel()
        return total_loss / total, correct / total

    def evaluate(model, loader, criterion):
        model.eval()
        total_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for src, tgt in loader:
                src, tgt = src.to(device), tgt.to(device)
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]

                tgt_mask = torch.ones(1, 1, tgt_input.size(1), tgt_input.size(1)).to(device) * \
                           generate_causal_mask(tgt_input.size(1), device).unsqueeze(0).unsqueeze(0)

                outputs = model(src, tgt_input, tgt_mask=tgt_mask)
                loss = criterion(outputs.reshape(-1, vocab_size), tgt_output.reshape(-1))
                total_loss += loss.item() * src.size(0)
                _, predicted = torch.max(outputs, 2)
                correct += (predicted == tgt_output).sum().item()
                total += tgt_output.numel()
        return total_loss / total, correct / total

    print(f"\n模型结构: Encoder={ENC_LAYERS}层(2子层), Decoder={DEC_LAYERS}层(3子层)")
    print(f"任务: 输入5个字符 → 输出相同5个字符 (seq2seq复制)\n")
    print("开始训练...")
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        test_loss, test_acc = evaluate(model, test_loader, criterion)
        print(f"Epoch {epoch:2d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}")

    # 推理测试
    print("\n推理测试（自回归生成）：")
    model.eval()
    test_cases = ["你好世界天", "有你我他她", "的一是在不"]
    for s in test_cases:
        src = encode(s).unsqueeze(0).to(device)
        with torch.no_grad():
            enc_out = model.encoder(src)
            generated = [BOS_IDX]
            for _ in range(SEQ_LEN):
                tgt = torch.tensor([generated]).to(device)
                tgt_mask = torch.ones(1, 1, tgt.size(1), tgt.size(1)).to(device) * \
                           generate_causal_mask(tgt.size(1), device).unsqueeze(0).unsqueeze(0)
                out = model.decoder(tgt, enc_out, tgt_mask=tgt_mask)
                next_token = out[0, -1].argmax().item()
                generated.append(next_token)
            result = decode(generated)
        print(f"  输入: '{s}' → 输出: '{result}' {'✓' if s == result else '✗'}")
