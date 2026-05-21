import os
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader

# =========================================
# 1. tokenizer
# =========================================

tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-chinese"
)

# 特殊 token
special_tokens = {
    "additional_special_tokens": [
        "<|user|>",
        "<|assistant|>",
        "<|end|>"
    ]
}

tokenizer.add_special_tokens(special_tokens)

# =========================================
# 2. 读取 SFT 数据
# =========================================

with open("sft_train.txt", "r", encoding="utf-8") as f:
    text = f.read()

# =========================================
# 3. tokenize
# =========================================

tokens = tokenizer.encode(text)

data = torch.tensor(tokens)

# =========================================
# 4. Dataset
# =========================================

block_size = 128

class SFTDataset(Dataset):

    def __init__(self, data, block_size):

        self.data = data
        self.block_size = block_size

    def __len__(self):

        return len(self.data) - self.block_size

    def __getitem__(self, idx):

        x = self.data[idx:idx+block_size]

        y = self.data[idx+1:idx+block_size+1]

        return x, y

dataset = SFTDataset(data, block_size)

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True
)

# =========================================
# 5. GPT Model
# =========================================

class GPT(nn.Module):

    def __init__(
        self,
        vocab_size,
        d_model=256,
        nhead=8,
        num_layers=4
    ):
        super().__init__()

        self.token_emb = nn.Embedding(
            vocab_size,
            d_model
        )

        self.pos_emb = nn.Embedding(
            block_size,
            d_model
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.ln = nn.LayerNorm(d_model)

        self.head = nn.Linear(
            d_model,
            vocab_size
        )

    def causal_mask(self, size):

        return torch.triu(
            torch.ones(size, size) * float('-inf'),
            diagonal=1
        )

    def forward(self, x):

        B, T = x.shape

        pos = torch.arange(
            T,
            device=x.device
        )

        x = (
            self.token_emb(x)
            + self.pos_emb(pos)
        )

        mask = self.causal_mask(T).to(x.device)

        x = self.transformer(
            x,
            mask=mask
        )

        x = self.ln(x)

        logits = self.head(x)

        return logits

# =========================================
# 6. init
# =========================================

device = "cuda" if torch.cuda.is_available() else "cpu"

model = GPT(
    vocab_size=len(tokenizer)
).to(device)

model.token_emb.weight.data.normal_(0, 0.02)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4
)

criterion = nn.CrossEntropyLoss()

# =========================================
# 7. train
# =========================================
if __name__ == "__main__":
    epochs = 30

    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(epochs):

        model.train()

        total_loss = 0

        pbar = tqdm(loader)

        for x, y in pbar:

            x = x.to(device)
            y = y.to(device)

            logits = model(x)

            loss = criterion(
                logits.reshape(-1, len(tokenizer)),
                y.reshape(-1)
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

            pbar.set_postfix({
                "loss": loss.item()
            })

        avg_loss = total_loss / len(loader)

        print(f"epoch={epoch}")
        print(f"avg_loss={avg_loss:.4f}")

    # =========================================
    # 训练结束后保存最终模型
    # =========================================

    torch.save({
        "model": model.state_dict(),
    }, "mini_gpt_sft.pt")

    print("training finished")