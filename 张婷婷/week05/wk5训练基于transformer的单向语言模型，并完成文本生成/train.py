
import os
import torch
import torch.nn as nn
import torch.optim as optim
from config import *
from dataset import build_dataloader
from model import DecoderOnlyLM

def main():
    # ✅ 自动创建文件夹（存在则不报错）
    os.makedirs("./data", exist_ok=True)
    os.makedirs("./checkpoints", exist_ok=True)

    # 加载数据
    loader, vocab_size, char2idx, idx2char = build_dataloader(
        file_path="./data/poems.txt",
        seq_len=SEQ_LEN,
        batch_size=BATCH_SIZE
    )
    # 初始化模型
    model = DecoderOnlyLM(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        n_layer=N_LAYER,
        n_head=N_HEAD,
        d_ff=D_FF,
        seq_len=SEQ_LEN,
        dropout=DROPOUT
    ).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR)

    print(f"Start training on {DEVICE}, vocab size: {vocab_size}")
    for epoch in range(EPOCHS):
        total_loss = 0.0
        model.train()
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            logits = model(batch_x)
            logits = logits.transpose(1, 2)
            loss = loss_fn(logits, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1:3d} | Avg Loss: {avg_loss:.4f}")
    # 保存模型
    torch.save({
        "model_state": model.state_dict(),
        "vocab_size": vocab_size,
        "char2idx": char2idx,
        "idx2char": idx2char
    }, SAVE_PATH)
    print(f"Model saved to {SAVE_PATH}")

if __name__ == "__main__":
    main()