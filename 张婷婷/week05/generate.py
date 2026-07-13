
import torch
from config import *
from model import DecoderOnlyLM

def generate(model, prompt, char2idx, idx2char, max_len=MAX_GEN_LEN):
    model.eval()
    tokens = [char2idx[c] for c in prompt if c in char2idx]
    gen = torch.tensor([tokens], dtype=torch.long).to(DEVICE)
    with torch.no_grad():
        for _ in range(max_len):
            logits = model(gen)
            last = logits[:, -1, :]
            next_token = torch.argmax(last, dim=-1, keepdim=True)
            gen = torch.cat([gen, next_token], dim=-1)
    return "".join([idx2char[int(i)] for i in gen[0]])

def main():
    # 加载保存的模型和词表
    ckpt = torch.load(SAVE_PATH, map_location=DEVICE)
    vocab_size = ckpt["vocab_size"]
    char2idx = ckpt["char2idx"]
    idx2char = ckpt["idx2char"]
    model = DecoderOnlyLM(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        n_layer=N_LAYER,
        n_head=N_HEAD,
        d_ff=D_FF,
        seq_len=SEQ_LEN
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    # 测试生成
    test_prompts = ["床前", "白日", "春眠"]
    for p in test_prompts:
        result = generate(model, p, char2idx, idx2char)
        print(f"Prompt: {p} -> {result}\n")

if __name__ == "__main__":
    main()