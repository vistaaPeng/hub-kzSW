import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from train import GPT

# =========================================
# 1. tokenizer
# =========================================

tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-chinese"
)

special_tokens = {
    "additional_special_tokens": [
        "<|user|>",
        "<|assistant|>",
        "<|end|>"
    ]
}

tokenizer.add_special_tokens(special_tokens)

# =========================================
# 2. load model
# =========================================

device = "cuda" if torch.cuda.is_available() else "cpu"

model = GPT(
    vocab_size=len(tokenizer)
).to(device)

ckpt = torch.load(
    "mini_gpt_sft.pt",
    map_location=device
)

model.load_state_dict(ckpt["model"])

model.eval()

# =========================================
# 3. generate
# =========================================

@torch.no_grad()
def generate(
    prompt,
    max_new_tokens=32,
    temperature=0.7,
    top_p=0.9
):

    ids = tokenizer.encode(
        prompt,
        return_tensors="pt"
    ).to(device)

    for _ in range(max_new_tokens):

        logits = model(ids)

        logits = logits[:, -1, :]

        logits = logits / temperature

        probs = F.softmax(logits, dim=-1)

        # top-p
        sorted_probs, sorted_indices = torch.sort(
            probs,
            descending=True
        )

        cumulative_probs = torch.cumsum(
            sorted_probs,
            dim=-1
        )

        mask = cumulative_probs > top_p

        mask[...,1:] = mask[...,:-1].clone()

        mask[...,0] = False

        sorted_probs[mask] = 0

        sorted_probs = sorted_probs / sorted_probs.sum()

        next_token = torch.multinomial(
            sorted_probs,
            1
        )

        next_id = sorted_indices.gather(
            -1,
            next_token
        )

        ids = torch.cat(
            [ids, next_id],
            dim=1
        )

        # EOS
        token_text = tokenizer.decode(
            next_id[0]
        )

        if "<|end|>" in token_text:
            break

    return tokenizer.decode(ids[0])

# =========================================
# 4. interactive chat
# =========================================

while True:

    user_input = input("\n用户: ")

    if user_input.lower() in ["exit", "quit"]:
        break

    prompt = f"""
    <|user|>
    {user_input}
    
    <|assistant|>
    """

    result = generate(prompt)

    # 只截取 assistant 后面的内容
    answer = result.split("<|assistant|>")[-1]

    # 遇到 end token 截断
    answer = answer.split("<|end|>")[0]

    print("\n模型:")
    print(answer.strip())