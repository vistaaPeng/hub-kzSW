"""逐层验证 DIY BERT 与 HuggingFace BERT 输出一致"""
import os
os.environ["HF_HOME"] = "M:/huggingface_cache"

import torch
from MyBert import MyBert
from load_weights import load_hf_bert, transfer_weights


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def verify_embedding(hf_bert, my_bert, input_ids):
    with torch.no_grad():
        hf_out = hf_bert.embeddings(input_ids)
        diy_out = my_bert.embeddings(input_ids)
    ok = torch.allclose(hf_out, diy_out, atol=1e-5)
    diff = (hf_out - diy_out).abs().max().item()
    print(f"  Embedding: {'PASS' if ok else 'FAIL'} (max diff={diff:.2e})")
    return ok


def verify_attention_layer(hf_bert, my_bert, hidden_states, attn_mask, layer_idx):
    with torch.no_grad():
        hf_out = hf_bert.encoder.layer[layer_idx].attention.self(hidden_states, attn_mask)[0]
        diy_out = my_bert.encoder.layer[layer_idx].attention(hidden_states, attn_mask)
    ok = torch.allclose(hf_out, diy_out, atol=1e-5)
    diff = (hf_out - diy_out).abs().max().item()
    print(f"  Layer {layer_idx} Attention: {'PASS' if ok else 'FAIL'} (max diff={diff:.2e})")
    return ok


def verify_ffn_layer(hf_bert, my_bert, hidden_states, layer_idx):
    hf_layer = hf_bert.encoder.layer[layer_idx]
    diy_layer = my_bert.encoder.layer[layer_idx]
    with torch.no_grad():
        hf_out = hf_layer.output(hf_layer.intermediate(hidden_states), hidden_states)
        diy_ffn = diy_layer.intermediate(hidden_states)
        diy_out = diy_layer.output(diy_ffn, hidden_states)
    ok = torch.allclose(hf_out, diy_out, atol=1e-5)
    diff = (hf_out - diy_out).abs().max().item()
    print(f"  Layer {layer_idx} FFN+Output: {'PASS' if ok else 'FAIL'} (max diff={diff:.2e})")
    return ok


def verify_full_model(hf_bert, my_bert, input_ids, attn_mask):
    with torch.no_grad():
        hf_out = hf_bert(input_ids, attention_mask=attn_mask)
        diy_hidden, diy_pooled = my_bert(input_ids, attention_mask=attn_mask)

    hidden_ok = torch.allclose(hf_out.last_hidden_state, diy_hidden, atol=1e-5)
    hidden_diff = (hf_out.last_hidden_state - diy_hidden).abs().max().item()
    pooler_ok = torch.allclose(hf_out.pooler_output, diy_pooled, atol=1e-5)
    pooler_diff = (hf_out.pooler_output - diy_pooled).abs().max().item()

    print(f"  Last hidden state: {'PASS' if hidden_ok else 'FAIL'} (max diff={hidden_diff:.2e})")
    print(f"  Pooler output:     {'PASS' if pooler_ok else 'FAIL'} (max diff={pooler_diff:.2e})")
    return hidden_ok and pooler_ok


def main():
    device = get_device()
    print("=" * 60)
    print(f"DIY BERT Verification  |  Device: {device}")
    print("=" * 60)

    print("\nLoading models...")
    hf_bert, config = load_hf_bert("bert-base-chinese", device=device)

    my_bert = MyBert(config)
    my_bert.eval().to(device)
    transfer_weights(my_bert, hf_bert.state_dict())
    print("Weights transferred successfully.")

    input_ids = torch.randint(0, config.vocab_size, (2, 64)).to(device)
    attn_mask_2d = torch.ones(2, 64).to(device)
    extended_mask = torch.zeros(2, 1, 1, 64).to(device)

    all_pass = True

    print("\n[1/3] Embedding")
    all_pass &= verify_embedding(hf_bert, my_bert, input_ids)

    with torch.no_grad():
        hf_emb = hf_bert.embeddings(input_ids)

    print("\n[2/3] Encoder Layers")
    for i in range(config.num_hidden_layers):
        all_pass &= verify_attention_layer(hf_bert, my_bert, hf_emb, extended_mask, i)
        attn_out = hf_bert.encoder.layer[i].attention(hf_emb, extended_mask)[0]
        all_pass &= verify_ffn_layer(hf_bert, my_bert, attn_out, i)
        hf_emb = hf_bert.encoder.layer[i](hf_emb, extended_mask)[0]

    print("\n[3/3] Full Model")
    all_pass &= verify_full_model(hf_bert, my_bert, input_ids, attn_mask_2d)

    print("\n" + "=" * 60)
    print(f"Result: {'ALL PASS' if all_pass else 'SOME FAILURES'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
