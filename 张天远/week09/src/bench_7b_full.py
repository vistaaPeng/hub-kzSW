import gc, json, time, torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = '/root/autodl-tmp/huggingface_cache/Qwen2.5-7B-Instruct/Qwen/Qwen2___5-7B-Instruct'
N = 20
MAX_TOK = 50
PROMPTS = ['什么是股票？','什么是基金？','什么是ETF？','什么是债券？','什么是期权？'] * 4
PROMPTS = PROMPTS[:N]

# ── serial ──
tok = AutoTokenizer.from_pretrained(MODEL); tok.pad_token = tok.eos_token
m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="cuda"); m.eval()
t0 = time.time(); total = 0
for p in PROMPTS:
    inp = tok(p, return_tensors="pt").to("cuda"); inp.pop("token_type_ids", None)
    out = m.generate(**inp, max_new_tokens=MAX_TOK, do_sample=False, pad_token_id=tok.pad_token_id)
    total += out.shape[1] - inp["input_ids"].shape[1]
dt_s = time.time() - t0
print(f"serial: {dt_s:.1f}s QPS={N/dt_s:.1f}")
del m; gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()

# ── batch=4 ──
tok.padding_side = "left"
m2 = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="cuda"); m2.eval()
BS = 4
t0 = time.time(); total2 = 0
for i in range(0, N, BS):
    batch = PROMPTS[i:i+BS]; enc = tok(batch, return_tensors="pt", padding=True).to("cuda")
    enc.pop("token_type_ids", None)
    out = m2.generate(**enc, max_new_tokens=MAX_TOK, do_sample=False, pad_token_id=tok.pad_token_id)
    total2 += (out[:, enc["input_ids"].shape[1]:] != tok.pad_token_id).sum().item()
dt_b = time.time() - t0
print(f"batch=4: {dt_b:.1f}s QPS={N/dt_b:.1f}")
del m2; gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()

# ── vLLM ──
llm = LLM(model=MODEL, max_model_len=256, gpu_memory_utilization=0.9, dtype="float16", enforce_eager=True)
tok2 = llm.get_tokenizer()
cp = [tok2.apply_chat_template([{"role":"user","content":q}], tokenize=False, add_generation_prompt=True) for q in PROMPTS]
t0 = time.time()
out = llm.generate(cp, SamplingParams(temperature=0, max_tokens=MAX_TOK))
dt_v = time.time() - t0
tot_v = sum(len(o.outputs[0].token_ids) for o in out)
print(f"vLLM: {dt_v:.1f}s QPS={N/dt_v:.1f}")

r = {"serial":{"time":dt_s,"qps":N/dt_s,"tps":total/dt_s},
     "batch":{"time":dt_b,"qps":N/dt_b,"tps":total2/dt_b},
     "vllm":{"time":dt_v,"qps":N/dt_v,"tps":tot_v/dt_v}}
with open("/root/autodl-tmp/vllm_deployment/outputs/throughput_7B.json","w") as f:
    json.dump({"model":MODEL,"model_name":"Qwen2.5-7B-Instruct","n":N,"max_tokens":MAX_TOK,"results":r}, f, indent=2)
sp = r["vllm"]["qps"] / r["serial"]["qps"]
print(f"speedup: {sp:.1f}x")
print("saved")
