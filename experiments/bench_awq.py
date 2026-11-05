import os, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

P = os.environ.get("QWEN_AWQ", r"C:\My_Programe\censored\models\qwen2.5-7b-awq")
tok = AutoTokenizer.from_pretrained(P)
t = time.time()
model = AutoModelForCausalLM.from_pretrained(
    P, device_map="auto", dtype=torch.float16,
    max_memory={0: "7200MiB", "cpu": "4GiB"},
    low_cpu_mem_usage=True, offload_buffers=True,
)
print("loaded %.1fs" % (time.time() - t), flush=True)
dm = getattr(model, "hf_device_map", {})
gpu = [k for k, v in dm.items() if v == 0]
print("GPU modules:", len(gpu), "CPU modules:", len(dm) - len(gpu), flush=True)

msgs = [{"role": "user", "content": "用一句话解释什么是光合作用。"}]
text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
inputs = tok(text, return_tensors="pt")
dev = model.get_input_embeddings().weight.device
inputs = {k: v.to(dev) for k, v in inputs.items()}
for cap in (20, 20):
    t = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=cap, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    dt = time.time() - t
    n = out.shape[1] - inputs["input_ids"].shape[1]
    print("gen %d tok in %.2fs = %.2f tok/s" % (n, dt, n / dt), flush=True)
print(tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
