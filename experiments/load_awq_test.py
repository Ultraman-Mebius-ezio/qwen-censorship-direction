import os, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

P = os.environ.get("QWEN_AWQ", r"C:\My_Programe\censored\models\qwen2.5-7b-awq")
tok = AutoTokenizer.from_pretrained(P)
print("tokenizer ok", flush=True)

t = time.time()
model = AutoModelForCausalLM.from_pretrained(
    P,
    device_map="auto",
    dtype=torch.float16,
    max_memory={0: "4GiB", "cpu": "8GiB"},
    low_cpu_mem_usage=True,
)
print("model loaded in %.1fs" % (time.time() - t), flush=True)
print("device map:", getattr(model, "hf_device_map", None), flush=True)

msgs = [{"role": "user", "content": "用一句话打个招呼。"}]
text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
inputs = tok(text, return_tensors="pt")
dev = model.get_input_embeddings().weight.device
inputs = {k: v.to(dev) for k, v in inputs.items()}
t = time.time()
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=40, do_sample=False, pad_token_id=tok.eos_token_id)
print("generated in %.1fs" % (time.time() - t), flush=True)
print(tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True), flush=True)
