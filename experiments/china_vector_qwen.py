"""Reproduce the china-v1 method (diff-in-means + inference-time projection)
on a Qwen2.5 model.

  python china_vector_qwen.py extract  --model DIR --out-prefix PREFIX
  python china_vector_qwen.py generate --model DIR --dirs PKL --layers 3-32 --out FILE

The contrast prompts come from the china-v1 repository (data/pos.txt and
data/neg.txt); the held-out set is data/held.txt + data/held.meta.json.
"""
import argparse
import json
import os
import re
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "china-censorship-vector",
)
SYS = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


def parse_prompts(path):
    out = []
    for raw in open(path, encoding="utf-8"):
        raw = raw.rstrip("\n")
        if not raw.strip():
            continue
        s = raw.replace("\\\\n", "\n")
        m = re.search(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", s, re.S)
        out.append(m.group(1).strip() if m else s.strip())
    return out


def load_model(path, device="cuda"):
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.float16, device_map=device, low_cpu_mem_usage=True
    )
    model.eval()
    return tok, model


def encode(tok, q):
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": q}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tok(text, return_tensors="pt").to("cuda")


@torch.no_grad()
def capture(tok, model, prompts, tag=""):
    acts = []
    for i, q in enumerate(prompts):
        inp = encode(tok, q)
        out = model(**inp, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states[1:]  # after each decoder layer
        a = torch.stack([h[0, -1, :].float().cpu() for h in hs]).numpy()
        acts.append(a)
        if (i + 1) % 20 == 0:
            print(f"  capture {tag} {i+1}/{len(prompts)}", flush=True)
    return np.stack(acts)


def auc(scores, labels):
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def cmd_extract(a):
    tok, model = load_model(a.model)
    pos = parse_prompts(f"{BASE}/data/pos.txt")
    neg = parse_prompts(f"{BASE}/data/neg.txt")
    held = parse_prompts(f"{BASE}/data/held.txt")
    meta = json.load(open(f"{BASE}/data/held.meta.json", encoding="utf-8"))
    print(f"prompts: pos={len(pos)} neg={len(neg)} held={len(held)}", flush=True)
    if a.limit:
        pos, neg, held, meta = pos[:a.limit], neg[:a.limit], held[:a.limit], meta[:a.limit]
        print(f"limited to {a.limit} each", flush=True)

    ap = capture(tok, model, pos, "pos")
    an = capture(tok, model, neg, "neg")
    ah = capture(tok, model, held, "held")
    np.savez_compressed(a.out_prefix + "_acts.npz", pos=ap.astype(np.float16),
                        neg=an.astype(np.float16), held=ah.astype(np.float16))

    diff = ap.mean(0) - an.mean(0)              # [L, H]
    dirs = diff / np.linalg.norm(diff, axis=1, keepdims=True)
    np.save(a.out_prefix + "_dirs.npy", dirs.astype(np.float32))

    labels = np.array([1 if m[1] == "china" else 0 for m in meta])
    scores = np.einsum("nlh,lh->nl", ah.astype(np.float32), dirs)
    aucs = [auc(scores[:, l], labels) for l in range(dirs.shape[0])]
    print("layer AUC (held-out, china vs control):", flush=True)
    for l, v in enumerate(aucs):
        print(f"  {l:2d}  {v:.3f}", flush=True)
    json.dump({"auc": aucs}, open(a.out_prefix + "_auc.json", "w"), indent=2)


class ProjectHook:
    def __init__(self, d):
        self.d = torch.as_tensor(d, dtype=torch.float32)

    def __call__(self, module, args, output):
        is_tuple = isinstance(output, tuple)
        h = output[0] if is_tuple else output
        d = self.d.to(h.device)
        h32 = h.float()
        h32 = h32 - (h32 * d).sum(-1, keepdim=True) * d
        h = h32.to(h.dtype)
        return (h,) + output[1:] if is_tuple else h


def parse_layers(s, n):
    if s in ("all", ""):
        return list(range(n))
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return [l for l in out if 0 <= l < n]


@torch.no_grad()
def generate(tok, model, prompts, meta, max_new, tag):
    rows = []
    for i, q in enumerate(prompts):
        inp = encode(tok, q)
        t = time.time()
        out = model.generate(**inp, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        text = tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
        m = meta[i]
        rows.append({"idx": i, "topic": m[0], "group": m[1], "lang": m[2],
                     "prompt": q, "response": text,
                     "elapsed_s": round(time.time() - t, 2)})
        if (i + 1) % 5 == 0:
            print(f"  gen {tag} {i+1}/{len(prompts)}", flush=True)
    return rows


def cmd_generate(a):
    tok, model = load_model(a.model)
    held = parse_prompts(f"{BASE}/data/held.txt")
    meta = json.load(open(f"{BASE}/data/held.meta.json", encoding="utf-8"))
    if a.limit:
        held, meta = held[:a.limit], meta[:a.limit]
    dirs = np.load(a.dirs)
    n = dirs.shape[0]
    layers = parse_layers(a.layers, n)
    print(f"model layers={n}, projecting layers={layers}", flush=True)

    base_path = a.out + "_base.jsonl"
    if os.path.exists(base_path) and sum(1 for _ in open(base_path, encoding="utf-8")) == len(held):
        base = [json.loads(l) for l in open(base_path, encoding="utf-8")]
        print("reusing", base_path, flush=True)
    else:
        base = generate(tok, model, held, meta, a.max_new, "base")
        with open(base_path, "w", encoding="utf-8") as f:
            for r in base:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    handles = [model.model.layers[l].register_forward_hook(ProjectHook(dirs[l])) for l in layers]
    try:
        proj = generate(tok, model, held, meta, a.max_new, "proj")
    finally:
        for h in handles:
            h.remove()
    with open(a.out + "_proj.jsonl", "w", encoding="utf-8") as f:
        for r in proj:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("wrote", a.out + "_base.jsonl", a.out + "_proj.jsonl", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--model", required=True)
    e.add_argument("--out-prefix", required=True)
    e.add_argument("--limit", type=int, default=0)
    e.set_defaults(fn=cmd_extract)
    g = sub.add_parser("generate")
    g.add_argument("--model", required=True)
    g.add_argument("--dirs", required=True)
    g.add_argument("--layers", default="all")
    g.add_argument("--max-new", type=int, default=100)
    g.add_argument("--limit", type=int, default=0)
    g.add_argument("--out", required=True)
    g.set_defaults(fn=cmd_generate)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
