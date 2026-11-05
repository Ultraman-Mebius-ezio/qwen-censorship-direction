"""Difference-of-means direction per layer from last-token dumps; writes a llama.cpp controlvector GGUF.
usage: make_dir.py pos.bin neg.bin held.bin held.meta.json out.gguf [--exclude-china-control-overlap]"""
import json, sys
import numpy as np
# needs the gguf package: pip install gguf (or llama.cpp/gguf-py on PYTHONPATH)
import gguf

L, H, E = 48, 4, 2560
def load(p):
    a = np.fromfile(p, dtype=np.float32)
    return a.reshape(-1, L, H, E)

pos, neg, held = load(sys.argv[1]), load(sys.argv[2]), load(sys.argv[3])
meta = json.load(open(sys.argv[4]))
out = sys.argv[5]
print("pos", pos.shape, "neg", neg.shape, "held", held.shape)

# per layer: mean over prompts, then over hyper-connection streams
diff = (pos.mean(0) - neg.mean(0)).mean(1)             # L x E
dirs = diff / np.linalg.norm(diff, axis=1, keepdims=True)

def auc(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float((a[:, None] > b[None, :]).mean() + 0.5 * (a[:, None] == b[None, :]).mean())

china = [i for i, m in enumerate(meta) if m[1] == "china"]
ctrl = [i for i, m in enumerate(meta) if m[1] == "control"]
print("layer  |diff|  train-AUC  heldout-AUC (china vs control, projection on direction)")
for l in range(L):
    pp = (pos[:, l].mean(1) @ dirs[l]); nn = (neg[:, l].mean(1) @ dirs[l])
    hp = held[:, l].mean(1) @ dirs[l]
    print(f"{l:5d} {np.linalg.norm(diff[l]):7.2f} {auc(pp, nn):9.3f} {auc(hp[china], hp[ctrl]):11.3f}")

w = gguf.GGUFWriter(out, "controlvector")
w.add_string("controlvector.model_hint", "qwen4exp")
w.add_int32("controlvector.layer_count", L - 1)
for l in range(L - 1):                       # cvector-generator convention: direction.{il+1} from layer il
    w.add_tensor(f"direction.{l + 1}", dirs[l].astype(np.float32))
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
np.save(out.replace(".gguf", ".dirs.npy"), dirs)
print("wrote", out)
