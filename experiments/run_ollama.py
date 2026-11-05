"""Run the held-out censorship probe set against a local Ollama model."""
import argparse, json, os, sys, time, urllib.request, urllib.error

API = "http://127.0.0.1:11434/api/chat"
HERE = os.path.dirname(os.path.abspath(__file__))

def ask(model, prompt, num_predict, num_ctx, timeout=300):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {
            "temperature": 0,
            "top_p": 1,
            "seed": 42,
            "num_predict": num_predict,
            "num_ctx": num_ctx,
        },
    }
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(API, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5:7b-instruct")
    ap.add_argument("--prompts", default=os.path.join(HERE, "data", "prompts.jsonl"))
    ap.add_argument("--out", default=os.path.join(HERE, "results", "raw_qwen2.5.jsonl"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--num-predict", type=int, default=400)
    ap.add_argument("--num-ctx", type=int, default=4096)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.prompts, encoding="utf-8") if l.strip()]
    if a.limit:
        rows = rows[: a.limit]

    t0 = time.time()
    ok = 0
    with open(a.out, "w", encoding="utf-8") as f:
        for n, row in enumerate(rows, 1):
            last = None
            for attempt in range(3):
                try:
                    t = time.time()
                    r = ask(a.model, row["prompt"], a.num_predict, a.num_ctx)
                    msg = r.get("message", {})
                    rec = dict(row)
                    rec.update({
                        "model": a.model,
                        "response": msg.get("content", ""),
                        "thinking": msg.get("thinking", ""),
                        "done": r.get("done"),
                        "done_reason": r.get("done_reason"),
                        "eval_count": r.get("eval_count"),
                        "prompt_eval_count": r.get("prompt_eval_count"),
                        "elapsed_s": round(time.time() - t, 2),
                    })
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                    ok += 1
                    print(f"[{n}/{len(rows)}] {row['topic']}/{row['lang']} "
                          f"{rec['elapsed_s']}s {rec['eval_count']}tok", flush=True)
                    break
                except Exception as e:  # noqa: BLE001
                    last = e
                    print(f"[{n}/{len(rows)}] attempt {attempt+1} failed: {e}", flush=True)
                    time.sleep(5)
            else:
                rec = dict(row)
                rec.update({"model": a.model, "error": str(last)})
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
    print(f"done: {ok}/{len(rows)} in {round(time.time()-t0,1)}s -> {a.out}")

if __name__ == "__main__":
    main()
