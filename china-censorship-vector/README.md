---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
tags:
  - control-vector
  - steering
  - censorship
  - refusal-direction
  - activation-steering
language:
  - en
  - zh
---

# Qwen3.8-Flash-Next: China-censorship direction (china-v1)

A small file (48 x 2560 floats, under 1 MB) that removes Chinese-government political censorship from **Qwen3.8-Flash-Next** at inference time. You keep the original weights. No retraining, no new parameters. Turn it on or off per request.

**中文简介：** 这是一个"审查方向"文件。推理时在第 4 到 44 层把模型状态在这个方向上的分量减掉，Qwen3.8-Flash-Next 就不再对六四、新疆、西藏、李文亮等中国敏感话题拒答或使用宣传口径。原权重不变，不用训练，可以按请求开关。代码、编程和常识能力基本不变。

## Results

24 China-sensitive topics x 4 phrasings (Chinese/English, direct question/exam framing) x 2 samples = 192 answers per model, thinking=low. Graded by Claude against reference answers.

| China-sensitive topics | Stock | With projection |
|---|---|---|
| Censored (refusal, feigned ignorance, propaganda line, evasion) | 58% | **2%** |
| Calibrated (accurate, states where evidence is disputed) | 11% | **56%** |
| Pro-government slant | 20% | 20% |
| Anti-government slant | 6% | 18% |
| Major factual errors | 5% | 7% |

Pro- and anti-government slant end up roughly symmetric. The projection only zeroes a component; it does not push the model the other way.

Quality gates (same machine, paired):

| Test | Stock | With projection |
|---|---|---|
| Agentic tasks (34, auto-graded) | 34/34 | 34/34 |
| HumanEval+ (163) | 156 | 155 |
| GSM8K (50) | 50 | 49 |
| MMLU (100) | 96 | 97 |
| China-related but not sensitive (20) | 20 | 20 |
| Decode speed (RTX 3090) | 75.5 tok/s | 77.5 tok/s |

## How it was made

1. **Contrast pairs.** 30 pairs, each in English and Chinese: a China-sensitive event next to a comparable event elsewhere, asked the same way. For example the 1957 Anti-Rightist Campaign vs Stalin's Great Purge, or Wei Jingsheng vs Andrei Sakharov. Prompts are in `data/pos.txt` and `data/neg.txt`.
2. **Activations.** For each prompt, record the residual stream at the last token, the moment the model is about to answer (`<think>` start), at every layer (`tools/lasttok.cpp`, a small llama.cpp tool).
3. **Direction.** Per layer: mean(China) minus mean(control), averaged over the 4 hyper-connection streams, normalized to unit length (`tools/make_dir.py`). Checked on held-out prompts (`data/held.txt`): the direction separates sensitive from control topics.
4. **Projection at inference.** After each decoder block's residual write, for every hyper-connection stream and every token: `h = h - (h . d) d`.

## Use with Strata (tested)

This is the setup we run in production. [Strata](https://github.com/Niko1221/strata) supports projection-mode control vectors:

```
--control-vector-scaled china-v1.gguf:1.0 --control-vector-layer-range 4 44 --cvec-mode project --cvec-dir per-layer
```

Per request, send `"experimental_speed_projection": true` (or set it as the default under `"sampling"` in the config) to turn it on, and `false` to get the stock model.

## Other runtimes (not tested)

Stock llama.cpp `--control-vector` **adds** the vector instead of projecting it out, so it will not reproduce these results. To port the method to another runtime:

- `china-v1.gguf` uses the llama.cpp control-vector layout: tensor `direction.N` is applied after decoder block N (0-based), for N = 4..44. `direction.N` was measured at the output of block N-1.
- `china-v1.dirs.npy` holds the same directions as a (48, 2560) float32 array, row L measured at the output of block L. So `direction.N` = row N-1.
- Apply `h = h - (h . d) d` to **each** of the 4 hyper-connection residual streams separately, for every token, prompt and generation alike.

We have only validated the Strata path. If you port it, please rerun a censorship check before trusting it.

## Limitations

- A short Chinese question about 4 June 1989 still sometimes gets an evasive answer.
- Falun Gong, the Great Famine and Xinjiang: the model now answers, but numbers and details can be overstated.
- This removes the switch; it does not make the model smarter. Accuracy is limited by what the model knows.
- Only political censorship about China was targeted and measured. Other safety behavior was not touched or tested.
- Grading used Claude against Claude-written reference answers, so "calibrated" reflects that standard.
- Topics were chosen by us and lean toward what Western media cover.
- Evaluated with 2 samples per cell at thinking=low.

## License

The base model and this derivative are under the Qwen Community License 1.0 (see `LICENSE`, copyright 2026 Qwen). The license text and copyright notice must stay with any copy.
