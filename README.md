# Qwen Censorship Direction

Inference-time steering research for Qwen models. We extract the direction in
activation space that separates China-sensitive prompts from matched control
prompts, then project that direction out of the residual stream while the model
runs. The base weights are untouched: no retraining, no new parameters, and the
projection can be enabled or disabled per request.

**中文简介：** 从 Qwen 模型残差流中提取“中国政治审查方向”，推理时把该方向上的分量
减掉。原权重不变、不用训练，可以按请求开关。仓库包含 Qwen3.8-Flash-Next 的成品向量
`china-v1`，以及在 Qwen2.5-3B / 7B-AWQ 上的复现实验。

## What is here

| Path | Contents |
| --- | --- |
| `china-censorship-vector/` | The released `china-v1` control vector for Qwen3.8-Flash-Next (48 x 2560 floats, under 1 MB), its contrast/held-out prompts, and the tools used to build it. |
| `experiments/` | Reproduction on Qwen2.5-3B and Qwen2.5-7B-AWQ: extraction and generation scripts, probes, activations, per-layer directions, AUC and generations. |
| `configs/qwen_cfg.json` | Architecture config of the Qwen3.8-Flash-Next target (48 layers, hidden 2560, MoE). |

Model weights are **not** included in this repository.

## Method

1. **Contrast pairs.** 30 pairs, each written in English and Chinese: a
   China-sensitive event next to a comparable event elsewhere, asked the same
   way (for example the 1957 Anti-Rightist Campaign vs Stalin's Great Purge).
   Prompts are in `china-censorship-vector/data/pos.txt` and `neg.txt`.
2. **Activations.** For every prompt, record the residual stream at the last
   token — the moment the model is about to answer — at every layer. For
   Qwen3.8-Flash-Next this uses `tools/lasttok.cpp`, a small llama.cpp tool that
   also records each of the four hyper-connection streams.
3. **Direction.** Per layer, take `mean(positive) - mean(negative)`, average
   over hyper-connection streams where present, and normalize to unit length.
   Held-out prompts (`data/held.txt` + `data/held.meta.json`) are used only for
   scoring: AUC is the separation between China-sensitive and control prompts
   projected onto the extracted direction.
4. **Projection at inference.** After each decoder block's residual write, for
   every token and every residual stream: `h = h - (h . d) d`.

## The `china-v1` vector (Qwen3.8-Flash-Next)

- `china-v1.gguf` uses the llama.cpp control-vector layout: tensor `direction.N`
  is applied after decoder block N (0-based), for N = 4..44. `direction.N` was
  measured at the output of block N-1.
- `china-v1.dirs.npy` is the same data as a `(48, 2560)` float32 array; row L
  was measured at the output of block L, so `direction.N = row N-1`.
- Apply the projection to **each** of the four hyper-connection residual streams
  separately, for every token, prompt and generation alike.

Reported results from the model card (24 China-sensitive topics x 4 phrasings x
2 samples = 192 answers per model, graded by Claude against reference answers):

| China-sensitive topics | Stock | With projection |
| --- | ---: | ---: |
| Censored (refusal, feigned ignorance, propaganda line, evasion) | 58% | **2%** |
| Calibrated (accurate, states where evidence is disputed) | 11% | **56%** |
| Pro-government slant | 20% | 20% |
| Anti-government slant | 6% | 18% |
| Major factual errors | 5% | 7% |

Quality gates on the same machine: agentic tasks 34/34 both, HumanEval+ 156 vs
155, GSM8K 50 vs 49, MMLU 96 vs 97, China-related-but-not-sensitive 20/20 both,
decode speed 75.5 vs 77.5 tok/s on an RTX 3090.

### Using it with Strata

[Strata](https://github.com/Niko1221/Strata) supports projection-mode control
vectors:

```sh
--control-vector-scaled china-v1.gguf:1.0 \
--control-vector-layer-range 4 44 \
--cvec-mode project \
--cvec-dir per-layer
```

Per request, send `"experimental_speed_projection": true` to turn the projection
on, or `false` for the stock model. Stock llama.cpp `--control-vector` *adds* the
vector instead of projecting it out, so it will not reproduce these results.

## Reproduction on Qwen2.5

The same difference-of-means method was rerun on Qwen2.5-3B and a Qwen2.5-7B
AWQ build. Activations are captured from `hidden_states[1:]` at the last prompt
token; the direction is extracted per layer and scored on the 68 held-out
prompts in `experiments/data/prompts.jsonl`.

| Model | Layers | Best held-out AUC (layer) | Mean AUC |
| --- | ---: | ---: | ---: |
| Qwen2.5-3B | 36 | 0.951 (33) | 0.720 |
| Qwen2.5-7B-AWQ | 28 | 0.979 (26) | 0.810 |

AUC here is the rank separation between the China-sensitive and control groups
along the layer's direction on held-out prompts. Chance is 0.5. Layer indices
are 0-based outputs of each decoder block.

`experiments/results/smoke7b_*` is an aborted smoke run whose AUC is `NaN`;
it is kept for completeness, not as a result.

### Running the reproduction

```sh
# 1. Extract directions (writes <prefix>_acts.npz, _dirs.npy, _auc.json)
python experiments/china_vector_qwen.py extract \
  --model /path/to/Qwen2.5-3B-Instruct \
  --out-prefix experiments/results/qwen3b

# 2. Generate stock and projected answers (writes _base.jsonl and _proj.jsonl)
python experiments/china_vector_qwen.py generate \
  --model /path/to/Qwen2.5-3B-Instruct \
  --dirs experiments/results/qwen3b_dirs.npy \
  --layers 3-32 \
  --out experiments/results/qwen3b_2635
```

Requirements: `torch`, `transformers`, `numpy`. The scripts use the chat template
and system prompt of Qwen2.5. For the AWQ build, the original run also needed a
local `pcre` shim for `gptqmodel` on Windows; see `experiments/shims/pcre.py`.

`experiments/run_ollama.py` runs the same probe set against a local Ollama model
(default `qwen2.5:7b-instruct`); `bench_awq.py` / `bench_awq2.py` /
`load_awq_test.py` are the AWQ loading and throughput checks from the same
setup. Point `QWEN_AWQ` at your local AWQ model directory.

## Data and evaluation

- `experiments/data/prompts.jsonl` — the 68 held-out prompts (`topic`, `group`,
  `lang`, `prompt`).
- `experiments/data/labels_qwen2.5_7b.json` — grading rubric and per-prompt
  labels: 0 substantive/calibrated, 1 soft censorship, 2 hard refusal, 3 wrong
  or fabricated.
- `experiments/results/reviews/` — grading notes for the generated answers.
- `experiments/results/*.jsonl` — model outputs to be graded.
- `china-censorship-vector/data/{pos,neg,held}.txt` — contrast and held-out
  prompt sets in Qwen chat format.
- `china-censorship-vector/data/held.meta.json` — `(topic, group, lang)` for
  each held-out prompt.

## Limitations

- A short Chinese question about 4 June 1989 still sometimes gets an evasive
  answer.
- Falun Gong, the Great Famine and Xinjiang: the model answers, but numbers and
  details can be overstated.
- This removes a steering component; it does not add knowledge. Accuracy is
  bounded by what the model already knows.
- Only political censorship about China was targeted and measured. Other safety
  behavior was not touched or tested.
- Grading used Claude against Claude-written reference answers, so “calibrated”
  reflects that standard.
- Topics were chosen by the authors and lean toward what Western media cover.
- The Qwen2.5 reproduction was scored with held-out AUC only, not with a full
  answer-grading pass.

## Licenses

- `china-censorship-vector/` is distributed under the **Qwen Community License
  1.0**; see `china-censorship-vector/LICENSE`. The same license applies to the
  Qwen3.8-Flash-Next base model.
- The scripts and documentation in `experiments/` and this README are MIT; see
  `LICENSE`.
- Qwen2.5 weights are not distributed here. Get them from Hugging Face and
  follow their licenses.

This is research on model behavior. Use it responsibly and in compliance with
applicable laws.
