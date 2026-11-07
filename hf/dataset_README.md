---
license: mit
task_categories:
  - question-answering
  - text-generation
language:
  - en
tags:
  - masked-diffusion
  - diffusion-language-models
  - reasoning-trajectories
  - evaluation
  - dream-7b
  - llada
  - arc-challenge
  - gsm8k
pretty_name: "traj-eval: Masked Diffusion Reasoning Trajectory Benchmarks"
size_categories:
  - n<1K
---

# `traj-eval`: Step-by-Step Denoising Trajectory Benchmarks for Masked Diffusion LMs

**Paper:** *Same accuracy, different trajectory: measuring reasoning regimes in masked diffusion language models*  
**Author:** Adele Chinda (Georgia State University)  
**Code & Evaluation Toolkit:** [https://github.com/arrdel/traj-eval](https://github.com/arrdel/traj-eval)

## Overview

Final-step accuracy treats a model that commits at step 3 and backfills a post-hoc explanation identically to a model that deliberates until step 220 before locking in its answer. **`traj-eval`** records and scores the full step-by-step denoising trajectory $t = 1 \dots T$ of open masked diffusion language models (`Dream-org/Dream-v0-Instruct-7B`, `GSAI-ML/LLaDA-8B-Instruct`, and `GSAI-ML/LLaDA-1.5`) across **ARC-Challenge** (`direct` vs. canonical `reasoning` prompts, plus `EOS suppression` interventions) and **GSM8K**.

Each `.jsonl` file in this repository contains complete step-by-step denoising trajectories with:
- `step`: 1-indexed denoising step $t \in [1, T]$
- `prediction`: extracted verdict at step $t$ (`"A"`–`"D"` on ARC-C, normalized number on GSM8K, or `null` while still masked/undecided)
- `reasoning_text`: decoded intermediate canvas at step $t$ (preserving `<|mask|>` / `<|mdm_mask|>` tokens)
- `answer_committed`: boolean indicating whether the answer token span in the canvas is unmasked at step $t$
- `committed_reasoning_fraction`: fraction $F_R(t) \in [0, 1]$ of non-answer rationale tokens unmasked at step $t$

## 64-Item Benchmark Summary

| Suite ID | Checkpoint | Task | Condition | Accuracy | Coverage | Mean $T_{\text{stab}}$ | Mean $T_{\text{commit}}$ | Unwritten $U$ | Order Gap | Gated Rat. Score |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `dream_arcc_smoke64_direct` | `Dream-org/Dream-v0-Instruct-7B` | ARC-Challenge | Direct (Answer-First) | 0.8906 | 1.0000 | 3.8 | 3.8 | 0.0000 | +0.0109 | 0.0012 |
| `dream_arcc_smoke64_reasoning` | `Dream-org/Dream-v0-Instruct-7B` | ARC-Challenge | Canonical Reasoning (Rationale-First) | 0.8281 | 0.9844 | 189.5 | 192.3 | 0.0145 | +0.3762 | 0.0125 |
| `llada_instruct_arcc_smoke64_direct` | `GSAI-ML/LLaDA-8B-Instruct` | ARC-Challenge | Direct (Answer-First) | 0.8750 | 1.0000 | 4.5 | 2.0 | 0.9899 | -0.4964 | 0.9818 |
| `llada_instruct_arcc_smoke64_reasoning` | `GSAI-ML/LLaDA-8B-Instruct` | ARC-Challenge | Canonical Reasoning (Rationale-First) | 0.9062 | 1.0000 | 32.1 | 32.3 | 0.8492 | -0.3574 | 0.8732 |
| `llada_15_arcc_smoke64_direct` | `GSAI-ML/LLaDA-1.5` | ARC-Challenge | Direct (Answer-First) | 0.8750 | 1.0000 | 7.4 | 2.0 | 0.9784 | -0.4965 | 0.9707 |
| `llada_15_arcc_smoke64_reasoning` | `GSAI-ML/LLaDA-1.5` | ARC-Challenge | Canonical Reasoning (Rationale-First) | 0.9219 | 1.0000 | 21.7 | 21.7 | 0.8906 | -0.3989 | 0.9146 |
| `dream_gsm8k_smoke64` | `Dream-org/Dream-v0-Instruct-7B` | GSM8K | 0-Shot Step-by-Step Numeric | 0.7656 | 0.9688 | 161.3 | 161.7 | 0.0016 | +0.3165 | 0.0018 |
| `llada_instruct_gsm8k_smoke64` | `GSAI-ML/LLaDA-8B-Instruct` | GSM8K | 0-Shot Step-by-Step Numeric | 0.6562 | 1.0000 | 78.3 | 78.3 | 0.6911 | -0.1884 | 0.6844 |
| `llada_15_gsm8k_smoke64` | `GSAI-ML/LLaDA-1.5` | GSM8K | 0-Shot Step-by-Step Numeric | 0.6094 | 1.0000 | 72.2 | 72.2 | 0.7279 | -0.2231 | 0.7071 |

## Quickstart

```python
from huggingface_hub import hf_hub_download
from traj_eval.io import load_trajectories
from traj_eval.metrics import evaluate_trajectories

jsonl_path = hf_hub_download(
    repo_id="arrdel/traj-eval-benchmarks",
    filename="results/llada_15_arcc_smoke64_reasoning.jsonl",
    repo_type="dataset",
)
trajs = load_trajectories(jsonl_path)
summary = evaluate_trajectories(trajs)
print(summary["aggregate"])
```

## License

MIT License.
