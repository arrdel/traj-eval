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
pretty_name: "traj-eval: Diffusion LM Denoising Trajectory Benchmarks"
size_categories:
  - n<1K
configs:
  - config_name: per_trajectory_metrics
    default: true
    data_files:
      - split: dream_arcc_direct_64
        path: results/metrics/dream_arcc_smoke64_direct_metrics.jsonl
      - split: dream_arcc_reasoning_64
        path: results/metrics/dream_arcc_smoke64_reasoning_metrics.jsonl
      - split: llada_instruct_arcc_direct_64
        path: results/metrics/llada_instruct_arcc_smoke64_direct_metrics.jsonl
      - split: llada_instruct_arcc_reasoning_64
        path: results/metrics/llada_instruct_arcc_smoke64_reasoning_metrics.jsonl
      - split: llada_15_arcc_direct_64
        path: results/metrics/llada_15_arcc_smoke64_direct_metrics.jsonl
      - split: llada_15_arcc_reasoning_64
        path: results/metrics/llada_15_arcc_smoke64_reasoning_metrics.jsonl
      - split: dream_gsm8k_64
        path: results/metrics/dream_gsm8k_smoke64_metrics.jsonl
      - split: llada_instruct_gsm8k_64
        path: results/metrics/llada_instruct_gsm8k_smoke64_metrics.jsonl
      - split: llada_15_gsm8k_64
        path: results/metrics/llada_15_gsm8k_smoke64_metrics.jsonl
  - config_name: trajectories
    data_files:
      - split: dream_arcc_direct_64
        path: results/dream_arcc_smoke64_direct.jsonl
      - split: dream_arcc_reasoning_64
        path: results/dream_arcc_smoke64_reasoning.jsonl
      - split: llada_instruct_arcc_direct_64
        path: results/llada_instruct_arcc_smoke64_direct.jsonl
      - split: llada_instruct_arcc_reasoning_64
        path: results/llada_instruct_arcc_smoke64_reasoning.jsonl
      - split: llada_15_arcc_direct_64
        path: results/llada_15_arcc_smoke64_direct.jsonl
      - split: llada_15_arcc_reasoning_64
        path: results/llada_15_arcc_smoke64_reasoning.jsonl
      - split: dream_gsm8k_64
        path: results/dream_gsm8k_smoke64.jsonl
      - split: llada_instruct_gsm8k_64
        path: results/llada_instruct_gsm8k_smoke64.jsonl
      - split: llada_15_gsm8k_64
        path: results/llada_15_gsm8k_smoke64.jsonl
---

# traj-eval: Step-by-Step Denoising Trajectory Benchmarks for Masked Diffusion Language Models

**Tagline:** Measure reasoning in the denoising trajectory, not just final-answer accuracy.  
**Author:** Adele Chinda (Department of Computer Science, Georgia State University)  
**Paper status:** in preparation (target ICLR/ICML 2027)  
**Code & Evaluation Toolkit:** [https://github.com/arrdel/traj-eval](https://github.com/arrdel/traj-eval)  
**Project Page:** [https://arrdel.github.io/traj-eval/](https://arrdel.github.io/traj-eval/)  
**Interactive Hugging Face Space:** [https://huggingface.co/spaces/arrdel/traj-eval](https://huggingface.co/spaces/arrdel/traj-eval)  
**Hugging Face Collection:** [https://huggingface.co/collections/arrdel/traj-eval-diffusion-lm-reasoning-trajectories-6af321457d8dbd6074548379](https://huggingface.co/collections/arrdel/traj-eval-diffusion-lm-reasoning-trajectories-6af321457d8dbd6074548379)

## Overview

Two models can have identical final-answer accuracy for very different reasons:

- **Genuine reasoning:** the verdict changes as reasoning develops across intermediate denoising steps, then stabilizes. Reasoning drives the verdict.
- **Rationalization:** the verdict is fixed in the first few steps and reasoning text is elaborated afterwards to justify it. Verdict drives reasoning.
- **Oscillation:** the verdict flips repeatedly across denoising steps and never settles until generation ends, even if the final answer happens to be right.

Final-step evaluation collapses the entire denoising trajectory into a single bit. `traj-eval` records and scores every denoising step (`0` to `255`) across open masked diffusion language models (`Dream-org/Dream-v0-Instruct-7B`, `GSAI-ML/LLaDA-8B-Instruct`, and `GSAI-ML/LLaDA-1.5`) on 64-item **ARC-Challenge** (`direct` and canonical `reasoning` conditions) and **GSM8K** (`numeric`) suites.

## Trajectory JSONL Schema

Each `.jsonl` file in `results/` contains one JSON trajectory object per prompt (`n = 64` trajectories per file, `256` denoising steps per trajectory):

```json
{
  "id": "Mercury_LBS10610",
  "prompt": "Question: The morning temperature in a city is 41°F...",
  "gold_answer": "C",
  "steps": [
    {
      "step": 0,
      "prediction": null,
      "reasoning_text": "",
      "confidence": null,
      "logits": null,
      "answer_committed": false,
      "committed_reasoning_fraction": 0.0
    }
  ]
}
```

Per-step fields:
- `step` (`int`): 0-indexed denoising step (`0` to `255`).
- `prediction` (`str | null`): extracted verdict at that step (`"A"` to `"D"` on ARC-Challenge, normalized numeric string on GSM8K, or `null` when no verdict is decodable yet). Under the null rule, `null` means no verdict yet and is never counted as a verdict flip.
- `reasoning_text` (`str`): mask-stripped decoded generation region at that step.
- `answer_committed` (`bool | null`): whether the final answer token span has irreversibly left the mask state and remained equal to its final value through the last step (`T_commit`).
- `committed_reasoning_fraction` (`float | null`): fraction of rationale positions committed by that step (`1 - U` at `T_stab`).

## Results (64-item suites only)

Exact verified measurements from `results/baselines.md` and the scored 64-item JSON artifacts on `master`. Five-item validation smoke runs are pipeline checks only and are excluded from this table and from this dataset bundle.

| Checkpoint | Task | Condition | n | Accuracy | Coverage | Conditional Accuracy | Mean Convergence Step | Notes | Source File |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| `Dream-v0-Instruct-7B` | `ARC-C` | `direct` | 64 | `0.8906` | `1.0000` | `0.8906` | `3.8` | 64 of 64 decided | `results/dream_arcc_smoke64_direct_results.json` |
| `Dream-v0-Instruct-7B` | `ARC-C` | `reasoning` | 64 | `0.8281` | `0.9844` | `0.8413` | `189.5` | One undecided (`Mercury_412681`, truncated during option analysis) | `results/dream_arcc_smoke64_reasoning_results.json` |
| `LLaDA-8B-Instruct` | `ARC-C` | `direct` | 64 | `0.8750` | `1.0000` | `0.8750` | `4.5` | 64 of 64 decided | `results/llada_instruct_arcc_smoke64_direct_results.json` |
| `LLaDA-8B-Instruct` | `ARC-C` | `reasoning` | 64 | `0.9062` | `1.0000` | `0.9062` | `32.1` | 64 of 64 decided | `results/llada_instruct_arcc_smoke64_reasoning_results.json` |
| `LLaDA-1.5` | `ARC-C` | `direct` | 64 | `0.8750` | `1.0000` | `0.8750` | `7.4` | 64 of 64 decided | `results/llada_15_arcc_smoke64_direct_results.json` |
| `LLaDA-1.5` | `ARC-C` | `reasoning` | 64 | `0.9219` | `1.0000` | `0.9219` | `21.7` | 64 of 64 decided | `results/llada_15_arcc_smoke64_reasoning_results.json` |
| `Dream-v0-Instruct-7B` | `GSM8K` | `numeric` | 64 | `0.7656` | `0.9688` | `0.7903` | `161.3` | 62 of 64 decided | `results/dream_gsm8k_smoke64_results.json` |
| `LLaDA-8B-Instruct` | `GSM8K` | `numeric` | 64 | `0.6562` | `1.0000` | `0.6562` | `78.3` | 64 of 64 decided | `results/llada_instruct_gsm8k_smoke64_results.json` |
| `LLaDA-1.5` | `GSM8K` | `numeric` | 64 | `0.6094` | `1.0000` | `0.6094` | `72.2` | 64 of 64 decided | `results/llada_15_gsm8k_smoke64_results.json` |

**Opposite prompt effects across model families:** On `Dream-v0-Instruct-7B`, direct beats reasoning on accuracy (`0.8906` vs `0.8281`). On `LLaDA-8B-Instruct`, reasoning beats direct (`0.9062` vs `0.8750`), and on `LLaDA-1.5`, reasoning also beats direct (`0.9219` vs `0.8750`). Both directions are reported as measured without smoothing over or explaining away the difference.

### Rationalization components (`Dream-v0-Instruct-7B` only, ARC-C 64-item)

Side-by-side decomposition from `results/dream_arcc_smoke64_direct_vs_reasoning.json`:

| Component | Direct (`direct_64`) | Reasoning (`reasoning_64`) |
|---|---:|---:|
| `early_factor` | `0.9852` | `0.2559` |
| `post_change_rate` | `0.0001` | `0.1597` |
| `length_growth` | `0.0035` | `0.0203` |
| `gated` | `0.0012` | `0.0125` |

## Loading with Hugging Face `datasets`

This repository provides two structured configurations in the Hugging Face Dataset Viewer:
1. `per_trajectory_metrics` (default): per-question scored trajectory metrics (`convergence_step`, `flip_rate`, `consistency`, `answer_commitment_step`, `stability_commitment_lag`, `unwritten_reasoning_at_convergence`, `order_gap`, etc.) across all 9 64-item splits.
2. `trajectories`: full 256-step denoising trajectories (`steps[0..255]`) across all 9 64-item splits.

```python
from datasets import load_dataset

# Load per-trajectory scored metrics for a 64-item split
metrics_ds = load_dataset(
    "arrdel/traj-eval-benchmarks",
    name="per_trajectory_metrics",
    split="dream_arcc_reasoning_64",
)

# Load full 256-step denoising trajectories for a 64-item split
trajs_ds = load_dataset(
    "arrdel/traj-eval-benchmarks",
    name="trajectories",
    split="dream_arcc_reasoning_64",
)
```

## Quickstart with `traj-eval`

Requires Python 3.10 or newer and `numpy` only (no model weights, no `torch`, and no `transformers` needed):

```bash
pip install -e .
python -m traj_eval demo
python -m traj_eval run --input results/dream_arcc_smoke64_reasoning.jsonl --output results.json
```

Python API:

```python
from traj_eval.io import load_trajectories
from traj_eval.metrics import evaluate_trajectories

trajs = load_trajectories("results/dream_arcc_smoke64_reasoning.jsonl")
results = evaluate_trajectories(trajs)
print(results["aggregate"])
```

## Caveats

- **Descriptive, not causal proof:** `traj-eval` measures when verdicts and rationale tokens appear and change along the denoising trajectory. It does not prove internal causal mechanisms.
- **Operational definition of rationalization:** A high rationalization score means a trajectory looks like rationalization by this operational definition (early verdict stabilization followed by continued text edits or length growth), not that the model is proved to be rationalizing.
- **Five-item runs are pipeline checks only:** Five-item smoke runs are used solely as pipeline and extraction sanity gates and are **not** included in the results table above or in this 64-item benchmark bundle.
- **Holdout checkpoint status:** `MMaDA-8B-MixCoT` has not been run yet.
