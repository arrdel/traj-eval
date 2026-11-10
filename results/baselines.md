# Baselines

Numbers in this file are either computed in this repo or explicitly not run. Do not fill a cell from another paper and label it a traj-eval result.

## Synthetic v0.1 — measured

Command: `python -m traj_eval demo --n-per-archetype 5` (seed 42). Scored by traj-eval 0.1.0. Data: `examples/synthetic_demo.jsonl`. Machine-readable copy: `results/synthetic_v0.1.json`.

Every archetype is constructed so the final answer is correct. Accuracy cannot tell them apart.

| Archetype | n | Accuracy | Mean flip rate | Mean rationalization | Early convergence | Mean convergence step | Mean convergence fraction |
|---|---:|---:|---:|---:|---:|---:|---:|
| genuine | 5 | 1.0000 | 0.2857 | 0.3333 | 0.0000 | 4.00 | 0.5000 |
| rationalization | 5 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.00 | 0.0000 |
| oscillation | 5 | 1.0000 | 1.0000 | 0.0375 | 0.0000 | 7.00 | 0.8750 |
| all | 15 | 1.0000 | 0.4286 | 0.4569 | 0.3333 | 3.67 | 0.4583 |

`rationalization_score` is a heuristic. These rows are a unit test of that heuristic, not a model evaluation.

Forced deliberation is not defined on this file. The function only subtracts two accuracies the caller already has.

## Real checkpoints — status overview

Chosen 17 Oct 2026. Text trio first (`LLaDA-8B-Instruct`, `LLaDA-1.5`, and `Dream-v0-Instruct-7B`), so commitment-order and trajectory measurements are directly comparable across open diffusion LMs. `MMaDA-8B-MixCoT` is the later modality holdout, not a substitute for `LLaDA-1.5` in the first table.

| Checkpoint | Hugging Face id | Status |
|---|---|---|
| LLaDA-8B-Instruct | `GSAI-ML/LLaDA-8B-Instruct` | All 5-item (ARC-C direct, ARC-C reasoning, GSM8K) and 64-item (ARC-C direct, ARC-C canonical reasoning, GSM8K) suites completed (2026-11-01). |
| LLaDA-1.5 | `GSAI-ML/LLaDA-1.5` | All 5-item (ARC-C direct, ARC-C reasoning, GSM8K) and 64-item (ARC-C direct, ARC-C canonical reasoning, GSM8K) suites completed (2026-11-01). |
| Dream-v0-Instruct-7B | `Dream-org/Dream-v0-Instruct-7B` | 5-item ARC-C direct, reasoning, EOS-suppression K=64, & GSM8K smokes clean + 64-item ARC-C paired (direct & canonical reasoning) and 64-item GSM8K suites completed (2026-10-30 / 2026-11-01). |
| MMaDA-8B-MixCoT (holdout) | `Gen-Verse/MMaDA-8B-MixCoT` | Not in the first suite. |

### 64-Item Benchmark Summary (Main Suites)

Five-item runs below are pipeline checks only and are excluded from this summary table.

| Checkpoint | Task | Condition | n | Accuracy | Coverage | Conditional Accuracy | Mean Convergence Step | Notes | Scored JSON |
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

#### Rationalization components (`Dream-v0-Instruct-7B` only, ARC-C 64-item)

Side-by-side decomposition from `results/dream_arcc_smoke64_direct_vs_reasoning.json`:

| Component | Direct (`direct_64`) | Reasoning (`reasoning_64`) |
|---|---:|---:|
| `early_factor` | `0.9852` | `0.2559` |
| `post_change_rate` | `0.0001` | `0.1597` |
| `length_growth` | `0.0035` | `0.0203` |
| `gated` | `0.0012` | `0.0125` |

Historical note (17 Oct 2026): no checkpoints were run on initial commit because all eight RTX A6000s were occupied (~43 GB of 49 GB, 100% utilization) and the base environment PyTorch (`2.11.0+cu130`) could not see the CUDA 12.1 driver. Subsequent runs below use a pinned `transformers==4.46.2` CUDA environment (`cdr-gpu` on `titan3` and `.venv-gpu` on `ai-lab2`) with one dedicated GPU per run and full run logs.

### Smoke protocol, to run unchanged once a GPU is free

Historical Oct 17 GSM8K protocol note (specifies `temperature=0` + EOS/EoT confidence suppression as used by the LLaDA family; `Dream-v0-Instruct-7B` GSM8K runs below use Dream's standard unrestricted sampler `temperature=0.2, top_p=0.95, alg=entropy, force_deliberation_steps=0` registered in `scripts/gpu_job_queue.py`). It is not the ARC-C smoke protocol below.

Pre-register before looking at accuracy:

- Data: GSM8K test, first 64 questions in the published test order. Store the id list in `results/smoke64_ids.json` when the run starts.
- Canvas length 256, denoising steps 256, temperature 0.
- Remasking: the checkpoint's released low-confidence rule.
- EOS/EOT confidence suppression: on. Report it. Turning it off is a different experiment.
- Log every step to the traj-eval JSONL schema: parsed prediction, reasoning text, and, once the adapter can, whether the answer span is irreversibly committed.
- Score with `python -m traj_eval run`. Paste the aggregate here with the git revision and the exact sampler flags.

Forced-deliberation smoke, only after the unrestricted smoke JSONL exists: continue for 64 extra steps after the parsed verdict has been stable for 16 steps, answer span still remaskable. Delta is `forced_deliberation_delta` on the paired 64.

No accuracy number from Yeom et al., d1, or a model card belongs in this table.

## Dream-v0-Instruct-7B — ARC-C 5-item smoke (smoke clean)

Status: **smoke clean** (pipeline check only; 5 items are not a model evaluation claim).

- **Date:** 2026-10-29
- **Git revision (`git rev-parse HEAD` at run):** `a6c7f740835f680dc6759a4058f17d5d9ce9f7ac`
- **Environment:** conda env `cdr-gpu` (`Python 3.11.17`, `torch 2.5.1+cu121`, `transformers 4.46.2`)
- **GPU:** index `4` (`CUDA_VISIBLE_DEVICES=4`, `device="cuda:0"`, `NVIDIA RTX A6000`)
- **`nvidia-smi` at launch:**
  ```
  index, name, memory.used [MiB], memory.total [MiB], utilization.gpu [%]
  0, NVIDIA RTX A6000, 32891 MiB, 49140 MiB, 99 %
  1, NVIDIA RTX A6000, 32965 MiB, 49140 MiB, 100 %
  2, NVIDIA RTX A6000, 32965 MiB, 49140 MiB, 100 %
  3, NVIDIA RTX A6000, 32965 MiB, 49140 MiB, 100 %
  4, NVIDIA RTX A6000, 16899 MiB, 49140 MiB, 0 %
  5, NVIDIA RTX A6000, 13259 MiB, 49140 MiB, 100 %
  6, NVIDIA RTX A6000, 30225 MiB, 49140 MiB, 98 %
  7, NVIDIA RTX A6000, 36073 MiB, 49140 MiB, 100 %
  ```
- **Sampler config:**
  - `model_id`: `"Dream-org/Dream-v0-Instruct-7B"`
  - `temperature`: `0.2`
  - `top_p`: `0.95`
  - `alg`: `"entropy"`
  - `alg_temp`: `0.0`
  - `steps`: `256`
  - `max_new_tokens`: `256`
- **Artifacts:**
  - Input: `results/dream_arcc_smoke5_input.json`
  - Fixed IDs: `results/dream_arcc_smoke5_ids.json`
  - Trajectories: `results/dream_arcc_smoke5.jsonl`
  - Run log: `results/dream_arcc_smoke5_run_log.json`
  - Scored output: `results/dream_arcc_smoke5_results.json`
- **Sanity & undecided count:**
  - `len(history) == 256` for all 5 items.
  - Final parsed verdict matched `sequences` verdict for all 5 items.
  - Non-null at final step: **5 / 5** (**0 undecided**).

### Pasted `traj-eval run` output

```
=== traj-eval summary ===
Trajectories:            5
Accuracy (final):        1.0000
Mean flip rate:          0.0000
Mean rationalization:    0.3950
Early convergence frac:  1.0000
Mean convergence step:   3.20
Median convergence step: 3.00
Mean convergence frac:   0.0125

Per-trajectory:
id                   final    gold   corr  conv  flip   rat    steps
--------------------------------------------------------------------
Mercury_7175875      C        C      Y     3     0.000  0.395  256  
Mercury_SC_409171    B        B      Y     3     0.000  0.395  256  
Mercury_SC_408547    C        C      Y     3     0.000  0.395  256  
Mercury_407327       D        D      Y     3     0.000  0.395  256  
MCAS_2006_9_44       D        D      Y     4     0.000  0.394  256  
```

### Pasted aggregate JSON (`results/dream_arcc_smoke5_results.json`)

*(Note: re-scored from the Oct 29 `7831ea7` JSONL to include the decomposed rationalization components, `mean_gated_rationalization_score`, and coverage fields; original accuracy, flip rate, and convergence numbers are unchanged, and `results/dream_arcc_smoke5_run_log.json` was not modified.)*

```json
{
  "n_trajectories": 5,
  "n_scored_for_accuracy": 5,
  "accuracy": 1.0,
  "coverage": 1.0,
  "conditional_accuracy": 1.0,
  "mean_flip_rate": 0.0,
  "mean_rationalization_score": 0.395,
  "mean_gated_rationalization_score": 0.0,
  "mean_early_factor": 0.9875,
  "mean_post_change_rate": 0.0,
  "mean_length_growth": 0.0,
  "early_convergence_fraction": 1.0,
  "disagreement_rate": 0.0,
  "sbw_lockin_rate": 0.0,
  "mean_convergence_step": 3.2,
  "median_convergence_step": 3.0,
  "mean_convergence_fraction": 0.0125,
  "median_convergence_fraction": 0.0117,
  "mean_answer_commitment_step": null,
  "mean_stability_commitment_lag": null,
  "mean_unwritten_reasoning_at_convergence": null,
  "mean_order_gap": null
}
```

### Diagnostic note on the 5-item trace

Under the direct-answer prompt template in `results/dream_arcc_smoke5_input.json` (`Respond in the form "Answer: <letter>" with the letter of the correct option.`), Dream unmasks `'Answer'` -> `'Answer:'` -> `'Answer: <letter>'` in steps 1–4 and fills the remaining ~252 canvas positions with EOS tokens (stripped by `skip_special_tokens=True`). Because no post-convergence rationale text is written (`mean_post_change_rate = 0.0`, `mean_length_growth = 0.0`, `mean_gated_rationalization_score = 0.0`), the additive heuristic `rationalization_score` equals `0.4 * early_factor` (~0.395) purely from early convergence relative to the 256-step canvas.

## Prompt Condition Decision & File Mapping (Issue #4 Reconciliation)

Per Issue #4, ARC-C evaluation uses **Option (a)**: the full ARC-C reasoning condition uses the **canonical reasoning prompt** at both 5-item and 64-item scale, and the earlier 2–3-sentence `cot` prompt files (`results/dream_arcc_cot_smoke5_input.json`, `results/dream_arcc_cot_smoke5_ids.json`, `results/arcc_smoke64_cot_input.json`) are **retired and removed** so prompt wording is never swapped between 5-item and 64-item runs. Direct and reasoning conditions are always reported side by side, never pooled into a single accuracy number.

| Condition | Exact closing prompt instruction | 5-item files (`Mercury_7175875` .. `MCAS_2006_9_44`) | 64-item files (first 64 4-option A–D test items) |
|---|---|---|---|
| `direct` | `Respond in the form "Answer: <letter>" with the letter of the correct option.` | `results/dream_arcc_smoke5_input.json`, `results/dream_arcc_smoke5_ids.json`, `results/dream_arcc_smoke5.jsonl`, `results/dream_arcc_smoke5_run_log.json`, `results/dream_arcc_smoke5_results.json` | `results/arcc_smoke64_direct_input.json`, `results/arcc_smoke64_ids.json` |
| `reasoning` (canonical) | `Think step by step, then give your final answer in the form "Answer: <letter>".` | `results/dream_arcc_smoke5_reasoning_input.json`, `results/dream_arcc_smoke5_reasoning_ids.json`, `results/dream_arcc_smoke5_reasoning.jsonl`, `results/dream_arcc_smoke5_reasoning_run_log.json`, `results/dream_arcc_smoke5_reasoning_results.json` | `results/arcc_smoke64_reasoning_input.json`, `results/arcc_smoke64_ids.json` |

## Pre-registered Gated Sequence

This list is a **gated pre-registration sequence**, executed one stage at a time by `scripts/gpu_job_queue.py` whenever a GPU with `>= 25 GB` free VRAM is available. Each stage opens its own PR against `master` and waits for review/merge before the next stage starts.

1. **Stage 1 — Dream-7B canonical reasoning-condition 5-item ARC-C smoke (COMPLETED, gate passed — see section below):**
   - Canonical prompt file: `results/dream_arcc_smoke5_reasoning_input.json`, `results/dream_arcc_smoke5_reasoning_ids.json` (closing line: `Think step by step, then give your final answer in the form "Answer: <letter>".`).
2. **Stage 2 — Dream-7B 64-item ARC-C paired direct + canonical reasoning suites:**
   - Pre-registered IDs: `results/arcc_smoke64_ids.json`
   - Direct input: `results/arcc_smoke64_direct_input.json`
   - Canonical reasoning input: `results/arcc_smoke64_reasoning_input.json` (exact same closing line as Stage 1: `Think step by step, then give your final answer in the form "Answer: <letter>".`).
3. **Stage 3 — Dream-7B GSM8K 5-item numeric smoke (`extract_gsm8k_verdict` + exemplar stripping):**
   - Inputs & IDs: `results/dream_gsm8k_smoke5_input.json`, `results/dream_gsm8k_smoke5_ids.json`
4. **Stage 4 — Dream-7B EOS-suppression K=64 5-item smoke (`--force-deliberation-steps 64`, named `EOS-suppression K=64`, separate from the post-stability protocol):**
   - Inputs & IDs: `results/dream_arcc_smoke5_input.json`, `results/dream_arcc_smoke5_ids.json`
   - Outputs: `results/dream_arcc_eos_suppression_k64_smoke5.{jsonl,run_log.json,results.json}`
5. **Stage 5 — LLaDA-8B-Instruct and LLaDA-1.5 5-item ARC-C validation smokes (direct + canonical reasoning):**
   - Run with `--model-family llada-instruct|llada-1.5` before any 64-item LLaDA runs.
6. **Stage 6 — 64-item ARC-C and 5-item / 64-item GSM8K suites across Dream-7B, LLaDA-8B-Instruct, and LLaDA-1.5 (registered in `scripts/gpu_job_queue.py`):**
   - ARC-C 64-item direct & canonical reasoning (`results/arcc_smoke64_ids.json`, `results/arcc_smoke64_{direct,reasoning}_input.json`).
   - GSM8K 5-item validation smokes (`results/llada_instruct_gsm8k_smoke5_ids.json`, `results/llada_15_gsm8k_smoke5_ids.json`, `results/dream_gsm8k_smoke5_input.json`) and 64-item numeric suites (`results/smoke64_ids.json`, `results/gsm8k_smoke64_input.json`).

## Dream-v0-Instruct-7B — ARC-C 5-item reasoning-condition smoke (reasoning smoke clean)

- **Date (UTC):** 2026-10-30
- **Job ID:** `dream_arcc_smoke5_reasoning`
- **Git revision (`git rev-parse HEAD` at run):** `267dfe3f90f693b40e5a41775553bb23febd7cf3` (includes the word-boundary fix in `_ANSWER_PREFIX_RES` so `"Provide the final answer\nAnswer: C"` on `Mercury_7175875` extracts `"C"` rather than the `"A"` in `"Answer"`)
- **Environment:** conda env `cdr-gpu` (`torch 2.5.1+cu121`, `transformers 4.46.2`)
- **GPU:** index `5` (`CUDA_VISIBLE_DEVICES=5`, `NVIDIA RTX A6000`, `48673 MiB` free / `3 MiB` used / `0%` util at launch)
- **`nvidia-smi` at launch:**
  ```
  index, name, memory.used [MiB], memory.total [MiB], utilization.gpu [%]
  0, NVIDIA RTX A6000, 32891 MiB, 49140 MiB, 100 %
  1, NVIDIA RTX A6000, 32973 MiB, 49140 MiB, 100 %
  2, NVIDIA RTX A6000, 32973 MiB, 49140 MiB, 99 %
  3, NVIDIA RTX A6000, 32973 MiB, 49140 MiB, 100 %
  4, NVIDIA RTX A6000, 3 MiB, 49140 MiB, 0 %
  5, NVIDIA RTX A6000, 3 MiB, 49140 MiB, 0 %
  6, NVIDIA RTX A6000, 3 MiB, 49140 MiB, 0 %
  7, NVIDIA RTX A6000, 3 MiB, 49140 MiB, 0 %
  ```
- **Sampler & adapter config:** `model_family=dream, temperature=0.2, top_p=0.95, alg=entropy, alg_temp=0.0, steps=256, max_new_tokens=256, force_deliberation_steps=0, frontier_width=None`
- **Artifacts:**
  - Input: `results/dream_arcc_smoke5_reasoning_input.json`
  - Fixed IDs: `results/dream_arcc_smoke5_reasoning_ids.json`
  - Trajectories: `results/dream_arcc_smoke5_reasoning.jsonl`
  - Run log: `results/dream_arcc_smoke5_reasoning_run_log.json`
  - Scored output: `results/dream_arcc_smoke5_reasoning_results.json`
  - Side-by-side comparison JSON: `results/dream_arcc_smoke5_direct_vs_reasoning.json`
- **Undecided count:** **0 undecided** (5 / 5 non-null at final step).
- **Reasoning smoke gate:** `reasoning smoke clean` (commitment clocks on `5/5`, post-verdict change on `3`, pre-verdict rationale on `5`).

### Pasted `traj-eval run` output

```
=== traj-eval summary ===
Trajectories:            5
Accuracy (final):        0.8000
Mean flip rate:          0.0000
Mean rationalization:    0.2068
  [components] early=0.2445 post_change=0.2574 len_growth=0.0299 gated=0.0166
Early convergence frac:  0.0000
Mean convergence step:   193.40
Median convergence step: 189.00
Mean convergence frac:   0.7555
Mean T_commit step:      196.00
Mean lag (T_comm-T_stab):2.60
Mean unwritten @ T_stab: 0.0137

Per-trajectory:
id                   final    gold   corr  conv  flip   rat    steps
--------------------------------------------------------------------
Mercury_7175875      C        C      Y     224   0.000  0.050  256  
Mercury_SC_409171    B        B      Y     165   0.000  0.142  256  
Mercury_SC_408547    C        C      Y     254   0.000  0.404  256  
Mercury_407327       D        D      Y     135   0.000  0.232  256  
MCAS_2006_9_44       C        D      N     189   0.000  0.206  256
```

### Pasted aggregate JSON (`results/dream_arcc_smoke5_reasoning_results.json`)

```json
{
  "n_trajectories": 5,
  "n_scored_for_accuracy": 5,
  "accuracy": 0.8,
  "coverage": 1.0,
  "conditional_accuracy": 0.8,
  "mean_flip_rate": 0.0,
  "mean_rationalization_score": 0.2068,
  "mean_gated_rationalization_score": 0.0166,
  "mean_early_factor": 0.2445,
  "mean_post_change_rate": 0.2574,
  "mean_length_growth": 0.0299,
  "early_convergence_fraction": 0.0,
  "disagreement_rate": 0.6,
  "sbw_lockin_rate": 0.0,
  "mean_convergence_step": 193.4,
  "median_convergence_step": 189.0,
  "mean_convergence_fraction": 0.7555,
  "median_convergence_fraction": 0.7383,
  "mean_answer_commitment_step": 196.0,
  "mean_stability_commitment_lag": 2.6,
  "mean_unwritten_reasoning_at_convergence": 0.0137,
  "mean_order_gap": 0.3836
}
```

### Side-by-side rationalization components (`direct` vs `reasoning`)

| Component | `direct` | `reasoning` | Delta (`reasoning - direct`) |
|---|---:|---:|---:|
| `early_factor` | 0.9875 | 0.2445 | -0.7430 |
| `post_change_rate` | 0.0000 | 0.2574 | +0.2574 |
| `length_growth` | 0.0000 | 0.0299 | +0.0299 |
| `score` | 0.3950 | 0.2068 | -0.1882 |
| `gated_score` | 0.0000 | 0.0166 | +0.0166 |

## Dream-v0-Instruct-7B — ARC-C 64-item paired direct vs. canonical reasoning suites (Stage 2)

- **Date (UTC):** 2026-10-30
- **Job IDs:** `dream_arcc_smoke64_direct` (GPU 0) and `dream_arcc_smoke64_reasoning` (GPU 1)
- **Git revision (`git rev-parse HEAD` at run):** `0870b06e2d055dfde520c6e5a5be690535e52fab`
- **Host & Environment:** `ai-lab2.dyn.gsu.edu` (`.venv-gpu`: `Python 3.12.2`, `torch 2.6.0+cu124`, `transformers 4.46.2`)
- **GPUs:** `CUDA_VISIBLE_DEVICES=0` (`direct_64`) and `CUDA_VISIBLE_DEVICES=1` (`reasoning_64`), `NVIDIA GeForce RTX 4090` (`24210 MiB` free / `2 MiB` used / `0%` util at launch)
- **`nvidia-smi` at launch (`ai-lab2`):**
  ```
  index, name, memory.used [MiB], memory.free [MiB], memory.total [MiB], utilization.gpu [%]
  0, NVIDIA GeForce RTX 4090, 2 MiB, 24210 MiB, 24564 MiB, 0 %
  1, NVIDIA GeForce RTX 4090, 2 MiB, 24210 MiB, 24564 MiB, 0 %
  2, NVIDIA GeForce RTX 4090, 2 MiB, 24210 MiB, 24564 MiB, 0 %
  3, NVIDIA GeForce RTX 4090, 2 MiB, 24210 MiB, 24564 MiB, 0 %
  ```
- **Sampler & adapter config (identical across both conditions):** `model_family=dream, temperature=0.2, top_p=0.95, alg=entropy, alg_temp=0.0, steps=256, max_new_tokens=256, force_deliberation_steps=0, frontier_width=None`
- **Artifacts:**
  - Fixed IDs: `results/arcc_smoke64_ids.json` (`n = 64`)
  - Direct condition: `results/arcc_smoke64_direct_input.json` -> `results/dream_arcc_smoke64_direct.jsonl`, `results/dream_arcc_smoke64_direct_run_log.json`, `results/dream_arcc_smoke64_direct_results.json`
  - Canonical reasoning condition: `results/arcc_smoke64_reasoning_input.json` -> `results/dream_arcc_smoke64_reasoning.jsonl`, `results/dream_arcc_smoke64_reasoning_run_log.json`, `results/dream_arcc_smoke64_reasoning_results.json`
  - Side-by-side comparison JSON: `results/dream_arcc_smoke64_direct_vs_reasoning.json`
- **Undecided count:**
  - `direct_64`: **0 undecided** (64 / 64 non-null at final step).
  - `reasoning_64`: **1 undecided** (63 / 64 non-null at final step; `Mercury_412681` truncated during option analysis before reaching a conclusion, so `prediction = null`, `coverage = 0.9844`, `conditional_accuracy = 0.8413`, `accuracy = 0.8281`).

### Pasted aggregate JSON — `direct_64` (`results/dream_arcc_smoke64_direct_results.json`)

```json
{
  "n_trajectories": 64,
  "n_scored_for_accuracy": 64,
  "accuracy": 0.8906,
  "coverage": 1.0,
  "conditional_accuracy": 0.8906,
  "mean_flip_rate": 0.0,
  "mean_rationalization_score": 0.3948,
  "mean_gated_rationalization_score": 0.0012,
  "mean_early_factor": 0.9852,
  "mean_post_change_rate": 0.0001,
  "mean_length_growth": 0.0035,
  "early_convergence_fraction": 1.0,
  "disagreement_rate": 0.0312,
  "sbw_lockin_rate": 0.1094,
  "mean_convergence_step": 3.7969,
  "median_convergence_step": 3.0,
  "mean_convergence_fraction": 0.0148,
  "median_convergence_fraction": 0.0117,
  "mean_answer_commitment_step": 3.7969,
  "mean_stability_commitment_lag": 0.0,
  "mean_unwritten_reasoning_at_convergence": 0.0,
  "mean_order_gap": 0.0109
}
```

### Pasted aggregate JSON — `reasoning_64` (`results/dream_arcc_smoke64_reasoning_results.json`)

```json
{
  "n_trajectories": 64,
  "n_scored_for_accuracy": 63,
  "accuracy": 0.8281,
  "coverage": 0.9844,
  "conditional_accuracy": 0.8413,
  "mean_flip_rate": 0.0106,
  "mean_rationalization_score": 0.1703,
  "mean_gated_rationalization_score": 0.0125,
  "mean_early_factor": 0.2559,
  "mean_post_change_rate": 0.1597,
  "mean_length_growth": 0.0203,
  "early_convergence_fraction": 0.0794,
  "disagreement_rate": 0.5469,
  "sbw_lockin_rate": 0.0159,
  "mean_convergence_step": 189.4603,
  "median_convergence_step": 205.0,
  "mean_convergence_fraction": 0.7401,
  "median_convergence_fraction": 0.8008,
  "mean_answer_commitment_step": 192.2857,
  "mean_stability_commitment_lag": 2.8254,
  "mean_unwritten_reasoning_at_convergence": 0.0145,
  "mean_order_gap": 0.3762
}
```

### Side-by-side rationalization components (`direct_64` vs `reasoning_64`)

| Component | `direct_64` (`n=64`) | `reasoning_64` (`n=64`) | Delta (`reasoning_64 - direct_64`) |
|---|---:|---:|---:|
| `early_factor` | 0.9852 | 0.2559 | -0.7293 |
| `post_change_rate` | 0.0001 | 0.1597 | +0.1596 |
| `length_growth` | 0.0035 | 0.0203 | +0.0168 |
| `score` | 0.3948 | 0.1703 | -0.2245 |
| `gated_score` | 0.0012 | 0.0125 | +0.0113 |

## Dream-v0-Instruct-7B — GSM8K 5-item numeric smoke (smoke clean (4/5 decided))

- **Date (UTC):** 2026-11-01
- **Job ID:** `dream_gsm8k_smoke5`
- **Git revision (`git rev-parse HEAD` at run):** `649a1f344398ff19fe28a1fab7e75c7569051c4b`
- **Environment:** conda env `cdr-gpu` (`torch 2.5.1+cu121`, `transformers 4.46.2`)
- **GPU:** index `0` (`CUDA_VISIBLE_DEVICES=0`, `NVIDIA GeForce RTX 4090`, `24210 MiB` free at launch)
- **`nvidia-smi` at launch:**
  ```
  index, name, memory.used [MiB], memory.total [MiB], utilization.gpu [%]
  0, NVIDIA GeForce RTX 4090, 2 MiB, 24564 MiB, 0 %
  1, NVIDIA GeForce RTX 4090, 2 MiB, 24564 MiB, 0 %
  2, NVIDIA GeForce RTX 4090, 2 MiB, 24564 MiB, 0 %
  3, NVIDIA GeForce RTX 4090, 2 MiB, 24564 MiB, 0 %
  ```
- **Sampler & adapter config:** `model_family=dream, temperature=0.2, top_p=0.95, alg=entropy, alg_temp=0.0, steps=256, max_new_tokens=256, force_deliberation_steps=0, frontier_width=None`
- **Artifacts:**
  - Input: `results/dream_gsm8k_smoke5_input.json`
  - Fixed IDs: `results/dream_gsm8k_smoke5_ids.json`
  - Trajectories: `results/dream_gsm8k_smoke5.jsonl`
  - Run log: `results/dream_gsm8k_smoke5_run_log.json`
  - Scored output: `results/dream_gsm8k_smoke5_results.json`
- **Undecided count:** **1 undecided** (4 / 5 non-null at final step).

### Pasted `traj-eval run` output

```
=== traj-eval summary ===
Trajectories:            5
Accuracy (final):        0.6000
Mean flip rate:          0.0106
Mean rationalization:    0.1809
  [components] early=0.4523 post_change=0.0000 len_growth=0.0000 gated=0.0000
Early convergence frac:  0.0000
Mean convergence step:   111.25
Median convergence step: 117.00
Mean convergence frac:   0.4346
Mean T_commit step:      111.25
Mean lag (T_comm-T_stab):0.00
Mean unwritten @ T_stab: 0.0000

Per-trajectory:
id                   final    gold   corr  conv  flip   rat    steps
--------------------------------------------------------------------
gsm8k_test_0         None     18                 0.000  0.000  256  
gsm8k_test_1         3        3      Y     126   0.000  0.203  256  
gsm8k_test_2         65000    70000  N     142   0.034  0.178  256  
gsm8k_test_3         540      540    Y     108   0.013  0.231  256  
gsm8k_test_4         20       20     Y     69    0.005  0.292  256
```

### Pasted aggregate JSON (`results/dream_gsm8k_smoke5_results.json`)

```json
{
  "n_trajectories": 5,
  "n_scored_for_accuracy": 4,
  "accuracy": 0.6,
  "coverage": 0.8,
  "conditional_accuracy": 0.75,
  "mean_flip_rate": 0.0106,
  "mean_rationalization_score": 0.1809,
  "mean_gated_rationalization_score": 0.0,
  "mean_early_factor": 0.4523,
  "mean_post_change_rate": 0.0,
  "mean_length_growth": 0.0,
  "early_convergence_fraction": 0.0,
  "disagreement_rate": 0.0,
  "sbw_lockin_rate": 0.0,
  "mean_convergence_step": 111.25,
  "median_convergence_step": 117.0,
  "mean_convergence_fraction": 0.4346,
  "median_convergence_fraction": 0.457,
  "mean_answer_commitment_step": 111.25,
  "mean_stability_commitment_lag": 0.0,
  "mean_unwritten_reasoning_at_convergence": 0.0,
  "mean_order_gap": 0.2158
}
```

## Dream-v0-Instruct-7B — ARC-C 5-item EOS-suppression K=64 smoke (smoke clean (5/5 decided))

- **Date (UTC):** 2026-11-01
- **Job ID:** `dream_arcc_eos_suppression_k64_smoke5`
- **Git revision (`git rev-parse HEAD` at run):** `2278ff57e798f54820822d519d63f320e07828a3`
- **Environment:** conda env `cdr-gpu` (`torch 2.5.1+cu121`, `transformers 4.46.2`)
- **GPU:** index `0` (`CUDA_VISIBLE_DEVICES=0`, `NVIDIA GeForce RTX 4090`, `24210 MiB` free at launch)
- **`nvidia-smi` at launch:**
  ```
  index, name, memory.used [MiB], memory.total [MiB], utilization.gpu [%]
  0, NVIDIA GeForce RTX 4090, 2 MiB, 24564 MiB, 0 %
  1, NVIDIA GeForce RTX 4090, 2 MiB, 24564 MiB, 0 %
  2, NVIDIA GeForce RTX 4090, 2 MiB, 24564 MiB, 0 %
  3, NVIDIA GeForce RTX 4090, 2 MiB, 24564 MiB, 0 %
  ```
- **Sampler & adapter config:** `model_family=dream, temperature=0.2, top_p=0.95, alg=entropy, alg_temp=0.0, steps=256, max_new_tokens=256, force_deliberation_steps=64, frontier_width=None`
- **Artifacts:**
  - Input: `results/dream_arcc_smoke5_input.json`
  - Fixed IDs: `results/dream_arcc_smoke5_ids.json`
  - Trajectories: `results/dream_arcc_eos_suppression_k64_smoke5.jsonl`
  - Run log: `results/dream_arcc_eos_suppression_k64_smoke5_run_log.json`
  - Scored output: `results/dream_arcc_eos_suppression_k64_smoke5_results.json`
- **Undecided count:** **0 undecided** (5 / 5 non-null at final step).

### Pasted `traj-eval run` output

```
=== traj-eval summary ===
Trajectories:            5
Accuracy (final):        0.8000
Mean flip rate:          0.0008
Mean rationalization:    0.7962
  [components] early=0.9617 post_change=0.5287 len_growth=1.0000 gated=0.6647
Early convergence frac:  1.0000
Mean convergence step:   9.80
Median convergence step: 3.00
Mean convergence frac:   0.0383
Mean T_commit step:      70.00
Mean lag (T_comm-T_stab):60.20
Mean unwritten @ T_stab: 0.4733

Per-trajectory:
id                   final    gold   corr  conv  flip   rat    steps
--------------------------------------------------------------------
Mercury_7175875      C        C      Y     3     0.000  0.994  256  
Mercury_SC_409171    B        B      Y     3     0.000  0.862  256  
Mercury_SC_408547    C        C      Y     3     0.000  0.700  256  
Mercury_407327       D        D      Y     3     0.000  0.787  256  
MCAS_2006_9_44       B        D      N     37    0.004  0.638  256
```

### Pasted aggregate JSON (`results/dream_arcc_eos_suppression_k64_smoke5_results.json`)

```json
{
  "n_trajectories": 5,
  "n_scored_for_accuracy": 5,
  "accuracy": 0.8,
  "coverage": 1.0,
  "conditional_accuracy": 0.8,
  "mean_flip_rate": 0.0008,
  "mean_rationalization_score": 0.7962,
  "mean_gated_rationalization_score": 0.6647,
  "mean_early_factor": 0.9617,
  "mean_post_change_rate": 0.5287,
  "mean_length_growth": 1.0,
  "early_convergence_fraction": 1.0,
  "disagreement_rate": 1.0,
  "sbw_lockin_rate": 0.2,
  "mean_convergence_step": 9.8,
  "median_convergence_step": 3.0,
  "mean_convergence_fraction": 0.0383,
  "median_convergence_fraction": 0.0117,
  "mean_answer_commitment_step": 70.0,
  "mean_stability_commitment_lag": 60.2,
  "mean_unwritten_reasoning_at_convergence": 0.4733,
  "mean_order_gap": 0.1359
}
```

### Side-by-side rationalization components (`unrestricted` vs `eos_suppression_k64`)

| Component | `unrestricted` | `eos_suppression_k64` | Delta (`eos_suppression_k64 - unrestricted`) |
|---|---:|---:|---:|
| `early_factor` | 0.9875 | 0.9617 | -0.0258 |
| `post_change_rate` | 0.0000 | 0.5287 | +0.5287 |
| `length_growth` | 0.0000 | 1.0000 | +1.0000 |
| `score` | 0.3950 | 0.7962 | +0.4012 |
| `gated_score` | 0.0000 | 0.6647 | +0.6647 |

## LLaDA-8B-Instruct — ARC-C 5-item direct-answer smoke (smoke clean (5/5 decided))

- **Date (UTC):** 2026-11-01
- **Job ID:** `llada_instruct_arcc_smoke5_direct`
- **Git revision (`git rev-parse HEAD` at run):** `91647a5aa4c8f10a7ef7ba52915730c028fc72bd`
- **Environment:** conda env `cdr-gpu` (`torch 2.5.1+cu121`, `transformers 4.46.2`)
- **GPU:** index `1` (`CUDA_VISIBLE_DEVICES=1`, `NVIDIA GeForce RTX 4090`, `24210 MiB` free at launch)
- **`nvidia-smi` at launch:**
  ```
  index, name, memory.used [MiB], memory.total [MiB], utilization.gpu [%]
  0, NVIDIA GeForce RTX 4090, 2 MiB, 24564 MiB, 0 %
  1, NVIDIA GeForce RTX 4090, 2 MiB, 24564 MiB, 0 %
  2, NVIDIA GeForce RTX 4090, 2 MiB, 24564 MiB, 0 %
  3, NVIDIA GeForce RTX 4090, 2 MiB, 24564 MiB, 0 %
  ```
- **Sampler & adapter config:** `model_family=llada-instruct, temperature=0.0, remasking=low_confidence, steps=256, gen_length=256, block_length=256, confidence_eos_eot_inf=True, frontier_width=None`
- **Artifacts:**
  - Input: `results/dream_arcc_smoke5_input.json`
  - Fixed IDs: `results/llada_instruct_arcc_smoke5_direct_ids.json`
  - Trajectories: `results/llada_instruct_arcc_smoke5_direct.jsonl`
  - Run log: `results/llada_instruct_arcc_smoke5_direct_run_log.json`
  - Scored output: `results/llada_instruct_arcc_smoke5_direct_results.json`
- **Undecided count:** **0 undecided** (5 / 5 non-null at final step).

### Pasted `traj-eval run` output

```
=== traj-eval summary ===
Trajectories:            5
Accuracy (final):        1.0000
Mean flip rate:          0.0000
Mean rationalization:    0.9969
  [components] early=0.9922 post_change=1.0000 len_growth=1.0000 gated=0.9922
Early convergence frac:  1.0000
Mean convergence step:   2.00
Median convergence step: 2.00
Mean convergence frac:   0.0078
Mean T_commit step:      2.00
Mean lag (T_comm-T_stab):0.00
Mean unwritten @ T_stab: 1.0000

Per-trajectory:
id                   final    gold   corr  conv  flip   rat    steps
--------------------------------------------------------------------
Mercury_7175875      C        C      Y     2     0.000  0.997  256  
Mercury_SC_409171    B        B      Y     2     0.000  0.997  256  
Mercury_SC_408547    C        C      Y     2     0.000  0.997  256  
Mercury_407327       D        D      Y     2     0.000  0.997  256  
MCAS_2006_9_44       D        D      Y     2     0.000  0.997  256
```

### Pasted aggregate JSON (`results/llada_instruct_arcc_smoke5_direct_results.json`)

```json
{
  "n_trajectories": 5,
  "n_scored_for_accuracy": 5,
  "accuracy": 1.0,
  "coverage": 1.0,
  "conditional_accuracy": 1.0,
  "mean_flip_rate": 0.0,
  "mean_rationalization_score": 0.9969,
  "mean_gated_rationalization_score": 0.9922,
  "mean_early_factor": 0.9922,
  "mean_post_change_rate": 1.0,
  "mean_length_growth": 1.0,
  "early_convergence_fraction": 1.0,
  "disagreement_rate": 1.0,
  "sbw_lockin_rate": 0.0,
  "mean_convergence_step": 2.0,
  "median_convergence_step": 2.0,
  "mean_convergence_fraction": 0.0078,
  "median_convergence_fraction": 0.0078,
  "mean_answer_commitment_step": 2.0,
  "mean_stability_commitment_lag": 0.0,
  "mean_unwritten_reasoning_at_convergence": 1.0,
  "mean_order_gap": -0.4961
}
```


## LLaDA-8B-Instruct — ARC-C 5-item reasoning-condition smoke (reasoning smoke clean)

- **Date (UTC):** 2026-11-01
- **Job ID:** `llada_instruct_arcc_smoke5_reasoning`
- **Git revision (`git rev-parse HEAD` at run):** `91647a5aa4c8f10a7ef7ba52915730c028fc72bd`
- **Environment:** conda env `cdr-gpu` (`torch 2.5.1+cu121`, `transformers 4.46.2`)
- **GPU:** index `1` (`CUDA_VISIBLE_DEVICES=1`, `NVIDIA GeForce RTX 4090`, `24210 MiB` free at launch)
- **`nvidia-smi` at launch:**
  ```
  index, name, memory.used [MiB], memory.total [MiB], utilization.gpu [%]
  0, NVIDIA GeForce RTX 4090, 2 MiB, 24564 MiB, 0 %
  1, NVIDIA GeForce RTX 4090, 2 MiB, 24564 MiB, 0 %
  2, NVIDIA GeForce RTX 4090, 2 MiB, 24564 MiB, 0 %
  3, NVIDIA GeForce RTX 4090, 2 MiB, 24564 MiB, 0 %
  ```
- **Sampler & adapter config:** `model_family=llada-instruct, temperature=0.0, remasking=low_confidence, steps=256, gen_length=256, block_length=256, confidence_eos_eot_inf=True, frontier_width=None`
- **Artifacts:**
  - Input: `results/dream_arcc_smoke5_reasoning_input.json`
  - Fixed IDs: `results/llada_instruct_arcc_smoke5_reasoning_ids.json`
  - Trajectories: `results/llada_instruct_arcc_smoke5_reasoning.jsonl`
  - Run log: `results/llada_instruct_arcc_smoke5_reasoning_run_log.json`
  - Scored output: `results/llada_instruct_arcc_smoke5_reasoning_results.json`
- **Undecided count:** **0 undecided** (5 / 5 non-null at final step).
- **Reasoning smoke gate:** `reasoning smoke clean` (commitment clocks on `5/5`, post-verdict change on `5`, pre-verdict rationale on `3`).

### Pasted `traj-eval run` output

```
=== traj-eval summary ===
Trajectories:            5
Accuracy (final):        1.0000
Mean flip rate:          0.0000
Mean rationalization:    0.9691
  [components] early=0.9227 post_change=1.0000 len_growth=1.0000 gated=0.9227
Early convergence frac:  1.0000
Mean convergence step:   19.80
Median convergence step: 16.00
Mean convergence frac:   0.0773
Mean T_commit step:      19.80
Mean lag (T_comm-T_stab):0.00
Mean unwritten @ T_stab: 0.9217

Per-trajectory:
id                   final    gold   corr  conv  flip   rat    steps
--------------------------------------------------------------------
Mercury_7175875      C        C      Y     2     0.000  0.997  256  
Mercury_SC_409171    B        B      Y     41    0.000  0.936  256  
Mercury_SC_408547    C        C      Y     36    0.000  0.944  256  
Mercury_407327       D        D      Y     4     0.000  0.994  256  
MCAS_2006_9_44       D        D      Y     16    0.000  0.975  256
```

### Pasted aggregate JSON (`results/llada_instruct_arcc_smoke5_reasoning_results.json`)

```json
{
  "n_trajectories": 5,
  "n_scored_for_accuracy": 5,
  "accuracy": 1.0,
  "coverage": 1.0,
  "conditional_accuracy": 1.0,
  "mean_flip_rate": 0.0,
  "mean_rationalization_score": 0.9691,
  "mean_gated_rationalization_score": 0.9227,
  "mean_early_factor": 0.9227,
  "mean_post_change_rate": 1.0,
  "mean_length_growth": 1.0,
  "early_convergence_fraction": 1.0,
  "disagreement_rate": 1.0,
  "sbw_lockin_rate": 0.0,
  "mean_convergence_step": 19.8,
  "median_convergence_step": 16.0,
  "mean_convergence_fraction": 0.0773,
  "median_convergence_fraction": 0.0625,
  "mean_answer_commitment_step": 19.8,
  "mean_stability_commitment_lag": 0.0,
  "mean_unwritten_reasoning_at_convergence": 0.9217,
  "mean_order_gap": -0.418
}
```

### Side-by-side rationalization components (`direct` vs `reasoning`)

| Component | `direct` | `reasoning` | Delta (`reasoning - direct`) |
|---|---:|---:|---:|
| `early_factor` | 0.9922 | 0.9227 | -0.0695 |
| `post_change_rate` | 1.0000 | 1.0000 | +0.0000 |
| `length_growth` | 1.0000 | 1.0000 | +0.0000 |
| `score` | 0.9969 | 0.9691 | -0.0278 |
| `gated_score` | 0.9922 | 0.9227 | -0.0695 |


## LLaDA-1.5 — ARC-C 5-item direct-answer smoke (smoke clean (5/5 decided))

- **Date (UTC):** 2026-11-01
- **Job ID:** `llada_15_arcc_smoke5_direct`
- **Git revision (`git rev-parse HEAD` at run):** `91647a5aa4c8f10a7ef7ba52915730c028fc72bd`
- **Environment:** conda env `cdr-gpu` (`torch 2.5.1+cu121`, `transformers 4.46.2`)
- **GPU:** index `2` (`CUDA_VISIBLE_DEVICES=2`, `NVIDIA GeForce RTX 4090`, `24210 MiB` free at launch)
- **`nvidia-smi` at launch:**
  ```
  index, name, memory.used [MiB], memory.total [MiB], utilization.gpu [%]
  0, NVIDIA GeForce RTX 4090, 2 MiB, 24564 MiB, 0 %
  1, NVIDIA GeForce RTX 4090, 2 MiB, 24564 MiB, 0 %
  2, NVIDIA GeForce RTX 4090, 2 MiB, 24564 MiB, 0 %
  3, NVIDIA GeForce RTX 4090, 2 MiB, 24564 MiB, 0 %
  ```
- **Sampler & adapter config:** `model_family=llada-1.5, temperature=0.0, remasking=low_confidence, steps=256, gen_length=256, block_length=256, confidence_eos_eot_inf=True, frontier_width=None`
- **Artifacts:**
  - Input: `results/dream_arcc_smoke5_input.json`
  - Fixed IDs: `results/llada_15_arcc_smoke5_direct_ids.json`
  - Trajectories: `results/llada_15_arcc_smoke5_direct.jsonl`
  - Run log: `results/llada_15_arcc_smoke5_direct_run_log.json`
  - Scored output: `results/llada_15_arcc_smoke5_direct_results.json`
- **Undecided count:** **0 undecided** (5 / 5 non-null at final step).

### Pasted `traj-eval run` output

```
=== traj-eval summary ===
Trajectories:            5
Accuracy (final):        1.0000
Mean flip rate:          0.0000
Mean rationalization:    0.9969
  [components] early=0.9922 post_change=1.0000 len_growth=1.0000 gated=0.9922
Early convergence frac:  1.0000
Mean convergence step:   2.00
Median convergence step: 2.00
Mean convergence frac:   0.0078
Mean T_commit step:      2.00
Mean lag (T_comm-T_stab):0.00
Mean unwritten @ T_stab: 1.0000

Per-trajectory:
id                   final    gold   corr  conv  flip   rat    steps
--------------------------------------------------------------------
Mercury_7175875      C        C      Y     2     0.000  0.997  256  
Mercury_SC_409171    B        B      Y     2     0.000  0.997  256  
Mercury_SC_408547    C        C      Y     2     0.000  0.997  256  
Mercury_407327       D        D      Y     2     0.000  0.997  256  
MCAS_2006_9_44       D        D      Y     2     0.000  0.997  256
```

### Pasted aggregate JSON (`results/llada_15_arcc_smoke5_direct_results.json`)

```json
{
  "n_trajectories": 5,
  "n_scored_for_accuracy": 5,
  "accuracy": 1.0,
  "coverage": 1.0,
  "conditional_accuracy": 1.0,
  "mean_flip_rate": 0.0,
  "mean_rationalization_score": 0.9969,
  "mean_gated_rationalization_score": 0.9922,
  "mean_early_factor": 0.9922,
  "mean_post_change_rate": 1.0,
  "mean_length_growth": 1.0,
  "early_convergence_fraction": 1.0,
  "disagreement_rate": 1.0,
  "sbw_lockin_rate": 0.0,
  "mean_convergence_step": 2.0,
  "median_convergence_step": 2.0,
  "mean_convergence_fraction": 0.0078,
  "median_convergence_fraction": 0.0078,
  "mean_answer_commitment_step": 2.0,
  "mean_stability_commitment_lag": 0.0,
  "mean_unwritten_reasoning_at_convergence": 1.0,
  "mean_order_gap": -0.4969
}
```


## LLaDA-1.5 — ARC-C 5-item reasoning-condition smoke (reasoning smoke clean)

- **Date (UTC):** 2026-11-01
- **Job ID:** `llada_15_arcc_smoke5_reasoning`
- **Git revision (`git rev-parse HEAD` at run):** `91647a5aa4c8f10a7ef7ba52915730c028fc72bd`
- **Environment:** conda env `cdr-gpu` (`torch 2.5.1+cu121`, `transformers 4.46.2`)
- **GPU:** index `2` (`CUDA_VISIBLE_DEVICES=2`, `NVIDIA GeForce RTX 4090`, `24210 MiB` free at launch)
- **`nvidia-smi` at launch:**
  ```
  index, name, memory.used [MiB], memory.total [MiB], utilization.gpu [%]
  0, NVIDIA GeForce RTX 4090, 2 MiB, 24564 MiB, 0 %
  1, NVIDIA GeForce RTX 4090, 2 MiB, 24564 MiB, 0 %
  2, NVIDIA GeForce RTX 4090, 2 MiB, 24564 MiB, 0 %
  3, NVIDIA GeForce RTX 4090, 2 MiB, 24564 MiB, 0 %
  ```
- **Sampler & adapter config:** `model_family=llada-1.5, temperature=0.0, remasking=low_confidence, steps=256, gen_length=256, block_length=256, confidence_eos_eot_inf=True, frontier_width=None`
- **Artifacts:**
  - Input: `results/dream_arcc_smoke5_reasoning_input.json`
  - Fixed IDs: `results/llada_15_arcc_smoke5_reasoning_ids.json`
  - Trajectories: `results/llada_15_arcc_smoke5_reasoning.jsonl`
  - Run log: `results/llada_15_arcc_smoke5_reasoning_run_log.json`
  - Scored output: `results/llada_15_arcc_smoke5_reasoning_results.json`
- **Undecided count:** **0 undecided** (5 / 5 non-null at final step).
- **Reasoning smoke gate:** `reasoning smoke clean` (commitment clocks on `5/5`, post-verdict change on `5`, pre-verdict rationale on `3`).

### Pasted `traj-eval run` output

```
=== traj-eval summary ===
Trajectories:            5
Accuracy (final):        1.0000
Mean flip rate:          0.0000
Mean rationalization:    0.9785
  [components] early=0.9461 post_change=1.0000 len_growth=1.0000 gated=0.9461
Early convergence frac:  1.0000
Mean convergence step:   13.80
Median convergence step: 8.00
Mean convergence frac:   0.0539
Mean T_commit step:      13.80
Mean lag (T_comm-T_stab):0.00
Mean unwritten @ T_stab: 0.9455

Per-trajectory:
id                   final    gold   corr  conv  flip   rat    steps
--------------------------------------------------------------------
Mercury_7175875      C        C      Y     4     0.000  0.994  256  
Mercury_SC_409171    B        B      Y     8     0.000  0.988  256  
Mercury_SC_408547    C        C      Y     36    0.000  0.944  256  
Mercury_407327       D        D      Y     4     0.000  0.994  256  
MCAS_2006_9_44       D        D      Y     17    0.000  0.973  256
```

### Pasted aggregate JSON (`results/llada_15_arcc_smoke5_reasoning_results.json`)

```json
{
  "n_trajectories": 5,
  "n_scored_for_accuracy": 5,
  "accuracy": 1.0,
  "coverage": 1.0,
  "conditional_accuracy": 1.0,
  "mean_flip_rate": 0.0,
  "mean_rationalization_score": 0.9785,
  "mean_gated_rationalization_score": 0.9461,
  "mean_early_factor": 0.9461,
  "mean_post_change_rate": 1.0,
  "mean_length_growth": 1.0,
  "early_convergence_fraction": 1.0,
  "disagreement_rate": 1.0,
  "sbw_lockin_rate": 0.0,
  "mean_convergence_step": 13.8,
  "median_convergence_step": 8.0,
  "mean_convergence_fraction": 0.0539,
  "median_convergence_fraction": 0.0312,
  "mean_answer_commitment_step": 13.8,
  "mean_stability_commitment_lag": 0.0,
  "mean_unwritten_reasoning_at_convergence": 0.9455,
  "mean_order_gap": -0.443
}
```

### Side-by-side rationalization components (`direct` vs `reasoning`)

| Component | `direct` | `reasoning` | Delta (`reasoning - direct`) |
|---|---:|---:|---:|
| `early_factor` | 0.9922 | 0.9461 | -0.0461 |
| `post_change_rate` | 1.0000 | 1.0000 | +0.0000 |
| `length_growth` | 1.0000 | 1.0000 | +0.0000 |
| `score` | 0.9969 | 0.9785 | -0.0184 |
| `gated_score` | 0.9922 | 0.9461 | -0.0461 |


## LLaDA-8B-Instruct — ARC-C 64-item direct-answer suite (smoke clean (64/64 decided))

- **Date (UTC):** 2026-11-01
- **Job ID:** `llada_instruct_arcc_smoke64_direct`
- **Git revision (`git rev-parse HEAD` at run):** `91647a5aa4c8f10a7ef7ba52915730c028fc72bd`
- **Environment:** conda env `cdr-gpu` (`torch 2.5.1+cu121`, `transformers 4.46.2`)
- **GPU:** index `1` (`CUDA_VISIBLE_DEVICES=1`, `NVIDIA GeForce RTX 4090`, `24210 MiB` free at launch)
- **`nvidia-smi` at launch:**
  ```
  index, name, memory.used [MiB], memory.total [MiB], utilization.gpu [%]
  0, NVIDIA GeForce RTX 4090, 2 MiB, 24564 MiB, 0 %
  1, NVIDIA GeForce RTX 4090, 2 MiB, 24564 MiB, 0 %
  2, NVIDIA GeForce RTX 4090, 2 MiB, 24564 MiB, 0 %
  3, NVIDIA GeForce RTX 4090, 2 MiB, 24564 MiB, 0 %
  ```
- **Sampler & adapter config:** `model_family=llada-instruct, temperature=0.0, remasking=low_confidence, steps=256, gen_length=256, block_length=256, confidence_eos_eot_inf=True, frontier_width=None`
- **Artifacts:**
  - Input: `results/arcc_smoke64_direct_input.json`
  - Fixed IDs: `results/arcc_smoke64_ids.json`
  - Trajectories: `results/llada_instruct_arcc_smoke64_direct.jsonl`
  - Run log: `results/llada_instruct_arcc_smoke64_direct_run_log.json`
  - Scored output: `results/llada_instruct_arcc_smoke64_direct_results.json`
- **Undecided count:** **0 undecided** (64 / 64 non-null at final step).

### Pasted `traj-eval run` output

```
=== traj-eval summary ===
Trajectories:            64
Accuracy (final):        0.8750
Mean flip rate:          0.0001
Mean rationalization:    0.9917
  [components] early=0.9826 post_change=1.0000 len_growth=0.9929 gated=0.9818
Early convergence frac:  0.9844
Mean convergence step:   4.45
Median convergence step: 2.00
Mean convergence frac:   0.0174
Mean T_commit step:      2.00
Mean lag (T_comm-T_stab):-2.45
Mean unwritten @ T_stab: 0.9899

Per-trajectory:
id                   final    gold   corr  conv  flip   rat    steps
--------------------------------------------------------------------
Mercury_7175875      C        C      Y     2     0.000  0.997  256  
Mercury_SC_409171    B        B      Y     2     0.000  0.997  256  
Mercury_SC_408547    C        C      Y     2     0.000  0.997  256  
Mercury_407327       D        D      Y     2     0.000  0.997  256  
MCAS_2006_9_44       D        D      Y     2     0.000  0.997  256  
Mercury_7270393      B        B      Y     2     0.000  0.997  256  
MCAS_2014_5_7        C        C      Y     2     0.000  0.997  256  
Mercury_7086660      C        C      Y     2     0.000  0.997  256  
Mercury_7168805      B        B      Y     2     0.000  0.997  256  
MCAS_2003_8_11       A        A      Y     0     0.000  1.000  256  
Mercury_7250058      C        B      N     2     0.000  0.997  256  
Mercury_7012740      A        A      Y     2     0.000  0.997  256  
Mercury_LBS10610     C        C      Y     2     0.000  0.997  256  
Mercury_SC_407400    C        C      Y     2     0.000  0.997  256  
Mercury_7212993      C        C      Y     2     0.000  0.997  256  
Mercury_SC_413240    A        A      Y     2     0.000  0.997  256  
Mercury_7186358      C        C      Y     2     0.000  0.997  256  
Mercury_7166425      B        B      Y     2     0.000  0.997  256  
MDSA_2007_8_3        A        A      Y     2     0.000  0.997  256  
Mercury_7094290      C        C      Y     2     0.000  0.997  256  
Mercury_7186568      B        B      Y     2     0.000  0.997  256  
Mercury_402216       B        B      Y     0     0.000  1.000  256  
Mercury_404894       A        A      Y     2     0.000  0.997  256  
MCAS_2002_8_11       C        C      Y     2     0.000  0.997  256  
Mercury_SC_405086    B        B      Y     0     0.000  1.000  256  
Mercury_SC_408324    D        D      Y     2     0.000  0.997  256  
Mercury_7218820      D        B      N     2     0.000  0.997  256  
Mercury_412202       D        B      N     2     0.000  0.997  256  
Mercury_SC_409139    D        C      N     2     0.000  0.997  256  
Mercury_400687       B        B      Y     2     0.000  0.997  256  
Mercury_7171605      D        D      Y     2     0.000  0.997  256  
Mercury_7210245      C        C      Y     2     0.000  0.997  256  
AKDE&ED_2008_4_25    A        A      Y     2     0.000  0.997  256  
AKDE&ED_2008_4_19    C        C      Y     2     0.000  0.997  256  
Mercury_SC_400402    C        A      N     2     0.000  0.997  256  
Mercury_7234308      A        A      Y     2     0.000  0.997  256  
ACTAAP_2014_5_8      B        B      Y     2     0.000  0.997  256  
Mercury_400407       B        B      Y     2     0.000  0.997  256  
Mercury_7116288      C        C      Y     2     0.000  0.997  256  
MCAS_2004_9_15-v1    B        B      Y     2     0.000  0.997  256  
NYSEDREGENTS_2015_4_26 C        C      Y     2     0.000  0.997  256  
Mercury_SC_401620    A        A      Y     2     0.000  0.997  256  
Mercury_400877       C        C      Y     2     0.000  0.997  256  
Mercury_7174213      D        D      Y     2     0.000  0.997  256  
NYSEDREGENTS_2008_8_34 B        B      Y     2     0.000  0.997  256  
Mercury_7212398      B        B      Y     2     0.000  0.997  256  
Mercury_SC_401290    C        C      Y     2     0.000  0.997  256  
Mercury_SC_402120    C        C      Y     2     0.000  0.997  256  
Mercury_184975       C        C      Y     2     0.000  0.997  256  
Mercury_SC_400578    B        A      N     2     0.000  0.997  256  
MCAS_2001_8_4        D        D      Y     2     0.000  0.997  256  
MCAS_2003_5_33       C        C      Y     2     0.000  0.997  256  
Mercury_7068513      A        A      Y     2     0.000  0.997  256  
AKDE&ED_2008_4_26    B        C      N     2     0.000  0.997  256  
Mercury_7235638      C        C      Y     2     0.000  0.997  256  
MDSA_2009_5_20       B        B      Y     165   0.008  0.652  256  
Mercury_178325       D        C      N     2     0.000  0.997  256  
Mercury_7212678      A        A      Y     2     0.000  0.997  256  
TAKS_2009_8_32       A        A      Y     2     0.000  0.997  256  
Mercury_412681       C        C      Y     2     0.000  0.997  256  
Mercury_400440       D        D      Y     2     0.000  0.997  256  
Mercury_SC_416529    B        B      Y     2     0.000  0.997  256  
MCAS_2006_8_1        D        D      Y     2     0.000  0.997  256  
TIMSS_2003_8_pg80    C        C      Y     2     0.000  0.997  256
```

### Pasted aggregate JSON (`results/llada_instruct_arcc_smoke64_direct_results.json`)

```json
{
  "n_trajectories": 64,
  "n_scored_for_accuracy": 64,
  "accuracy": 0.875,
  "coverage": 1.0,
  "conditional_accuracy": 0.875,
  "mean_flip_rate": 0.0001,
  "mean_rationalization_score": 0.9917,
  "mean_gated_rationalization_score": 0.9818,
  "mean_early_factor": 0.9826,
  "mean_post_change_rate": 1.0,
  "mean_length_growth": 0.9929,
  "early_convergence_fraction": 0.9844,
  "disagreement_rate": 1.0,
  "sbw_lockin_rate": 0.125,
  "mean_convergence_step": 4.4531,
  "median_convergence_step": 2.0,
  "mean_convergence_fraction": 0.0174,
  "median_convergence_fraction": 0.0078,
  "mean_answer_commitment_step": 2.0,
  "mean_stability_commitment_lag": -2.4531,
  "mean_unwritten_reasoning_at_convergence": 0.9899,
  "mean_order_gap": -0.4964
}
```


## LLaDA-1.5 — ARC-C 64-item direct-answer suite (smoke clean (64/64 decided))

- **Date (UTC):** 2026-11-01
- **Job ID:** `llada_15_arcc_smoke64_direct`
- **Git revision (`git rev-parse HEAD` at run):** `91647a5aa4c8f10a7ef7ba52915730c028fc72bd`
- **Environment:** conda env `cdr-gpu` (`torch 2.5.1+cu121`, `transformers 4.46.2`)
- **GPU:** index `2` (`CUDA_VISIBLE_DEVICES=2`, `NVIDIA GeForce RTX 4090`, `24210 MiB` free at launch)
- **`nvidia-smi` at launch:**
  ```
  index, name, memory.used [MiB], memory.total [MiB], utilization.gpu [%]
  0, NVIDIA GeForce RTX 4090, 2 MiB, 24564 MiB, 0 %
  1, NVIDIA GeForce RTX 4090, 2 MiB, 24564 MiB, 0 %
  2, NVIDIA GeForce RTX 4090, 2 MiB, 24564 MiB, 0 %
  3, NVIDIA GeForce RTX 4090, 2 MiB, 24564 MiB, 0 %
  ```
- **Sampler & adapter config:** `model_family=llada-1.5, temperature=0.0, remasking=low_confidence, steps=256, gen_length=256, block_length=256, confidence_eos_eot_inf=True, frontier_width=None`
- **Artifacts:**
  - Input: `results/arcc_smoke64_direct_input.json`
  - Fixed IDs: `results/arcc_smoke64_ids.json`
  - Trajectories: `results/llada_15_arcc_smoke64_direct.jsonl`
  - Run log: `results/llada_15_arcc_smoke64_direct_run_log.json`
  - Scored output: `results/llada_15_arcc_smoke64_direct_results.json`
- **Undecided count:** **0 undecided** (64 / 64 non-null at final step).

### Pasted `traj-eval run` output

```
=== traj-eval summary ===
Trajectories:            64
Accuracy (final):        0.8750
Mean flip rate:          0.0004
Mean rationalization:    0.9856
  [components] early=0.9711 post_change=1.0000 len_growth=0.9856 gated=0.9707
Early convergence frac:  0.9688
Mean convergence step:   7.41
Median convergence step: 2.00
Mean convergence frac:   0.0289
Mean T_commit step:      2.00
Mean lag (T_comm-T_stab):-5.41
Mean unwritten @ T_stab: 0.9784

Per-trajectory:
id                   final    gold   corr  conv  flip   rat    steps
--------------------------------------------------------------------
Mercury_7175875      C        C      Y     2     0.000  0.997  256  
Mercury_SC_409171    B        B      Y     2     0.000  0.997  256  
Mercury_SC_408547    C        C      Y     2     0.000  0.997  256  
Mercury_407327       D        D      Y     2     0.000  0.997  256  
MCAS_2006_9_44       D        D      Y     2     0.000  0.997  256  
Mercury_7270393      B        B      Y     2     0.000  0.997  256  
MCAS_2014_5_7        C        C      Y     2     0.000  0.997  256  
Mercury_7086660      C        C      Y     2     0.000  0.997  256  
Mercury_7168805      B        B      Y     2     0.000  0.997  256  
MCAS_2003_8_11       A        A      Y     2     0.000  0.997  256  
Mercury_7250058      C        B      N     2     0.000  0.997  256  
Mercury_7012740      A        A      Y     2     0.000  0.997  256  
Mercury_LBS10610     C        C      Y     2     0.000  0.997  256  
Mercury_SC_407400    C        C      Y     2     0.000  0.997  256  
Mercury_7212993      C        C      Y     2     0.000  0.997  256  
Mercury_SC_413240    A        A      Y     2     0.000  0.997  256  
Mercury_7186358      C        C      Y     2     0.000  0.997  256  
Mercury_7166425      B        B      Y     2     0.000  0.997  256  
MDSA_2007_8_3        A        A      Y     2     0.000  0.997  256  
Mercury_7094290      C        C      Y     2     0.000  0.997  256  
Mercury_7186568      B        B      Y     2     0.000  0.997  256  
Mercury_402216       B        B      Y     2     0.000  0.997  256  
Mercury_404894       A        A      Y     2     0.000  0.997  256  
MCAS_2002_8_11       C        C      Y     2     0.000  0.997  256  
Mercury_SC_405086    B        B      Y     0     0.000  1.000  256  
Mercury_SC_408324    D        D      Y     2     0.000  0.997  256  
Mercury_7218820      D        B      N     2     0.000  0.997  256  
Mercury_412202       D        B      N     2     0.000  0.997  256  
Mercury_SC_409139    D        C      N     2     0.000  0.997  256  
Mercury_400687       B        B      Y     2     0.000  0.997  256  
Mercury_7171605      D        D      Y     2     0.000  0.997  256  
Mercury_7210245      C        C      Y     2     0.000  0.997  256  
AKDE&ED_2008_4_25    A        A      Y     2     0.000  0.997  256  
AKDE&ED_2008_4_19    C        C      Y     17    0.008  0.973  256  
Mercury_SC_400402    C        A      N     2     0.000  0.997  256  
Mercury_7234308      A        A      Y     2     0.000  0.997  256  
ACTAAP_2014_5_8      B        B      Y     2     0.000  0.997  256  
Mercury_400407       B        B      Y     2     0.000  0.997  256  
Mercury_7116288      C        C      Y     2     0.000  0.997  256  
MCAS_2004_9_15-v1    B        B      Y     2     0.000  0.997  256  
NYSEDREGENTS_2015_4_26 C        C      Y     2     0.000  0.997  256  
Mercury_SC_401620    A        A      Y     2     0.000  0.997  256  
Mercury_400877       C        C      Y     2     0.000  0.997  256  
Mercury_7174213      D        D      Y     2     0.000  0.997  256  
NYSEDREGENTS_2008_8_34 B        B      Y     236   0.008  0.447  256  
Mercury_7212398      B        B      Y     2     0.000  0.997  256  
Mercury_SC_401290    C        C      Y     2     0.000  0.997  256  
Mercury_SC_402120    C        C      Y     2     0.000  0.997  256  
Mercury_184975       C        C      Y     2     0.000  0.997  256  
Mercury_SC_400578    B        A      N     2     0.000  0.997  256  
MCAS_2001_8_4        D        D      Y     2     0.000  0.997  256  
MCAS_2003_5_33       C        C      Y     2     0.000  0.997  256  
Mercury_7068513      A        A      Y     2     0.000  0.997  256  
AKDE&ED_2008_4_26    B        C      N     103   0.008  0.839  256  
Mercury_7235638      C        C      Y     2     0.000  0.997  256  
MDSA_2009_5_20       B        B      Y     0     0.000  1.000  256  
Mercury_178325       D        C      N     2     0.000  0.997  256  
Mercury_7212678      A        A      Y     2     0.000  0.997  256  
TAKS_2009_8_32       A        A      Y     2     0.000  0.997  256  
Mercury_412681       C        C      Y     2     0.000  0.997  256  
Mercury_400440       D        D      Y     2     0.000  0.997  256  
Mercury_SC_416529    B        B      Y     2     0.000  0.997  256  
MCAS_2006_8_1        D        D      Y     2     0.000  0.997  256  
TIMSS_2003_8_pg80    C        C      Y     2     0.000  0.997  256
```

### Pasted aggregate JSON (`results/llada_15_arcc_smoke64_direct_results.json`)

```json
{
  "n_trajectories": 64,
  "n_scored_for_accuracy": 64,
  "accuracy": 0.875,
  "coverage": 1.0,
  "conditional_accuracy": 0.875,
  "mean_flip_rate": 0.0004,
  "mean_rationalization_score": 0.9856,
  "mean_gated_rationalization_score": 0.9707,
  "mean_early_factor": 0.9711,
  "mean_post_change_rate": 1.0,
  "mean_length_growth": 0.9856,
  "early_convergence_fraction": 0.9688,
  "disagreement_rate": 1.0,
  "sbw_lockin_rate": 0.1094,
  "mean_convergence_step": 7.4062,
  "median_convergence_step": 2.0,
  "mean_convergence_fraction": 0.0289,
  "median_convergence_fraction": 0.0078,
  "mean_answer_commitment_step": 2.0,
  "mean_stability_commitment_lag": -5.4062,
  "mean_unwritten_reasoning_at_convergence": 0.9784,
  "mean_order_gap": -0.4965
}
```

## Dream-v0-Instruct-7B — GSM8K 64-item numeric suite (smoke clean (62/64 decided))

- **Date (UTC):** 2026-11-01
- **Job ID:** `dream_gsm8k_smoke64`
- **Git revision (`git rev-parse HEAD` at run):** `2278ff57e798f54820822d519d63f320e07828a3`
- **Environment:** conda env `cdr-gpu` (`torch 2.5.1+cu121`, `transformers 4.46.2`)
- **GPU:** index `0` (`CUDA_VISIBLE_DEVICES=0`, `NVIDIA GeForce RTX 4090`, `24210 MiB` free at launch)
- **`nvidia-smi` at launch:**
  ```
  index, name, memory.used [MiB], memory.total [MiB], utilization.gpu [%]
  0, NVIDIA GeForce RTX 4090, 2 MiB, 24564 MiB, 0 %
  1, NVIDIA GeForce RTX 4090, 2 MiB, 24564 MiB, 0 %
  2, NVIDIA GeForce RTX 4090, 2 MiB, 24564 MiB, 0 %
  3, NVIDIA GeForce RTX 4090, 2 MiB, 24564 MiB, 0 %
  ```
- **Sampler & adapter config:** `model_family=dream, temperature=0.2, top_p=0.95, alg=entropy, alg_temp=0.0, steps=256, max_new_tokens=256, force_deliberation_steps=0, frontier_width=None`
- **Artifacts:**
  - Input: `results/gsm8k_smoke64_input.json`
  - Fixed IDs: `results/smoke64_ids.json`
  - Trajectories: `results/dream_gsm8k_smoke64.jsonl`
  - Run log: `results/dream_gsm8k_smoke64_run_log.json`
  - Scored output: `results/dream_gsm8k_smoke64_results.json`
- **Undecided count:** **2 undecided** (62 / 64 non-null at final step).

### Pasted `traj-eval run` output

```
=== traj-eval summary ===
Trajectories:            64
Accuracy (final):        0.7656
Mean flip rate:          0.0602
Mean rationalization:    0.1550
  [components] early=0.3583 post_change=0.0277 len_growth=0.0032 gated=0.0018
Early convergence frac:  0.0000
Mean convergence step:   161.32
Median convergence step: 159.50
Mean convergence frac:   0.6302
Mean T_commit step:      161.68
Mean lag (T_comm-T_stab):0.35
Mean unwritten @ T_stab: 0.0016

Per-trajectory:
id                   final    gold   corr  conv  flip   rat    steps
--------------------------------------------------------------------
gsm8k_test_0         None     18                 0.000  0.000  256  
gsm8k_test_1         3        3      Y     126   0.000  0.203  256  
gsm8k_test_2         65000    70000  N     142   0.034  0.178  256  
gsm8k_test_3         540      540    Y     108   0.013  0.231  256  
gsm8k_test_4         20       20     Y     69    0.005  0.292  256  
gsm8k_test_5         64       64     Y     222   0.029  0.053  256  
gsm8k_test_6         260      260    Y     99    0.013  0.249  256  
gsm8k_test_7         120      160    N     254   0.667  0.003  256  
gsm8k_test_8         135      45     N     199   0.035  0.089  256  
gsm8k_test_9         460      460    Y     138   0.017  0.188  256  
gsm8k_test_10        366      366    Y     110   0.014  0.236  256  
gsm8k_test_11        694      694    Y     143   0.018  0.177  256  
gsm8k_test_12        12       13     N     90    0.006  0.269  256  
gsm8k_test_13        27       18     N     207   0.020  0.077  256  
gsm8k_test_14        60       60     Y     224   0.031  0.063  256  
gsm8k_test_15        125      125    Y     158   0.020  0.153  256  
gsm8k_test_16        230      230    Y     161   0.049  0.148  256  
gsm8k_test_17        57500    57500  Y     204   0.073  0.081  256  
gsm8k_test_18        7        7      Y     157   0.000  0.155  256  
gsm8k_test_19        3        6      N     124   0.000  0.206  256  
gsm8k_test_20        14       15     N     151   0.009  0.164  256  
gsm8k_test_21        2        14     N     161   0.000  0.148  256  
gsm8k_test_22        7        7      Y     160   0.000  0.155  256  
gsm8k_test_23        8        8      Y     129   0.000  0.198  256  
gsm8k_test_24        26       26     Y     208   0.021  0.075  256  
gsm8k_test_25        2        2      Y     214   0.000  0.076  256  
gsm8k_test_26        243      243    Y     238   0.105  0.052  256  
gsm8k_test_27        16       16     Y     154   0.010  0.159  256  
gsm8k_test_28        15       25     N     203   0.019  0.091  256  
gsm8k_test_29        104      104    Y     125   0.015  0.205  256  
gsm8k_test_30        109      109    Y     254   0.500  0.404  256  
gsm8k_test_31        80       80     Y     212   0.061  0.069  256  
gsm8k_test_32        35       35     Y     60    0.005  0.309  256  
gsm8k_test_33        70       70     Y     159   0.010  0.156  256  
gsm8k_test_34        23       23     Y     90    0.006  0.263  256  
gsm8k_test_35        9        9      Y     75    0.000  0.286  256  
gsm8k_test_36        75       75     Y     179   0.013  0.126  256  
gsm8k_test_37        13       2      N     183   0.014  0.114  256  
gsm8k_test_38        10       10     Y     203   0.019  0.083  256  
gsm8k_test_39        18       18     Y     176   0.013  0.125  256  
gsm8k_test_40        8        8      Y     74    0.000  0.284  256  
gsm8k_test_41        200      200    Y     196   0.032  0.094  256  
gsm8k_test_42        26       26     Y     111   0.007  0.230  256  
gsm8k_test_43        48       48     Y     255   1.000  0.002  256  
gsm8k_test_44        20       20     Y     223   0.081  0.052  256  
gsm8k_test_45        104      104    Y     252   0.417  0.141  256  
gsm8k_test_46        277      163    N     92    0.012  0.259  256  
gsm8k_test_47        800      800    Y     148   0.018  0.173  256  
gsm8k_test_48        8        8      Y     101   0.000  0.258  256  
gsm8k_test_49        30       30     Y     160   0.010  0.150  256  
gsm8k_test_50        294      294    Y     149   0.018  0.167  256  
gsm8k_test_51        5        5      Y     104   0.000  0.255  256  
gsm8k_test_52        15       15     Y     181   0.013  0.117  256  
gsm8k_test_53        40       40     Y     210   0.022  0.081  256  
gsm8k_test_54        40       40     Y     127   0.008  0.205  256  
gsm8k_test_55        14       14     Y     129   0.008  0.198  256  
gsm8k_test_56        3        3      Y     162   0.000  0.147  256  
gsm8k_test_57        83       83     Y     160   0.010  0.150  256  
gsm8k_test_58        47       57     N     190   0.015  0.110  256  
gsm8k_test_59        187      187    Y     116   0.014  0.219  256  
gsm8k_test_60        17       17     Y     196   0.046  0.094  256  
gsm8k_test_61        1430     1430   Y     151   0.028  0.164  256  
gsm8k_test_62        3750     25000  N     246   0.231  0.060  256  
gsm8k_test_63        None     1596               0.000  0.000  256
```

### Pasted aggregate JSON (`results/dream_gsm8k_smoke64_results.json`)

```json
{
  "n_trajectories": 64,
  "n_scored_for_accuracy": 62,
  "accuracy": 0.7656,
  "coverage": 0.9688,
  "conditional_accuracy": 0.7903,
  "mean_flip_rate": 0.0602,
  "mean_rationalization_score": 0.155,
  "mean_gated_rationalization_score": 0.0018,
  "mean_early_factor": 0.3583,
  "mean_post_change_rate": 0.0277,
  "mean_length_growth": 0.0032,
  "early_convergence_fraction": 0.0,
  "disagreement_rate": 0.4062,
  "sbw_lockin_rate": 0.0,
  "mean_convergence_step": 161.3226,
  "median_convergence_step": 159.5,
  "mean_convergence_fraction": 0.6302,
  "median_convergence_fraction": 0.623,
  "mean_answer_commitment_step": 161.6774,
  "mean_stability_commitment_lag": 0.3548,
  "mean_unwritten_reasoning_at_convergence": 0.0016,
  "mean_order_gap": 0.3165
}
```

## LLaDA-8B-Instruct — GSM8K 5-item numeric smoke (smoke clean (5/5 decided))

- **Date (UTC):** 2026-11-01
- **Job ID:** `llada_instruct_gsm8k_smoke5`
- **Git revision (`git rev-parse HEAD` at run):** `dee6813a09feefe9ca0538b44603decafe7c0e3b`
- **Environment:** conda env `cdr-gpu` (`torch 2.5.1+cu121`, `transformers 4.46.2`)
- **GPU:** index `3` (`CUDA_VISIBLE_DEVICES=3`, `NVIDIA GeForce RTX 4090`, `24210 MiB` free at launch)
- **`nvidia-smi` at launch:**
  ```
  index, name, memory.used [MiB], memory.total [MiB], utilization.gpu [%]
  0, NVIDIA GeForce RTX 4090, 2 MiB, 24564 MiB, 0 %
  1, NVIDIA GeForce RTX 4090, 2 MiB, 24564 MiB, 0 %
  2, NVIDIA GeForce RTX 4090, 2 MiB, 24564 MiB, 0 %
  3, NVIDIA GeForce RTX 4090, 2 MiB, 24564 MiB, 0 %
  ```
- **Sampler & adapter config:** `model_family=llada-instruct, temperature=0.0, remasking=low_confidence, steps=256, gen_length=256, block_length=256, confidence_eos_eot_inf=True, frontier_width=None`
- **Artifacts:**
  - Input: `results/dream_gsm8k_smoke5_input.json`
  - Fixed IDs: `results/llada_instruct_gsm8k_smoke5_ids.json`
  - Trajectories: `results/llada_instruct_gsm8k_smoke5.jsonl`
  - Run log: `results/llada_instruct_gsm8k_smoke5_run_log.json`
  - Scored output: `results/llada_instruct_gsm8k_smoke5_results.json`
- **Undecided count:** **0 undecided** (5 / 5 non-null at final step).

### Pasted `traj-eval run` output

```
=== traj-eval summary ===
Trajectories:            5
Accuracy (final):        0.8000
Mean flip rate:          0.0133
Mean rationalization:    0.7861
  [components] early=0.5766 post_change=1.0000 len_growth=0.7772 gated=0.5579
Early convergence frac:  0.2000
Mean convergence step:   108.40
Median convergence step: 89.00
Mean convergence frac:   0.4234
Mean T_commit step:      108.40
Mean lag (T_comm-T_stab):0.00
Mean unwritten @ T_stab: 0.5463

Per-trajectory:
id                   final    gold   corr  conv  flip   rat    steps
--------------------------------------------------------------------
gsm8k_test_0         18       18     Y     68    0.004  0.894  256  
gsm8k_test_1         3        3      Y     159   0.000  0.663  256  
gsm8k_test_2         125000   70000  N     89    0.013  0.861  256  
gsm8k_test_3         540      540    Y     213   0.045  0.533  256  
gsm8k_test_4         20       20     Y     13    0.004  0.980  256
```

### Pasted aggregate JSON (`results/llada_instruct_gsm8k_smoke5_results.json`)

```json
{
  "n_trajectories": 5,
  "n_scored_for_accuracy": 5,
  "accuracy": 0.8,
  "coverage": 1.0,
  "conditional_accuracy": 0.8,
  "mean_flip_rate": 0.0133,
  "mean_rationalization_score": 0.7861,
  "mean_gated_rationalization_score": 0.5579,
  "mean_early_factor": 0.5766,
  "mean_post_change_rate": 1.0,
  "mean_length_growth": 0.7772,
  "early_convergence_fraction": 0.2,
  "disagreement_rate": 1.0,
  "sbw_lockin_rate": 0.0,
  "mean_convergence_step": 108.4,
  "median_convergence_step": 89.0,
  "mean_convergence_fraction": 0.4234,
  "median_convergence_fraction": 0.3477,
  "mean_answer_commitment_step": 108.4,
  "mean_stability_commitment_lag": 0.0,
  "mean_unwritten_reasoning_at_convergence": 0.5463,
  "mean_order_gap": -0.0578
}
```


## LLaDA-1.5 — GSM8K 5-item numeric smoke (smoke clean (5/5 decided))

- **Date (UTC):** 2026-11-01
- **Job ID:** `llada_15_gsm8k_smoke5`
- **Git revision (`git rev-parse HEAD` at run):** `a76e1e9b653338cf16b48418d836411ea7a7dcdd`
- **Environment:** conda env `cdr-gpu` (`torch 2.5.1+cu121`, `transformers 4.46.2`)
- **GPU:** index `3` (`CUDA_VISIBLE_DEVICES=3`, `NVIDIA GeForce RTX 4090`, `22780 MiB` free at launch)
- **`nvidia-smi` at launch:**
  ```
  index, name, memory.used [MiB], memory.total [MiB], utilization.gpu [%]
  0, NVIDIA GeForce RTX 4090, 2 MiB, 24564 MiB, 0 %
  1, NVIDIA GeForce RTX 4090, 2 MiB, 24564 MiB, 0 %
  2, NVIDIA GeForce RTX 4090, 2 MiB, 24564 MiB, 0 %
  3, NVIDIA GeForce RTX 4090, 2 MiB, 24564 MiB, 0 %
  ```
- **Sampler & adapter config:** `model_family=llada-1.5, temperature=0.0, remasking=low_confidence, steps=256, gen_length=256, block_length=256, confidence_eos_eot_inf=True, frontier_width=None`
- **Artifacts:**
  - Input: `results/dream_gsm8k_smoke5_input.json`
  - Fixed IDs: `results/llada_15_gsm8k_smoke5_ids.json`
  - Trajectories: `results/llada_15_gsm8k_smoke5.jsonl`
  - Run log: `results/llada_15_gsm8k_smoke5_run_log.json`
  - Scored output: `results/llada_15_gsm8k_smoke5_results.json`
- **Undecided count:** **0 undecided** (5 / 5 non-null at final step).

### Pasted `traj-eval run` output

```
=== traj-eval summary ===
Trajectories:            5
Accuracy (final):        0.8000
Mean flip rate:          0.0042
Mean rationalization:    0.8670
  [components] early=0.7242 post_change=1.0000 len_growth=0.8864 gated=0.7121
Early convergence frac:  0.4000
Mean convergence step:   70.60
Median convergence step: 53.00
Mean convergence frac:   0.2758
Mean T_commit step:      70.60
Mean lag (T_comm-T_stab):0.00
Mean unwritten @ T_stab: 0.7237

Per-trajectory:
id                   final    gold   corr  conv  flip   rat    steps
--------------------------------------------------------------------
gsm8k_test_0         18       18     Y     53    0.004  0.917  256  
gsm8k_test_1         3        3      Y     174   0.000  0.615  256  
gsm8k_test_2         105000   70000  N     83    0.013  0.870  256  
gsm8k_test_3         540      540    Y     30    0.000  0.953  256  
gsm8k_test_4         20       20     Y     13    0.004  0.980  256
```

### Pasted aggregate JSON (`results/llada_15_gsm8k_smoke5_results.json`)

```json
{
  "n_trajectories": 5,
  "n_scored_for_accuracy": 5,
  "accuracy": 0.8,
  "coverage": 1.0,
  "conditional_accuracy": 0.8,
  "mean_flip_rate": 0.0042,
  "mean_rationalization_score": 0.867,
  "mean_gated_rationalization_score": 0.7121,
  "mean_early_factor": 0.7242,
  "mean_post_change_rate": 1.0,
  "mean_length_growth": 0.8864,
  "early_convergence_fraction": 0.4,
  "disagreement_rate": 1.0,
  "sbw_lockin_rate": 0.0,
  "mean_convergence_step": 70.6,
  "median_convergence_step": 53.0,
  "mean_convergence_fraction": 0.2758,
  "median_convergence_fraction": 0.207,
  "mean_answer_commitment_step": 70.6,
  "mean_stability_commitment_lag": 0.0,
  "mean_unwritten_reasoning_at_convergence": 0.7237,
  "mean_order_gap": -0.2203
}
```


## LLaDA-8B-Instruct — GSM8K 64-item numeric suite (smoke clean (64/64 decided))

- **Date (UTC):** 2026-11-01
- **Job ID:** `llada_instruct_gsm8k_smoke64`
- **Git revision (`git rev-parse HEAD` at run):** `a76e1e9b653338cf16b48418d836411ea7a7dcdd`
- **Environment:** conda env `cdr-gpu` (`torch 2.5.1+cu121`, `transformers 4.46.2`)
- **GPU:** index `3` (`CUDA_VISIBLE_DEVICES=3`, `NVIDIA GeForce RTX 4090`, `24210 MiB` free at launch)
- **`nvidia-smi` at launch:**
  ```
  index, name, memory.used [MiB], memory.total [MiB], utilization.gpu [%]
  0, NVIDIA GeForce RTX 4090, 2 MiB, 24564 MiB, 0 %
  1, NVIDIA GeForce RTX 4090, 2 MiB, 24564 MiB, 0 %
  2, NVIDIA GeForce RTX 4090, 2 MiB, 24564 MiB, 0 %
  3, NVIDIA GeForce RTX 4090, 2 MiB, 24564 MiB, 0 %
  ```
- **Sampler & adapter config:** `model_family=llada-instruct, temperature=0.0, remasking=low_confidence, steps=256, gen_length=256, block_length=256, confidence_eos_eot_inf=True, frontier_width=None`
- **Artifacts:**
  - Input: `results/gsm8k_smoke64_input.json`
  - Fixed IDs: `results/smoke64_ids.json`
  - Trajectories: `results/llada_instruct_gsm8k_smoke64.jsonl`
  - Run log: `results/llada_instruct_gsm8k_smoke64_run_log.json`
  - Scored output: `results/llada_instruct_gsm8k_smoke64_results.json`
- **Undecided count:** **0 undecided** (64 / 64 non-null at final step).

### Pasted `traj-eval run` output

```
=== traj-eval summary ===
Trajectories:            64
Accuracy (final):        0.6562
Mean flip rate:          0.0143
Mean rationalization:    0.8473
  [components] early=0.6940 post_change=1.0000 len_growth=0.8484 gated=0.6844
Early convergence frac:  0.4844
Mean convergence step:   78.34
Median convergence step: 53.50
Mean convergence frac:   0.3060
Mean T_commit step:      78.34
Mean lag (T_comm-T_stab):0.00
Mean unwritten @ T_stab: 0.6911

Per-trajectory:
id                   final    gold   corr  conv  flip   rat    steps
--------------------------------------------------------------------
gsm8k_test_0         18       18     Y     68    0.004  0.894  256  
gsm8k_test_1         3        3      Y     159   0.000  0.663  256  
gsm8k_test_2         125000   70000  N     89    0.013  0.861  256  
gsm8k_test_3         540      540    Y     213   0.045  0.533  256  
gsm8k_test_4         20       20     Y     13    0.004  0.980  256  
gsm8k_test_5         64       64     Y     39    0.004  0.939  256  
gsm8k_test_6         260      260    Y     231   0.077  0.460  256  
gsm8k_test_7         100      160    N     43    0.008  0.933  256  
gsm8k_test_8         370      45     N     79    0.008  0.877  256  
gsm8k_test_9         460      460    Y     20    0.008  0.969  256  
gsm8k_test_10        366      366    Y     139   0.017  0.703  256  
gsm8k_test_11        694      694    Y     200   0.010  0.530  256  
gsm8k_test_12        15       13     N     53    0.004  0.917  256  
gsm8k_test_13        18       18     Y     84    0.005  0.869  256  
gsm8k_test_14        60       60     Y     31    0.004  0.952  256  
gsm8k_test_15        125      125    Y     15    0.008  0.977  256  
gsm8k_test_16        230      230    Y     221   0.056  0.489  256  
gsm8k_test_17        92500    57500  N     62    0.005  0.903  256  
gsm8k_test_18        14       7      N     47    0.005  0.927  256  
gsm8k_test_19        6        6      Y     11    0.000  0.983  256  
gsm8k_test_20        18       15     N     72    0.004  0.887  256  
gsm8k_test_21        10       14     N     232   0.036  0.460  256  
gsm8k_test_22        7        7      Y     23    0.000  0.964  256  
gsm8k_test_23        8        8      Y     39    0.000  0.939  256  
gsm8k_test_24        26       26     Y     32    0.004  0.950  256  
gsm8k_test_25        2        2      Y     15    0.000  0.977  256  
gsm8k_test_26        242      243    N     211   0.043  0.510  256  
gsm8k_test_27        16       16     Y     67    0.000  0.895  256  
gsm8k_test_28        25       25     Y     31    0.004  0.952  256  
gsm8k_test_29        104      104    Y     45    0.009  0.930  256  
gsm8k_test_30        65       109    N     57    0.004  0.911  256  
gsm8k_test_31        80       80     Y     69    0.004  0.892  256  
gsm8k_test_32        35       35     Y     29    0.004  0.955  256  
gsm8k_test_33        70       70     Y     21    0.004  0.967  256  
gsm8k_test_34        23       23     Y     178   0.013  0.590  256  
gsm8k_test_35        9        9      Y     130   0.000  0.797  256  
gsm8k_test_36        75       75     Y     93    0.000  0.855  256  
gsm8k_test_37        1        2      N     16    0.000  0.975  256  
gsm8k_test_38        12       10     N     54    0.004  0.916  256  
gsm8k_test_39        36       18     N     24    0.004  0.963  256  
gsm8k_test_40        8        8      Y     17    0.000  0.973  256  
gsm8k_test_41        200      200    Y     4     0.000  0.994  256  
gsm8k_test_42        26       26     Y     35    0.000  0.945  256  
gsm8k_test_43        200      48     N     76    0.010  0.881  256  
gsm8k_test_44        20       20     Y     8     0.004  0.988  256  
gsm8k_test_45        84       104    N     181   0.013  0.592  256  
gsm8k_test_46        243      163    N     18    0.008  0.972  256  
gsm8k_test_47        800      800    Y     31    0.004  0.952  256  
gsm8k_test_48        8        8      Y     89    0.000  0.861  256  
gsm8k_test_49        30       30     Y     83    0.005  0.870  256  
gsm8k_test_50        392      294    N     108   0.011  0.831  256  
gsm8k_test_51        5        5      Y     4     0.000  0.994  256  
gsm8k_test_52        18       15     N     20    0.004  0.969  256  
gsm8k_test_53        120      40     N     208   0.008  0.533  256  
gsm8k_test_54        40       40     Y     51    0.005  0.920  256  
gsm8k_test_55        14       14     Y     251   0.326  0.410  256  
gsm8k_test_56        3        3      Y     25    0.000  0.961  256  
gsm8k_test_57        153      83     N     72    0.011  0.887  256  
gsm8k_test_58        57       57     Y     64    0.004  0.900  256  
gsm8k_test_59        187      187    Y     212   0.053  0.504  256  
gsm8k_test_60        17       17     Y     49    0.004  0.923  256  
gsm8k_test_61        1330     1430   N     9     0.004  0.986  256  
gsm8k_test_62        25000    25000  Y     17    0.004  0.973  256  
gsm8k_test_63        1680     1596   N     127   0.012  0.764  256
```

### Pasted aggregate JSON (`results/llada_instruct_gsm8k_smoke64_results.json`)

```json
{
  "n_trajectories": 64,
  "n_scored_for_accuracy": 64,
  "accuracy": 0.6562,
  "coverage": 1.0,
  "conditional_accuracy": 0.6562,
  "mean_flip_rate": 0.0143,
  "mean_rationalization_score": 0.8473,
  "mean_gated_rationalization_score": 0.6844,
  "mean_early_factor": 0.694,
  "mean_post_change_rate": 1.0,
  "mean_length_growth": 0.8484,
  "early_convergence_fraction": 0.4844,
  "disagreement_rate": 1.0,
  "sbw_lockin_rate": 0.1094,
  "mean_convergence_step": 78.3438,
  "median_convergence_step": 53.5,
  "mean_convergence_fraction": 0.306,
  "median_convergence_fraction": 0.209,
  "mean_answer_commitment_step": 78.3438,
  "mean_stability_commitment_lag": 0.0,
  "mean_unwritten_reasoning_at_convergence": 0.6911,
  "mean_order_gap": -0.1884
}
```

## LLaDA-8B-Instruct — ARC-C 64-item canonical reasoning suite (smoke clean (64/64 decided))

- **Date (UTC):** 2026-11-01
- **Job ID:** `llada_instruct_arcc_smoke64_reasoning`
- **Git revision (`git rev-parse HEAD` at run):** `0870b06e2d055dfde520c6e5a5be690535e52fab`
- **Environment:** conda env `cdr-gpu` (`torch 2.5.1+cu121`, `transformers 4.46.2`)
- **GPU:** index `3` (`CUDA_VISIBLE_DEVICES=3`, `NVIDIA GeForce RTX 4090`, `24210 MiB` free at launch)
- **`nvidia-smi` at launch:**
  ```
  index, name, memory.used [MiB], memory.total [MiB], utilization.gpu [%]
  0, NVIDIA GeForce RTX 4090, 2 MiB, 24564 MiB, 0 %
  1, NVIDIA GeForce RTX 4090, 2 MiB, 24564 MiB, 0 %
  2, NVIDIA GeForce RTX 4090, 2 MiB, 24564 MiB, 0 %
  3, NVIDIA GeForce RTX 4090, 2 MiB, 24564 MiB, 0 %
  ```
- **Sampler & adapter config:** `model_family=llada-instruct, temperature=0.0, remasking=low_confidence, steps=256, gen_length=256, block_length=256, confidence_eos_eot_inf=True, frontier_width=None`
- **Artifacts:**
  - Input: `results/arcc_smoke64_reasoning_input.json`
  - Fixed IDs: `results/arcc_smoke64_ids.json`
  - Trajectories: `results/llada_instruct_arcc_smoke64_reasoning.jsonl`
  - Run log: `results/llada_instruct_arcc_smoke64_reasoning_run_log.json`
  - Scored output: `results/llada_instruct_arcc_smoke64_reasoning_results.json`
- **Undecided count:** **0 undecided** (64 / 64 non-null at final step).

### Pasted `traj-eval run` output

```
=== traj-eval summary ===
Trajectories:            64
Accuracy (final):        0.9062
Mean flip rate:          0.0014
Mean rationalization:    0.9464
  [components] early=0.8746 post_change=1.0000 len_growth=0.9828 gated=0.8732
Early convergence frac:  0.8281
Mean convergence step:   32.11
Median convergence step: 21.00
Mean convergence frac:   0.1254
Mean T_commit step:      32.31
Mean lag (T_comm-T_stab):0.20
Mean unwritten @ T_stab: 0.8492

Per-trajectory:
id                   final    gold   corr  conv  flip   rat    steps
--------------------------------------------------------------------
Mercury_7175875      C        C      Y     2     0.000  0.997  256  
Mercury_SC_409171    B        B      Y     41    0.000  0.936  256  
Mercury_SC_408547    C        C      Y     36    0.000  0.944  256  
Mercury_407327       D        D      Y     4     0.000  0.994  256  
MCAS_2006_9_44       D        D      Y     16    0.000  0.975  256  
Mercury_7270393      B        B      Y     21    0.000  0.967  256  
MCAS_2014_5_7        C        C      Y     5     0.000  0.992  256  
Mercury_7086660      C        C      Y     11    0.000  0.983  256  
Mercury_7168805      B        B      Y     5     0.000  0.992  256  
MCAS_2003_8_11       A        A      Y     1     0.000  0.998  256  
Mercury_7250058      C        B      N     6     0.000  0.991  256  
Mercury_7012740      A        A      Y     36    0.000  0.944  256  
Mercury_LBS10610     C        C      Y     19    0.000  0.970  256  
Mercury_SC_407400    C        C      Y     2     0.000  0.997  256  
Mercury_7212993      C        C      Y     24    0.000  0.963  256  
Mercury_SC_413240    A        A      Y     52    0.014  0.919  256  
Mercury_7186358      C        C      Y     4     0.000  0.994  256  
Mercury_7166425      B        B      Y     126   0.000  0.802  256  
MDSA_2007_8_3        A        A      Y     12    0.000  0.981  256  
Mercury_7094290      C        C      Y     12    0.000  0.981  256  
Mercury_7186568      B        B      Y     56    0.000  0.912  256  
Mercury_402216       B        B      Y     27    0.000  0.958  256  
Mercury_404894       A        A      Y     4     0.000  0.994  256  
MCAS_2002_8_11       C        C      Y     5     0.000  0.992  256  
Mercury_SC_405086    B        B      Y     21    0.000  0.967  256  
Mercury_SC_408324    D        D      Y     4     0.000  0.994  256  
Mercury_7218820      B        B      Y     7     0.000  0.989  256  
Mercury_412202       B        B      Y     69    0.000  0.892  256  
Mercury_SC_409139    C        C      Y     5     0.000  0.992  256  
Mercury_400687       B        B      Y     32    0.000  0.950  256  
Mercury_7171605      D        D      Y     4     0.000  0.994  256  
Mercury_7210245      C        C      Y     4     0.000  0.994  256  
AKDE&ED_2008_4_25    A        A      Y     63    0.000  0.902  256  
AKDE&ED_2008_4_19    C        C      Y     46    0.000  0.928  256  
Mercury_SC_400402    A        A      Y     33    0.000  0.948  256  
Mercury_7234308      A        A      Y     81    0.000  0.873  256  
ACTAAP_2014_5_8      B        B      Y     6     0.000  0.991  256  
Mercury_400407       C        B      N     57    0.000  0.911  256  
Mercury_7116288      C        C      Y     19    0.000  0.970  256  
MCAS_2004_9_15-v1    B        B      Y     40    0.005  0.938  256  
NYSEDREGENTS_2015_4_26 C        C      Y     8     0.000  0.988  256  
Mercury_SC_401620    A        A      Y     37    0.000  0.942  256  
Mercury_400877       C        C      Y     160   0.000  0.685  256  
Mercury_7174213      D        D      Y     33    0.000  0.948  256  
NYSEDREGENTS_2008_8_34 B        B      Y     48    0.000  0.925  256  
Mercury_7212398      B        B      Y     17    0.000  0.973  256  
Mercury_SC_401290    C        C      Y     31    0.004  0.952  256  
Mercury_SC_402120    C        C      Y     131   0.007  0.793  256  
Mercury_184975       C        C      Y     13    0.000  0.980  256  
Mercury_SC_400578    B        A      N     29    0.004  0.955  256  
MCAS_2001_8_4        D        D      Y     56    0.000  0.912  256  
MCAS_2003_5_33       C        C      Y     15    0.000  0.977  256  
Mercury_7068513      A        A      Y     28    0.000  0.956  256  
AKDE&ED_2008_4_26    B        C      N     41    0.000  0.936  256  
Mercury_7235638      B        C      N     25    0.000  0.961  256  
MDSA_2009_5_20       B        B      Y     39    0.000  0.939  256  
Mercury_178325       D        C      N     9     0.000  0.986  256  
Mercury_7212678      A        A      Y     10    0.000  0.984  256  
TAKS_2009_8_32       A        A      Y     33    0.000  0.948  256  
Mercury_412681       C        C      Y     4     0.000  0.994  256  
Mercury_400440       D        D      Y     210   0.054  0.520  256  
Mercury_SC_416529    B        B      Y     7     0.000  0.989  256  
MCAS_2006_8_1        D        D      Y     4     0.000  0.994  256  
TIMSS_2003_8_pg80    C        C      Y     49    0.000  0.923  256
```

### Pasted aggregate JSON (`results/llada_instruct_arcc_smoke64_reasoning_results.json`)

```json
{
  "n_trajectories": 64,
  "n_scored_for_accuracy": 64,
  "accuracy": 0.9062,
  "coverage": 1.0,
  "conditional_accuracy": 0.9062,
  "mean_flip_rate": 0.0014,
  "mean_rationalization_score": 0.9464,
  "mean_gated_rationalization_score": 0.8732,
  "mean_early_factor": 0.8746,
  "mean_post_change_rate": 1.0,
  "mean_length_growth": 0.9828,
  "early_convergence_fraction": 0.8281,
  "disagreement_rate": 1.0,
  "sbw_lockin_rate": 0.0781,
  "mean_convergence_step": 32.1094,
  "median_convergence_step": 21.0,
  "mean_convergence_fraction": 0.1254,
  "median_convergence_fraction": 0.082,
  "mean_answer_commitment_step": 32.3125,
  "mean_stability_commitment_lag": 0.2031,
  "mean_unwritten_reasoning_at_convergence": 0.8492,
  "mean_order_gap": -0.3574
}
```

### Side-by-side rationalization components (`direct_64` vs `reasoning_64`)

| Component | `direct_64` | `reasoning_64` | Delta (`reasoning_64 - direct_64`) |
|---|---:|---:|---:|
| `early_factor` | 0.9826 | 0.8746 | -0.1080 |
| `post_change_rate` | 1.0000 | 1.0000 | +0.0000 |
| `length_growth` | 0.9929 | 0.9828 | -0.0101 |
| `score` | 0.9917 | 0.9464 | -0.0453 |
| `gated_score` | 0.9818 | 0.8732 | -0.1086 |

## LLaDA-1.5 — ARC-C 64-item canonical reasoning suite (smoke clean (64/64 decided))

- **Date (UTC):** 2026-11-07
- **Job ID:** `llada_15_arcc_smoke64_reasoning`
- **Git revision (`git rev-parse HEAD` at run):** `0870b06e2d055dfde520c6e5a5be690535e52fab`
- **Environment:** conda env `cdr-gpu` (`torch 2.5.1+cu121`, `transformers 4.46.2`)
- **GPU:** index `0` (`CUDA_VISIBLE_DEVICES=0`, `NVIDIA GeForce RTX 4090`, `24210 MiB` free at launch)
- **`nvidia-smi` at launch:**
  ```
  index, name, memory.used [MiB], memory.total [MiB], utilization.gpu [%]
  0, NVIDIA GeForce RTX 4090, 2 MiB, 24564 MiB, 0 %
  1, NVIDIA GeForce RTX 4090, 2 MiB, 24564 MiB, 0 %
  2, NVIDIA GeForce RTX 4090, 2 MiB, 24564 MiB, 0 %
  3, NVIDIA GeForce RTX 4090, 2 MiB, 24564 MiB, 0 %
  ```
- **Sampler & adapter config:** `model_family=llada-1.5, temperature=0.0, remasking=low_confidence, steps=256, gen_length=256, block_length=256, confidence_eos_eot_inf=True, frontier_width=None`
- **Artifacts:**
  - Input: `results/arcc_smoke64_reasoning_input.json`
  - Fixed IDs: `results/arcc_smoke64_ids.json`
  - Trajectories: `results/llada_15_arcc_smoke64_reasoning.jsonl`
  - Run log: `results/llada_15_arcc_smoke64_reasoning_run_log.json`
  - Scored output: `results/llada_15_arcc_smoke64_reasoning_results.json`
- **Undecided count:** **0 undecided** (64 / 64 non-null at final step).

### Pasted `traj-eval run` output

```
=== traj-eval summary ===
Trajectories:            64
Accuracy (final):        0.9219
Mean flip rate:          0.0003
Mean rationalization:    0.9652
  [components] early=0.9151 post_change=1.0000 len_growth=0.9958 gated=0.9146
Early convergence frac:  0.9062
Mean convergence step:   21.73
Median convergence step: 12.50
Mean convergence frac:   0.0849
Mean T_commit step:      21.73
Mean lag (T_comm-T_stab):0.00
Mean unwritten @ T_stab: 0.8906

Per-trajectory:
id                   final    gold   corr  conv  flip   rat    steps
--------------------------------------------------------------------
Mercury_7175875      C        C      Y     4     0.000  0.994  256  
Mercury_SC_409171    B        B      Y     8     0.000  0.988  256  
Mercury_SC_408547    C        C      Y     36    0.000  0.944  256  
Mercury_407327       D        D      Y     4     0.000  0.994  256  
MCAS_2006_9_44       D        D      Y     17    0.000  0.973  256  
Mercury_7270393      B        B      Y     17    0.000  0.973  256  
MCAS_2014_5_7        C        C      Y     4     0.000  0.994  256  
Mercury_7086660      C        C      Y     4     0.000  0.994  256  
Mercury_7168805      B        B      Y     5     0.000  0.992  256  
MCAS_2003_8_11       A        A      Y     1     0.000  0.998  256  
Mercury_7250058      C        B      N     5     0.000  0.992  256  
Mercury_7012740      A        A      Y     36    0.000  0.944  256  
Mercury_LBS10610     C        C      Y     19    0.000  0.970  256  
Mercury_SC_407400    C        C      Y     2     0.000  0.997  256  
Mercury_7212993      C        C      Y     24    0.000  0.963  256  
Mercury_SC_413240    A        A      Y     16    0.000  0.975  256  
Mercury_7186358      C        C      Y     4     0.000  0.994  256  
Mercury_7166425      B        B      Y     15    0.000  0.977  256  
MDSA_2007_8_3        A        A      Y     11    0.000  0.983  256  
Mercury_7094290      C        C      Y     5     0.000  0.992  256  
Mercury_7186568      B        B      Y     56    0.000  0.912  256  
Mercury_402216       B        B      Y     5     0.000  0.992  256  
Mercury_404894       A        A      Y     3     0.000  0.995  256  
MCAS_2002_8_11       C        C      Y     5     0.000  0.992  256  
Mercury_SC_405086    B        B      Y     21    0.000  0.967  256  
Mercury_SC_408324    D        D      Y     4     0.000  0.994  256  
Mercury_7218820      B        B      Y     12    0.000  0.981  256  
Mercury_412202       B        B      Y     69    0.000  0.892  256  
Mercury_SC_409139    C        C      Y     5     0.000  0.992  256  
Mercury_400687       B        B      Y     32    0.000  0.950  256  
Mercury_7171605      D        D      Y     4     0.000  0.994  256  
Mercury_7210245      C        C      Y     4     0.000  0.994  256  
AKDE&ED_2008_4_25    A        A      Y     36    0.000  0.944  256  
AKDE&ED_2008_4_19    C        C      Y     42    0.000  0.934  256  
Mercury_SC_400402    A        A      Y     33    0.000  0.948  256  
Mercury_7234308      A        A      Y     61    0.000  0.905  256  
ACTAAP_2014_5_8      B        B      Y     6     0.000  0.991  256  
Mercury_400407       C        B      N     57    0.000  0.911  256  
Mercury_7116288      C        C      Y     18    0.000  0.972  256  
MCAS_2004_9_15-v1    B        B      Y     37    0.004  0.942  256  
NYSEDREGENTS_2015_4_26 C        C      Y     6     0.000  0.991  256  
Mercury_SC_401620    A        A      Y     36    0.000  0.944  256  
Mercury_400877       C        C      Y     160   0.000  0.696  256  
Mercury_7174213      D        D      Y     4     0.000  0.994  256  
NYSEDREGENTS_2008_8_34 B        B      Y     4     0.000  0.994  256  
Mercury_7212398      B        B      Y     17    0.000  0.973  256  
Mercury_SC_401290    C        C      Y     29    0.004  0.955  256  
Mercury_SC_402120    C        C      Y     129   0.007  0.798  256  
Mercury_184975       C        C      Y     13    0.000  0.980  256  
Mercury_SC_400578    B        A      N     29    0.004  0.955  256  
MCAS_2001_8_4        D        D      Y     27    0.000  0.958  256  
MCAS_2003_5_33       C        C      Y     4     0.000  0.994  256  
Mercury_7068513      A        A      Y     27    0.000  0.958  256  
AKDE&ED_2008_4_26    B        C      N     41    0.000  0.936  256  
Mercury_7235638      C        C      Y     14    0.000  0.978  256  
MDSA_2009_5_20       B        B      Y     39    0.000  0.939  256  
Mercury_178325       D        C      N     5     0.000  0.992  256  
Mercury_7212678      A        A      Y     4     0.000  0.994  256  
TAKS_2009_8_32       A        A      Y     7     0.000  0.989  256  
Mercury_412681       C        C      Y     4     0.000  0.994  256  
Mercury_400440       D        D      Y     3     0.000  0.995  256  
Mercury_SC_416529    B        B      Y     7     0.000  0.989  256  
MCAS_2006_8_1        D        D      Y     4     0.000  0.994  256  
TIMSS_2003_8_pg80    C        C      Y     31    0.000  0.952  256
```

### Pasted aggregate JSON (`results/llada_15_arcc_smoke64_reasoning_results.json`)

```json
{
  "n_trajectories": 64,
  "n_scored_for_accuracy": 64,
  "accuracy": 0.9219,
  "coverage": 1.0,
  "conditional_accuracy": 0.9219,
  "mean_flip_rate": 0.0003,
  "mean_rationalization_score": 0.9652,
  "mean_gated_rationalization_score": 0.9146,
  "mean_early_factor": 0.9151,
  "mean_post_change_rate": 1.0,
  "mean_length_growth": 0.9958,
  "early_convergence_fraction": 0.9062,
  "disagreement_rate": 1.0,
  "sbw_lockin_rate": 0.0625,
  "mean_convergence_step": 21.7344,
  "median_convergence_step": 12.5,
  "mean_convergence_fraction": 0.0849,
  "median_convergence_fraction": 0.0488,
  "mean_answer_commitment_step": 21.7344,
  "mean_stability_commitment_lag": 0.0,
  "mean_unwritten_reasoning_at_convergence": 0.8906,
  "mean_order_gap": -0.3989
}
```

### Side-by-side rationalization components (`direct_64` vs `reasoning_64`)

| Component | `direct_64` | `reasoning_64` | Delta (`reasoning_64 - direct_64`) |
|---|---:|---:|---:|
| `early_factor` | 0.9711 | 0.9151 | -0.0560 |
| `post_change_rate` | 1.0000 | 1.0000 | +0.0000 |
| `length_growth` | 0.9856 | 0.9958 | +0.0102 |
| `score` | 0.9856 | 0.9652 | -0.0204 |
| `gated_score` | 0.9707 | 0.9146 | -0.0561 |

## LLaDA-1.5 — GSM8K 64-item numeric suite (smoke clean (64/64 decided))

- **Date (UTC):** 2026-11-07
- **Job ID:** `llada_15_gsm8k_smoke64`
- **Git revision (`git rev-parse HEAD` at run):** `0870b06e2d055dfde520c6e5a5be690535e52fab`
- **Environment:** conda env `cdr-gpu` (`torch 2.5.1+cu121`, `transformers 4.46.2`)
- **GPU:** index `1` (`CUDA_VISIBLE_DEVICES=1`, `NVIDIA GeForce RTX 4090`, `24210 MiB` free at launch)
- **`nvidia-smi` at launch:**
  ```
  index, name, memory.used [MiB], memory.total [MiB], utilization.gpu [%]
  0, NVIDIA GeForce RTX 4090, 2 MiB, 24564 MiB, 0 %
  1, NVIDIA GeForce RTX 4090, 2 MiB, 24564 MiB, 0 %
  2, NVIDIA GeForce RTX 4090, 2 MiB, 24564 MiB, 0 %
  3, NVIDIA GeForce RTX 4090, 2 MiB, 24564 MiB, 0 %
  ```
- **Sampler & adapter config:** `model_family=llada-1.5, temperature=0.0, remasking=low_confidence, steps=256, gen_length=256, block_length=256, confidence_eos_eot_inf=True, frontier_width=None`
- **Artifacts:**
  - Input: `results/gsm8k_smoke64_input.json`
  - Fixed IDs: `results/smoke64_ids.json`
  - Trajectories: `results/llada_15_gsm8k_smoke64.jsonl`
  - Run log: `results/llada_15_gsm8k_smoke64_run_log.json`
  - Scored output: `results/llada_15_gsm8k_smoke64_results.json`
- **Undecided count:** **0 undecided** (64 / 64 non-null at final step).

### Pasted `traj-eval run` output

```
=== traj-eval summary ===
Trajectories:            64
Accuracy (final):        0.6094
Mean flip rate:          0.0071
Mean rationalization:    0.8515
  [components] early=0.7181 post_change=0.9844 len_growth=0.8527 gated=0.7071
Early convergence frac:  0.5781
Mean convergence step:   72.17
Median convergence step: 47.00
Mean convergence frac:   0.2819
Mean T_commit step:      72.17
Mean lag (T_comm-T_stab):0.00
Mean unwritten @ T_stab: 0.7279

Per-trajectory:
id                   final    gold   corr  conv  flip   rat    steps
--------------------------------------------------------------------
gsm8k_test_0         18       18     Y     53    0.004  0.917  256  
gsm8k_test_1         3        3      Y     174   0.000  0.615  256  
gsm8k_test_2         105000   70000  N     83    0.013  0.870  256  
gsm8k_test_3         540      540    Y     30    0.000  0.953  256  
gsm8k_test_4         20       20     Y     13    0.004  0.980  256  
gsm8k_test_5         78       64     N     44    0.005  0.931  256  
gsm8k_test_6         260      260    Y     188   0.000  0.563  256  
gsm8k_test_7         120      160    N     25    0.008  0.961  256  
gsm8k_test_8         370      45     N     95    0.008  0.852  256  
gsm8k_test_9         460      460    Y     20    0.008  0.969  256  
gsm8k_test_10        366      366    Y     139   0.015  0.703  256  
gsm8k_test_11        694      694    Y     195   0.010  0.548  256  
gsm8k_test_12        15       13     N     47    0.004  0.927  256  
gsm8k_test_13        18       18     Y     72    0.005  0.887  256  
gsm8k_test_14        60       60     Y     31    0.004  0.952  256  
gsm8k_test_15        125      125    Y     15    0.008  0.977  256  
gsm8k_test_16        230      230    Y     221   0.056  0.490  256  
gsm8k_test_17        97500    57500  N     62    0.005  0.903  256  
gsm8k_test_18        14       7      N     47    0.005  0.927  256  
gsm8k_test_19        6        6      Y     36    0.000  0.944  256  
gsm8k_test_20        18       15     N     72    0.004  0.887  256  
gsm8k_test_21        15       14     N     177   0.006  0.613  256  
gsm8k_test_22        7        7      Y     18    0.000  0.972  256  
gsm8k_test_23        8        8      Y     39    0.000  0.939  256  
gsm8k_test_24        26       26     Y     17    0.004  0.973  256  
gsm8k_test_25        2        2      Y     15    0.000  0.977  256  
gsm8k_test_26        242      243    N     210   0.011  0.514  256  
gsm8k_test_27        16       16     Y     37    0.000  0.942  256  
gsm8k_test_28        25       25     Y     30    0.004  0.953  256  
gsm8k_test_29        104      104    Y     26    0.009  0.959  256  
gsm8k_test_30        69       109    N     93    0.004  0.855  256  
gsm8k_test_31        80       80     Y     59    0.004  0.908  256  
gsm8k_test_32        35       35     Y     21    0.004  0.967  256  
gsm8k_test_33        70       70     Y     18    0.004  0.972  256  
gsm8k_test_34        23       23     Y     183   0.014  0.579  256  
gsm8k_test_35        9        9      Y     76    0.000  0.881  256  
gsm8k_test_36        375      75     N     21    0.004  0.967  256  
gsm8k_test_37        1        2      N     16    0.000  0.975  256  
gsm8k_test_38        12       10     N     29    0.004  0.955  256  
gsm8k_test_39        36       18     N     33    0.004  0.948  256  
gsm8k_test_40        8        8      Y     16    0.000  0.975  256  
gsm8k_test_41        200      200    Y     4     0.000  0.994  256  
gsm8k_test_42        26       26     Y     32    0.000  0.950  256  
gsm8k_test_43        60       48     N     8     0.004  0.988  256  
gsm8k_test_44        2000     20     N     255   0.008  0.002  256  
gsm8k_test_45        84       104    N     159   0.010  0.654  256  
gsm8k_test_46        163      163    Y     51    0.008  0.920  256  
gsm8k_test_47        1400     800    N     24    0.008  0.963  256  
gsm8k_test_48        8        8      Y     54    0.000  0.916  256  
gsm8k_test_49        30       30     Y     68    0.004  0.894  256  
gsm8k_test_50        288      294    N     163   0.010  0.662  256  
gsm8k_test_51        5        5      Y     2     0.000  0.997  256  
gsm8k_test_52        12       15     N     47    0.004  0.927  256  
gsm8k_test_53        200      40     N     89    0.008  0.861  256  
gsm8k_test_54        40       40     Y     51    0.005  0.920  256  
gsm8k_test_55        14       14     Y     234   0.048  0.449  256  
gsm8k_test_56        3        3      Y     27    0.000  0.958  256  
gsm8k_test_57        163      83     N     72    0.008  0.887  256  
gsm8k_test_58        57       57     Y     32    0.004  0.950  256  
gsm8k_test_59        187      187    Y     212   0.053  0.507  256  
gsm8k_test_60        17       17     Y     49    0.004  0.923  256  
gsm8k_test_61        1330     1430   N     7     0.004  0.989  256  
gsm8k_test_62        25000    25000  Y     23    0.008  0.964  256  
gsm8k_test_63        1520     1596   N     160   0.012  0.645  256
```

### Pasted aggregate JSON (`results/llada_15_gsm8k_smoke64_results.json`)

```json
{
  "n_trajectories": 64,
  "n_scored_for_accuracy": 64,
  "accuracy": 0.6094,
  "coverage": 1.0,
  "conditional_accuracy": 0.6094,
  "mean_flip_rate": 0.0071,
  "mean_rationalization_score": 0.8515,
  "mean_gated_rationalization_score": 0.7071,
  "mean_early_factor": 0.7181,
  "mean_post_change_rate": 0.9844,
  "mean_length_growth": 0.8527,
  "early_convergence_fraction": 0.5781,
  "disagreement_rate": 0.9844,
  "sbw_lockin_rate": 0.1875,
  "mean_convergence_step": 72.1719,
  "median_convergence_step": 47.0,
  "mean_convergence_fraction": 0.2819,
  "median_convergence_fraction": 0.1836,
  "mean_answer_commitment_step": 72.1719,
  "mean_stability_commitment_lag": 0.0,
  "mean_unwritten_reasoning_at_convergence": 0.7279,
  "mean_order_gap": -0.2231
}
```
