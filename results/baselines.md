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

## Real checkpoints — not run

Chosen 17 Oct 2026. Text trio first, because Yeom et al. (arXiv:2608.05687) already published commitment-order numbers on these three and the paper has to be comparable. MMaDA is the later modality holdout from the research plan, not a substitute for LLaDA-1.5 in the first table.

| Checkpoint | Hugging Face id | Status |
|---|---|---|
| LLaDA-8B-Instruct | `GSAI-ML/LLaDA-8B-Instruct` | Adapter ready (`src/traj_eval/adapters/llada.py`). GPU run pending. |
| LLaDA-1.5 | `GSAI-ML/LLaDA-1.5` | Adapter ready (`src/traj_eval/adapters/llada.py`). GPU run pending. |
| Dream-v0-Instruct-7B | `Dream-org/Dream-v0-Instruct-7B` | 5-item ARC-C direct smoke clean (2026-10-29) + 5-item canonical reasoning smoke clean (2026-10-30). |
| MMaDA-8B-MixCoT (holdout) | `Gen-Verse/MMaDA-8B-MixCoT` | Not in the first suite. |

Why nothing was launched: all eight RTX A6000s on this machine were in use (about 43 GB of 49 GB, utilization 100%). The default PyTorch in the base environment is `2.11.0+cu130` and does not see the driver. A later run should use a CUDA 12.1 build, one GPU that is actually free, and a sampler config written down before the job starts.

### Smoke protocol, to run unchanged once a GPU is free

Pre-registered GSM8K protocol, not yet run; it is not the ARC-C smoke protocol below.

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

Per Issue #4 and `docs/CURSOR_FOLLOWUP_2026-10-30.md` §3b, ARC-C evaluation uses **Option (a)**: the full ARC-C reasoning condition uses the **canonical reasoning prompt** at both 5-item and 64-item scale, and the earlier 2–3-sentence `cot` prompt files (`results/dream_arcc_cot_smoke5_input.json`, `results/dream_arcc_cot_smoke5_ids.json`, `results/arcc_smoke64_cot_input.json`) are **retired and removed** so prompt wording is never swapped between 5-item and 64-item runs. Direct and reasoning conditions are always reported side by side, never pooled into a single accuracy number.

| Condition | Exact closing prompt instruction | 5-item files (`Mercury_7175875` .. `MCAS_2006_9_44`) | 64-item files (first 64 4-option A–D test items) |
|---|---|---|---|
| `direct` | `Respond in the form "Answer: <letter>" with the letter of the correct option.` | `results/dream_arcc_smoke5_input.json`, `results/dream_arcc_smoke5_ids.json`, `results/dream_arcc_smoke5.jsonl`, `results/dream_arcc_smoke5_run_log.json`, `results/dream_arcc_smoke5_results.json` | `results/arcc_smoke64_direct_input.json`, `results/arcc_smoke64_ids.json` |
| `reasoning` (canonical) | `Think step by step, then give your final answer in the form "Answer: <letter>".` | `results/dream_arcc_smoke5_reasoning_input.json`, `results/dream_arcc_smoke5_reasoning_ids.json`, `results/dream_arcc_smoke5_reasoning.jsonl`, `results/dream_arcc_smoke5_reasoning_run_log.json`, `results/dream_arcc_smoke5_reasoning_results.json` | `results/arcc_smoke64_reasoning_input.json`, `results/arcc_smoke64_ids.json` |

## Pre-registered Gated Sequence (`docs/CURSOR_FOLLOWUP_2026-10-30.md` §§5–6)

This list is a **gated pre-registration sequence**, executed one stage at a time by `scripts/gpu_job_queue.py` whenever a GPU with `>= 25 GB` free VRAM is available. Each stage opens its own PR against `master` and waits for review/merge before the next stage starts.

1. **Stage 1 — Dream-7B canonical reasoning-condition 5-item ARC-C smoke (COMPLETED, gate passed — see section below):**
   - Canonical prompt file: `results/dream_arcc_smoke5_reasoning_input.json`, `results/dream_arcc_smoke5_reasoning_ids.json` (closing line: `Think step by step, then give your final answer in the form "Answer: <letter>".`).
2. **Stage 2 — Dream-7B 64-item ARC-C paired direct + canonical reasoning suites:**
   - Pre-registered IDs: `results/arcc_smoke64_ids.json`
   - Direct input: `results/arcc_smoke64_direct_input.json`
   - Canonical reasoning input: `results/arcc_smoke64_reasoning_input.json` (exact same closing line as Stage 1: `Think step by step, then give your final answer in the form "Answer: <letter>".`).
3. **Stage 3 — Dream-7B GSM8K 5-item numeric smoke (`extract_gsm8k_verdict` + exemplar stripping):**
   - Inputs & IDs: `results/dream_gsm8k_smoke5_input.json`, `results/dream_gsm8k_smoke5_ids.json`
4. **Stage 4 — Dream-7B EOS-suppression K=64 5-item smoke (`--force-deliberation-steps 64`, named `EOS-suppression K=64` per `docs/CURSOR_FOLLOWUP_2026-10-30.md` §3c, separate from the post-stability protocol):**
   - Inputs & IDs: `results/dream_arcc_smoke5_input.json`, `results/dream_arcc_smoke5_ids.json`
   - Outputs: `results/dream_arcc_eos_suppression_k64_smoke5.{jsonl,run_log.json,results.json}`
5. **Stage 5 — LLaDA-8B-Instruct and LLaDA-1.5 5-item ARC-C validation smokes (direct + canonical reasoning):**
   - Run with `--model-family llada-instruct|llada-1.5` before any 64-item LLaDA runs.

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
- **Reasoning smoke gate (`CURSOR_FOLLOWUP_2026-10-30` §5):** `reasoning smoke clean` (commitment clocks on `5/5`, post-verdict change on `3`, pre-verdict rationale on `5`).

### Pasted `traj-eval run` output

```
=== traj-eval summary ===
Trajectories:            5
Accuracy (final):        0.8000
Mean flip rate:          0.0126
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
Mercury_7175875      C        C      Y     224   0.018  0.050  256  
Mercury_SC_409171    B        B      Y     165   0.019  0.142  256  
Mercury_SC_408547    C        C      Y     254   0.026  0.404  256  
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
  "mean_flip_rate": 0.0126,
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
