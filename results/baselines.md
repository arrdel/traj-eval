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
| LLaDA-8B-Instruct | `GSAI-ML/LLaDA-8B-Instruct` | Not run. No adapter. Weights not downloaded by this repo. |
| LLaDA-1.5 | `GSAI-ML/LLaDA-1.5` | Not run. |
| Dream-v0-Instruct-7B | `Dream-org/Dream-v0-Instruct-7B` | Not run. |
| MMaDA-8B-MixCoT (holdout) | `Gen-Verse/MMaDA-8B-MixCoT` | Not in the first suite. |

Why nothing was launched: all eight RTX A6000s on this machine were in use (about 43 GB of 49 GB, utilization 100%). The default PyTorch in the base environment is `2.11.0+cu130` and does not see the driver. A later run should use a CUDA 12.1 build, one GPU that is actually free, and a sampler config written down before the job starts.

### Smoke protocol, to run unchanged once a GPU is free

Pre-register before looking at accuracy:

- Data: GSM8K test, first 64 questions in the published test order. Store the id list in `results/smoke64_ids.json` when the run starts.
- Canvas length 256, denoising steps 256, temperature 0.
- Remasking: the checkpoint's released low-confidence rule.
- EOS/EOT confidence suppression: on. Report it. Turning it off is a different experiment.
- Log every step to the traj-eval JSONL schema: parsed prediction, reasoning text, and, once the adapter can, whether the answer span is irreversibly committed.
- Score with `python -m traj_eval run`. Paste the aggregate here with the git revision and the exact sampler flags.

Forced-deliberation smoke, only after the unrestricted smoke JSONL exists: continue for 64 extra steps after the parsed verdict has been stable for 16 steps, answer span still remaskable. Delta is `forced_deliberation_delta` on the paired 64.

No accuracy number from Yeom et al., d1, or a model card belongs in this table.
