# Experimental design: traj-eval expansion (weeks 3-4)

Status: draft for review. Pre-registered before the expansion runs below. Written from `results/baselines.md` and the scored 64-item artifacts on `master`; no numbers here come from any other source.

## Research question

Do diffusion language models reason, or just denoise? Concretely: does the denoising trajectory carry information about how an answer was reached, beyond what final-answer accuracy shows?

## What is already measured (do not re-run)

64-item suites completed 2026-11-01:

- LLaDA-8B-Instruct (`GSAI-ML/LLaDA-8B-Instruct`): ARC-C direct + canonical reasoning, GSM8K. ARC-C reasoning 0.9062 vs direct 0.875.
- LLaDA-1.5 (`GSAI-ML/LLaDA-1.5`): same suites. ARC-C reasoning 0.9219 vs direct 0.875.
- Dream-v0-Instruct-7B (`Dream-org/Dream-v0-Instruct-7B`): ARC-C direct + canonical reasoning and GSM8K 64-item suites (2026-10-30 / 2026-11-01). ARC-C direct 0.8906 vs reasoning 0.8281.
- Synthetic v0.1 (5 items per archetype, seed 42): genuine, rationalization, and oscillation archetypes, all with final accuracy 1.0000; the trajectory metrics separate them where accuracy cannot.

## Expansion: what runs next

1. MMaDA-8B-MixCoT holdout (`Gen-Verse/MMaDA-8B-MixCoT`): ARC-C direct 64-item, ARC-C canonical reasoning 64-item, GSM8K direct + reasoning 64-item (first 64 questions in published test order; id list in `results/smoke64_ids.json`). Not run yet.
2. Multimodal checkable tasks (MathVista, ChartQA, CLEVR): pending the advisor decision on starting model and checkable tasks (open since Oct 29). No runs until that decision is made and the task set is pre-registered here.

## Metrics (main table only)

Validated metrics implemented in `src/traj_eval/metrics.py`: `verdict_convergence_step`, `flip_rate`, `step_consistency`, `trajectory_accuracy` with coverage and conditional accuracy, `rationalization_components` (Dream only, descriptive heuristic, not causal proof), `answer_commitment_step`, `stability_commitment_lag`, `unwritten_reasoning_at_convergence`, `forced_deliberation_delta` (only after the unrestricted smoke JSONL exists).

Explicitly excluded from the main table until verified (issue #5): SBW lock-in, CoTsel selection gaps, Pass@k, GRPO credit schemes (StepRS, ForkGRPO, DTCA, broadcast, pre/post-convergence credit). Five-item runs are pipeline checks only, never table evidence.

## Prompts and sampler (frozen)

- Canonical reasoning prompt and direct answer-first prompt, unchanged. Any new prompt is pre-registered with its own output files, never overwriting an existing condition.
- LLaDA family: temperature 0, EOS/EoT confidence suppression on, canvas length 256, denoising steps 256, the checkpoint released low-confidence remasking rule.
- Dream-v0-Instruct-7B: unrestricted sampler temperature 0.2, top_p 0.95, alg entropy, force_deliberation_steps 0 (registered in `scripts/gpu_job_queue.py`).
- Environment: pinned `transformers==4.46.2` CUDA environment (`cdr-gpu` on `titan3`, `.venv-gpu` on `ai-lab2`), one dedicated GPU per run, full run logs.

## Exact GPU runs (weeks 3-4)

| # | Run | Checkpoint | Task | Condition | Sampler |
|---|---|---|---|---|---|
| 1 | MMaDA ARC-C direct | MMaDA-8B-MixCoT | ARC-C 64-item | direct | temp 0, EOS suppression on |
| 2 | MMaDA ARC-C reasoning | MMaDA-8B-MixCoT | ARC-C 64-item | canonical reasoning | temp 0, EOS suppression on |
| 3 | MMaDA GSM8K direct | MMaDA-8B-MixCoT | GSM8K 64-item | direct | temp 0, EOS suppression on |
| 4 | MMaDA GSM8K reasoning | MMaDA-8B-MixCoT | GSM8K 64-item | canonical reasoning | temp 0, EOS suppression on |
| 5 | MMaDA forced-deliberation smoke | MMaDA-8B-MixCoT | ARC-C 64-item | reasoning + forced deliberation | only after runs 1-4 JSONL exists |

Each run: log every step to the traj-eval JSONL schema, score with `python -m traj_eval run`, paste the aggregate in `results/baselines.md` with git revision and exact sampler flags. No new features until the reasoning smoke gate passes (`check_reasoning_smoke_gate`).

## What would change my mind (falsifier)

- If the trajectory metrics do not separate conditions on the MMaDA holdout the way they do on the text trio (convergence step and flip rate show no systematic difference between direct and reasoning conditions while accuracy differs), the trajectory signal does not generalize, and the claim narrows to the measured checkpoints.
- If rationalization-style trajectories (early lock-in with post-change text growth) appear at the same rate under both conditions, the rationalization decomposition does not discriminate conditions, and it stays a descriptive diagnostic only.
- If the MMaDA holdout shows no trajectory information beyond final accuracy, the paper claim becomes: trajectory evaluation is informative on the measured text checkpoints only, and the multimodal extension is future work.
