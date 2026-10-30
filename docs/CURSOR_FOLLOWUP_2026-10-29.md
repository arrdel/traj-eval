# Cursor Follow-up — Review of Oct 29–30 Work + Next Directions

**Read this first. It supersedes the sequencing in
`docs/CURSOR_HANDOFF_DREAM_SMOKE.md` section 4, but not its rules.
The null rule, the no-weights-in-tests rule, the lazy-import rule,
and the "small commits, baselines in the same commit as the run" rule
all still hold.**

Date: 2026-10-29 (review of commits through `5fb1735`)
Repo: `arrdel/traj-eval`, branch `master`
Reviewer: Muse (independent read-back from GitHub, not a re-run)

## 1. Evaluation: what Cursor did right

- **Smoke is genuinely clean.** Commit `7831ea7` (run at `a6c7f74`):
  38/38 tests passed after pull, fake smoke + demo clean, then the real
  Dream-v0-Instruct-7B ARC-C 5-item smoke on GPU 4 (RTX A6000,
  32.2 GB free at launch, `cdr-gpu`, torch 2.5.1+cu121,
  transformers 4.46.2, sampler temperature 0.2, top_p 0.95,
  alg entropy, alg_temp 0.0, steps 256, max_new_tokens 256).
  `len(history) == 256` on all 5 items, final verdict matched
  `sequences` verdict on all 5, 0 undecided. Artifacts, run log,
  scored results, and the pasted aggregate in `results/baselines.md`
  are all present and consistent. Keep doing exactly this level of
  logging.
- **The AutoModel fix (`a6c7f74`) was correct and minimal.**
  Dream registers under `AutoModel`, not `AutoModelForCausalLM`.
  No unrelated changes in that commit.
- **Commitment logging (`5fb1735`) was the right next code step,
  even though it jumped the old order.** The smoke trace itself
  demanded it (see section 2). `rationalization_components`,
  `answer_commitment_step` (T_commit), `stability_commitment_lag`,
  and `unwritten_reasoning_at_convergence` match the paper outline's
  method section (which already anticipated these fields), and the
  README metrics table was updated. The fake-history test
  (`test_history_to_trajectory_tracks_answer_first_backfill`) is
  good: it pins answer-first commitment with later rationale backfill.
- **Baselines discipline held.** No model claim was made from 5 items.
  The smoke section is labeled "pipeline check only" and the
  diagnostic note correctly refuses to read 0.395 as rationalization.

## 2. The finding that changes the plan

The smoke trace is degenerate, and that is the most important result
so far. Every item is identical in shape:

```
step 0: ''            prediction None
step 1: 'Answer'      prediction None
step 2: 'Answer:'     prediction None
step 3: 'Answer: C'   prediction C
steps 4-255: 'Answer: C' (unchanged, rest of canvas EOS, stripped)
```

Accuracy 1.0000, flip rate 0.0000, convergence step 3-4, and
`rationalization_score` 0.395 = `0.4 * early_factor` alone, because
`post_change_rate = 0.0` and `length_growth = 0.0`. There is no
reasoning trajectory in this condition at all. **Do NOT scale the
direct-answer condition to full ARC-C as the reasoning analysis.**
It would produce 1,000 more copies of this trace and tell us nothing
about reasoning regimes. The paper outline requires reporting the
three heuristic components, never the scalar alone, and this is why.

Also note: the smoke JSONL was written before `5fb1735`, so its steps
have no `answer_committed` / `committed_reasoning_fraction` fields.
The commitment metrics in `results/dream_arcc_smoke5_results.json`
are all null. They cannot be backfilled from the JSONL: the raw
history is not stored there. They need a fresh run.

## 3. Fixes first (no GPU needed, one small commit each) — COMPLETED (2026-10-30)

All 5 pre-GPU items have been implemented, tested (`56 passed`), and pushed:

1. **[x] Stale docstring (`5cce209`).** `src/traj_eval/adapters/dream.py` module
   docstring updated to describe `compute_step_commitments` and
   `history_to_trajectory`'s commitment fields (`answer_committed`,
   `committed_reasoning_fraction`), plus `frontier_width` and
   `force_deliberation_steps`.
2. **[x] Rationalization weight inconsistency (`521f692`).**
   `rationalization_components` no-post-steps branch updated from
   `0.3 * early_factor` to `0.4 * early_factor` (plus `gated_score`
   added alongside `score`) with unit test
   `test_rationalization_last_step_convergence_uses_0_4_weight`.
3. **[x] Answer-position detection (`5cce209`).** `compute_step_commitments`
   now uses 3-pass matching: (1) exact punctuation/whitespace-stripped
   token match scanning from the end, (2) substring fallback, (3)
   `content_pos[-1]` fallback, verified by
   `test_compute_step_commitments_exact_match_beats_trailing_substring`.
4. **[x] Relabel stale baselines section (`d32cd1e`).**
   Added explicit header `"Pre-registered GSM8K protocol, not yet run;
   it is not the ARC-C smoke protocol above."` in `results/baselines.md`.
5. **[x] Update old handoff's section 4 (`d32cd1e`).**
   `docs/CURSOR_HANDOFF_DREAM_SMOKE.md` §4 now points to this follow-up
   document.
6. **[x] Pre-register reasoning-condition 5-item smoke (`d32cd1e`).**
   `results/dream_arcc_smoke5_reasoning_input.json` and
   `results/dream_arcc_smoke5_reasoning_ids.json` are committed with the
   exact closing line specified in §4 below. Waiting on GPU reservation
   (`>= 25 GB` free VRAM) to execute the run.

## 4. Next GPU run: reasoning-condition 5-item smoke (the ONLY next experiment)

Do not run full ARC-C yet. Do not start LLaDA, MMaDA, GSM8K, or
forced deliberation yet.

**Pre-register, before running:**

- Same 5 ids, same order, same sampler as the direct-answer smoke:
  temperature 0.2, top_p 0.95, alg entropy, alg_temp 0.0,
  steps 256, max_new_tokens 256, `cdr-gpu`, one GPU with >= 25 GB
  free, `CUDA_VISIBLE_DEVICES` pinned and logged.
- New prompt template, fixed for all 5 items, question + A-D options
  as in `results/dream_arcc_smoke5_input.json`, closing line:
  `Think step by step, then give your final answer in the form
  "Answer: <letter>".`
- New files only, never overwrite the direct-answer artifacts:
  - Input: `results/dream_arcc_smoke5_reasoning_input.json`
  - Ids: `results/dream_arcc_smoke5_reasoning_ids.json`
    (same 5 ids, written before the run)
  - Trajectories: `results/dream_arcc_smoke5_reasoning.jsonl`
  - Run log: `results/dream_arcc_smoke5_reasoning_run_log.json`
  - Scored output: `results/dream_arcc_smoke5_reasoning_results.json`

**Run and record exactly as in `7831ea7`:** pull first, verify the
full suite passes and report the count, run, assert
`len(history) == steps` and final-verdict == sequences-verdict,
score with `python -m traj_eval run`, and append a new baselines
section in the same commit with git revision, env, GPU index,
full sampler config, pasted aggregate, and undecided count.

**Gate for passing:**

- Commitment fields must be non-null on the reasoning trajectories
  (T_commit, lag, and unwritten fraction actually populated from
  real history, not just from hand-built test steps).
- At least some trajectories must contain real post-verdict or
  pre-verdict reasoning text (post_change_rate > 0 on some items),
  not just "Answer: X" from step 3 onward.
- Report the three rationalization components separately for both
  conditions, side by side. Never report the scalar alone.

**If the gate fails** (reasoning traces still collapse to
"Answer: X" in the first few steps with no rationale): STOP. Do not
scale, do not try a second prompt variant silently. Commit the
failed run with its log and a note saying "reasoning smoke failed:
<reason>", and ask for review. A failed prompt condition is a
finding, not something to paper over with a third variant.

## 5. Only after the reasoning smoke is clean, in this order

1. Full ARC-C, both conditions (direct + reasoning), same sampler,
   ids fixed first, run log required. Compare conditions, do not
   pool them into one accuracy number.
2. GSM8K: separate numeric-answer extraction + exemplar stripping,
   new function + tests first. Do NOT reuse the A-D regex.
3. Forced deliberation on the smoke set, with K + extra budget
   pre-registered in `results/baselines.md` BEFORE looking at
   accuracy.
4. LLaDA-8B-Instruct + LLaDA-1.5 adapters for the main text table.
5. MMaDA-8B-MixCoT as holdout, after the text trio. Never as a
   substitute for LLaDA-1.5 in the first table.

## 6. Standing rules (unchanged)

- No weights in tests. Ever. Fakes/stubs only.
- No torch/transformers imports at module import time in
  `src/traj_eval/`. Lazy imports inside `run_*` only.
- Do not change the null rule (`None` = no verdict, not a flip)
  without updating metrics, adapter, handoff, and paper outline
  together.
- Do not change a metric definition without updating its docstring,
  its tests, the README metrics table, and a note in the run log.
- Do not claim a result from 5 items. "Smoke clean" / "smoke
  failed: <reason>" only. Accuracy 1.0 on 5 ARC-C items is a
  pipeline check, not a model evaluation.
- Small commits, conventional messages, one logical change per
  commit. Baselines updated in the same commit as the run.
