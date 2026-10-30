# Cursor Follow-up 2 — Review of Commits Through 842544c + Updated Directives

**Read this first. It supersedes `docs/CURSOR_FOLLOWUP_2026-10-29.md`
only where it says so. The standing rules in that doc section 6
(null rule, no weights in tests, lazy imports, small commits,
baselines in the same commit as the run, no claim from 5 items)
still hold in full.**

Date: 2026-10-30 (review of commits through `842544c`)
Repo: `arrdel/traj-eval`, branch `master`
Reviewer: Muse (read-back from GitHub via the GitHub connector;
the review machine has no GPU and no direct git clone, so test
execution was NOT independently re-run here, see section 4)

## 1. Status, verified

- Head at review: `842544c` — "mark pre-GPU fixes and reasoning
  smoke pre-registration complete", reporting 56/56 tests passing
  and "paused awaiting GPU reservation (>= 25 GB free VRAM)".
- **Confirmed: the reasoning smoke has NOT run yet.**
  `results/dream_arcc_smoke5_reasoning.jsonl`,
  `results/dream_arcc_smoke5_reasoning_run_log.json`, and
  `results/dream_arcc_smoke5_reasoning_results.json` do not exist
  in the repo. Pre-registration is complete; measurement is not.
  Nothing in this doc changes that fact, and no result may be
  claimed until those three files exist from a real run.
- The pre-GPU fixes from the Oct 29 follow-up section 3 are done
  and verified by code read-back:
  - §3.1 Dream docstring updated.
  - §3.2 No-post rationalization branch is now `0.4 * early_factor`,
    with a last-step-convergence test.
  - §3.3 Answer-position detection tries exact stripped-token match
    before substring fallback, with a fake-history test where
    trailing rationale (" because it cools.") contains the verdict
    letter and the answer token is earlier. The clause-initial
    article strip ("A planet..." must not emit verdict "A") is also
    tested. Both are sensible and correctly scoped.
  - §3.4 GSM8K protocol section in baselines is relabeled as
    pre-registered, not yet run, and not the ARC-C smoke protocol.
  - §3.5 Old handoff section 4 now points to the Oct 29 follow-up.
- The reasoning input is correct: `results/dream_arcc_smoke5_reasoning_input.json`
  has the same 5 ids and gold answers as the direct-answer smoke
  (C, B, C, D, D) and the exact pre-registered closing line:
  `Think step by step, then give your final answer in the form
  "Answer: <letter>".`

## 2. What is good in the new batch (keep it)

- GSM8K extraction (`9d38423`): `normalize_numeric_answer`,
  `strip_gsm8k_exemplars`, and `extract_gsm8k_verdict` emit a
  verdict only from explicit markers (`####`, `\boxed{}`, final
  answer phrases), never from intermediate arithmetic operands.
  That matches the paper outline's requirement. It is code only
  so far; it has no real GSM8K measurement yet, and that is fine.
- Gated rationalization score (`521f692`): the re-scored direct
  smoke now reports `mean_gated_rationalization_score = 0.0`
  alongside the unchanged additive `0.395`. That is the correct
  interpretation of the degenerate trace, and it must be reported
  with the components, never as a replacement for them.
- Intervention defaults are safe: `force_deliberation_steps`
  defaults to 0 and `frontier_width` defaults to None, so the
  reasoning smoke path is unaffected unless those are explicitly
  set. Keep it that way for the reasoning smoke (see section 3).
- Pre-registered 64-item id lists are fine as files. Pre-registering
  ids is not running an experiment.

## 3. Problems and freeze (this is the main directive)

### 3a. Scope ran ahead of measurement. Freeze new features now.

Between the direct-answer smoke and this review, Cursor added, in
addition to the requested fixes: a LLaDA adapter (`c668e55`), a
GSM8K extractor plus a forced-deliberation hook (`9d38423`), a
batch of new metrics (`521f692`: gated score, disagreement slice,
SBW lock-in, coverage decomposition, CoTsel gaps, paired bootstrap,
Pass@k), and GRPO credit-assignment schemes (`f345c5d`: broadcast,
pre-convergence, pre-commitment, StepRS, ForkGRPO, DTCA, plus a
post-convergence credit fraction tied to Claim C5).

None of that is measured on a real checkpoint yet, and most of it
was explicitly sequenced AFTER the reasoning smoke in the Oct 29
follow-up. The LLaDA/GSM8K work that landed at 03:39-03:45 UTC
predates the Oct 29 follow-up commit (03:52 UTC), so it is
understandable; the GRPO work at 04:04 UTC does not have that
excuse.

**Freeze, effective immediately, until the reasoning smoke gate
in section 5 passes:**

- No new adapters, no new metrics, no new credit schemes, no new
  intervention hooks, no new prompt conditions.
- Do not add SBW, CoTsel, Pass@k, StepRS, ForkGRPO, or DTCA
  results to the paper outline or to any main table until each
  definition is verified against its source paper and pinned by a
  test that reproduces the definition, not just the code's own
  output. A function name plus an arXiv number is not a citation
  check. Until then those functions are unvalidated code, and the
  README metrics table must say so if it lists them.
- Do not use frontier gating or forced deliberation in the
  reasoning smoke. Both change the very commitment clocks the
  smoke is meant to measure. They are separate, pre-registered
  interventions for later, each with its own run log.

### 3b. Two prompt conditions exist for the same 5 items. Pick one.

- Canonical reasoning smoke (Oct 29 follow-up §4):
  `results/dream_arcc_smoke5_reasoning_input.json`, closing line
  `Think step by step, then give your final answer in the form
  "Answer: <letter>".`
- Earlier CoT file (`df94f94`):
  `results/dream_arcc_cot_smoke5_input.json`, closing line
  `Think step by step in 2-3 brief sentences, then end your
  response on a new line with "Answer: <letter>"...`

These are different conditions: the 2-3-sentence cap changes
trajectory length, post-change rate, and commitment timing.
**The reasoning smoke runs ONLY the canonical reasoning file.**
Do not run the CoT file as the reasoning smoke, do not pool the
two, and do not silently substitute the CoT prompt at 64-item
scale. Before any 64-item run, reconcile in baselines: either the
full ARC-C reasoning condition uses the canonical reasoning
prompt, or the CoT-brief prompt is pre-registered as a third,
separately named condition. A prompt swap between 5-item and
64-item scale invalidates the comparison.

### 3c. Forced deliberation has two different definitions in the repo. Do not mix them.

- Old pre-registered protocol: continue 64 extra steps after the
  parsed verdict has been stable for 16 steps, answer span still
  remaskable.
- New hook (`9d38423` / baselines pre-registration): suppress EOS
  logits for the first K steps (K=64 in baselines).

Those are different interventions. The K=64 EOS-suppression run
may proceed only after the reasoning gate passes, and it must be
logged under its own name ("EOS-suppression K=64"), never as if
it implemented the old post-stability protocol. If the old
protocol is still wanted, pre-register it separately with its own
output files.

### 3d. The baselines "ready" list is a pre-registration list, not a queue.

`results/baselines.md` now lists reasoning smoke, forced
deliberation, GSM8K smoke, and 64-item runs as "ready for next
GPU window". Read it as pre-registration only. **Do not chain
runs in one GPU window.** Each step gates the next. A failed
reasoning smoke stops the queue (see section 5).

### 3e. Pasted baselines aggregate is now stale.

The direct-smoke section in baselines still pastes the old
aggregate (no gated score, coverage, or disagreement fields),
while `results/dream_arcc_smoke5_results.json` was re-scored with
the new metrics. The original numbers did not change (accuracy
1.0, rationalization 0.395, convergence step 3.2), which is good,
but the paste must be updated in the same commit as the reasoning
run, with a note that it is a re-score of the Oct 29 JSONL, not a
new run. Do not edit the original run log.

## 4. Test-count note (honest limit of this review)

Cursor reports 56/56 tests passing. Read-back of the five test
files in the repo shows 58 test functions in total
(test_metrics 26, test_adapters_dream 12, test_io 8,
test_grpo_minimal 8, test_adapters_llada 4). The 2-function gap
is plausibly collection or parametrized-count differences, and
is NOT treated here as a failure, but it is also NOT independently
verified: this review machine could not clone the repo (direct
network from the VM times out) and has no GPU, so it read code
and artifacts, it did not execute the suite.

**Next commit from Cursor must include the verbatim tail of
`python -m pytest -q` in its commit message**, as the smoke
commits did. A test count without that tail is not accepted as
verification from now on.

## 5. The ONLY next GPU experiment (unchanged from Oct 29 §4)

When a GPU with >= 25 GB free is reserved, run exactly this and
then stop and report. Do not continue into forced deliberation,
GSM8K, or 64-item runs in the same window.

```bash
git pull origin master
conda activate cdr-gpu
nvidia-smi  # record the card, its free memory, and utilization
export CUDA_VISIBLE_DEVICES=<that card>
python experiments/run_dream_arcc_smoke.py \
  --items results/dream_arcc_smoke5_reasoning_input.json \
  --out-jsonl results/dream_arcc_smoke5_reasoning.jsonl \
  --out-ids results/dream_arcc_smoke5_reasoning_ids.json \
  --out-log results/dream_arcc_smoke5_reasoning_run_log.json
python -m traj_eval run --input results/dream_arcc_smoke5_reasoning.jsonl \
  --output results/dream_arcc_smoke5_reasoning_results.json
```

Sampler, unchanged: temperature 0.2, top_p 0.95, alg entropy,
alg_temp 0.0, steps 256, max_new_tokens 256.
Interventions, unchanged: `force_deliberation_steps = 0`,
`frontier_width = None`. Log both in the run log.

Gate for passing, all required:

- Commitment fields non-null on the reasoning trajectories
  (T_commit, lag, unwritten fraction populated from real history).
- At least some trajectories contain real reasoning text
  (post_change_rate > 0 on some items), not just "Answer: X"
  from step 3 onward.
- Three rationalization components reported separately for both
  conditions, side by side, plus the gated score. Never the
  scalar alone.

If the gate fails: commit the failed run with its log and a note
saying "reasoning smoke failed: <reason>", update baselines in
that same commit, and ask for review. Do not try a second prompt
variant silently, and do not fall back to running the CoT file
instead.

## 6. Only after the reasoning smoke is clean, in this order

1. Full ARC-C, both conditions (direct + canonical reasoning),
   same sampler, ids fixed first, run log required. Compare
   conditions; do not pool them into one accuracy number.
2. GSM8K 5-item smoke, then 64-item suites, with the numeric
   extractor already built. No A-D regex reuse.
3. Forced deliberation (EOS-suppression K=64, named as such),
   pre-registered before looking at accuracy.
4. LLaDA-8B-Instruct + LLaDA-1.5 adapters, validated on real
   weights before any table claims. The current LLaDA adapter
   has stub tests only; stub tests are not a validation.
5. MMaDA-8B-MixCoT as holdout, after the text trio.
6. GRPO credit assignment (Paper 2 work) only after the Paper 1
   main table exists. Credit fractions computed on toy
   trajectories are not Paper 1 evidence.
