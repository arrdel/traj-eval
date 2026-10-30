# Cursor Agent Handoff — Dream Adapter Landed, Smoke Is Next

**Read this first. Do not re-do what is already done. Do not jump ahead.**

Date: 2026-10-19
Repo: `arrdel/traj-eval`, branch `master`
Last pushed commit in this handoff: Dream adapter + null-aware metrics + smoke runner + this doc.
Local verification before push: 32 tests passed on the adapter author's machine
(24 original v0.1 tests + 8 new Dream adapter tests). Remote also has GRPO
tests from commit `266870c` — total on remote after this push should be 38.
**Your first job: pull, then run the full suite and confirm the count.**

## 1. What just landed (do NOT rebuild it)

- `src/traj_eval/adapters/__init__.py` + `src/traj_eval/adapters/dream.py`
  Dream adapter for `Dream-org/Dream-v0-Instruct-7B`, ARC-C only.
  - `extract_verdict(text)` — final A-D letter or `None`, never guesses.
    Answer-prefix patterns take precedence, last match wins.
  - `decode_generation_region(history_entry, prompt_len, tokenizer, mask_id)`
    Slices off prompt, strips mask tokens, decodes remainder.
  - `history_to_trajectory(...)` — one step per `history` entry,
    `prediction` = A-D or `None`, `reasoning_text` = decoded mask-stripped region.
  - `run_dream_arcc_smoke(items, config)` — real GPU path, lazy torch/transformers
    imports, writes JSONL + ids + run log, asserts `len(history) == steps`
    and final verdict == sequences verdict.
  - `DreamSmokeConfig` — defaults: `max_new_tokens=256, steps=256,
    temperature=0.2, top_p=0.95, alg="entropy", alg_temp=0.0`.
- Null-aware metrics (BREAKING but intentional, documented):
  - `TrajectoryStep.prediction` is now `str | None`. `None` = "no verdict yet".
  - `flip_rate`: `None -> "B"` is NOT a flip. Flips counted only between two
    non-null differing verdicts. Denominator = `len(non-null) - 1`.
  - `_convergence_index`: computed over non-null sequence, mapped to original index.
  - `Trajectory.is_correct` returns `None` when final prediction is `None`.
  - `io._parse_step`: missing/`null` prediction -> `None`, JSON `null` round-trips.
  - Existing string-only trajectories behave exactly as before (demo output unchanged).
- `experiments/run_dream_arcc_smoke.py`
  - Real: `--items <5-item JSON> --out-jsonl ...` (requires cdr-gpu + GPU).
  - Fake (no GPU): `--fake --n-items 2` — exercises adapter with stub model.
- `tests/test_adapters_dream.py` — 8 tests, no torch/weights.
- This file: `docs/CURSOR_HANDOFF_DREAM_SMOKE.md`.

## 2. Pull and verify (do this FIRST, before any new code)

```bash
git pull origin master
git log --oneline -5
git status
pip install -e ".[dev]"
python -m pytest -q
# Expect: all pass. If count is not 38, STOP and report the failure, do not push.
python -m traj_eval demo
python experiments/run_dream_arcc_smoke.py --fake --n-items 2
python -m traj_eval run --input /tmp/fake_smoke.jsonl
# Fake smoke JSONL path depends on your --out-jsonl; use the path you passed.
```

If any test fails, fix the test or the code that broke it, do NOT delete the
failing test to make the suite green. Report the failure in your commit message
if you fix it.

## 3. Your next task: the Dream ARC-C 5-item smoke (the ONLY next experiment)

Do NOT start LLaDA, MMaDA, forced deliberation, or GRPO training. Do NOT
expand to full ARC-C or GSM8K until the 5-item smoke is clean and logged.

### 3a. Environment (non-negotiable)

- `conda activate cdr-gpu` — torch 2.5.1+cu121, transformers 4.46.2.
- Do NOT use base (torch 2.11 + transformers 5.5, CUDA unavailable) or repo `.venv` (no transformers).
- Pick ONE GPU with >= ~25 GB free at run time. Check first:
  ```bash
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
  ```
- Pin it: `export CUDA_VISIBLE_DEVICES=<that index>`. Do NOT run against occupied GPUs.
- Do NOT change temperature / top_p / steps mid-run. Log exactly what ran.

### 3b. Inputs (ids fixed BEFORE the run)

1. Build `results/dream_arcc_smoke5_input.json`: JSON list of 5 ARC-C items,
   each `{"id": str, "prompt": str, "gold_answer": str}`.
   Prompt must ask for a letter answer (ARC-C format from the spec).
2. Write ids to `results/dream_arcc_smoke5_ids.json` BEFORE running.
   The adapter also writes them, but the protocol requires ids fixed first.

### 3c. Run

```bash
conda activate cdr-gpu
export CUDA_VISIBLE_DEVICES=<free card>
python experiments/run_dream_arcc_smoke.py \
  --items results/dream_arcc_smoke5_input.json \
  --out-jsonl results/dream_arcc_smoke5.jsonl \
  --out-ids results/dream_arcc_smoke5_ids.json \
  --out-log results/dream_arcc_smoke5_run_log.json
```

Sanity asserts inside the adapter will fail loudly if:
- `len(history) != steps`, or
- final parsed verdict != verdict parsed from `sequences`.

If an assert fails, STOP. Do not fudge the assert. Paste the full traceback
and the run log into your next commit message / PR description and ask for review.

### 3d. Score and record (do this immediately after the run)

```bash
python -m traj_eval run --input results/dream_arcc_smoke5.jsonl --output results/dream_arcc_smoke5_results.json
```

Then append to `results/baselines.md`:
- git revision (`git rev-parse HEAD`)
- env (`cdr-gpu`, torch/transformers versions, `nvidia-smi` output)
- GPU index, `CUDA_VISIBLE_DEVICES`
- FULL sampler config: temperature, top_p, alg, alg_temp, steps, max_new_tokens
- Aggregate from `traj_eval run` (paste, do not retype)
- How many of the 5 items were non-null at the last step (undecided count)

Do NOT claim a result from 5 items. The smoke proves the pipeline is clean,
not that any model reasons. Write "smoke clean" or "smoke failed: <reason>",
nothing stronger.

## 4. After the smoke is clean (SUPERSEDED — see `docs/CURSOR_FOLLOWUP_2026-10-29.md`)

**Stop:** The sequencing below was superseded on 2026-10-29 by
`docs/CURSOR_FOLLOWUP_2026-10-29.md` §§3–5 (which puts commitment logging +
the 5-item reasoning-condition ARC-C smoke `dream_arcc_smoke5_reasoning_*`
ahead of scaling to full ARC-C). Follow `docs/CURSOR_FOLLOWUP_2026-10-29.md`.

Original sequence (for historical reference only):
1. Full ARC-C (same adapter, same sampler, ids fixed first, run log required).
2. GSM8K — SEPARATE numeric-answer extraction + exemplar stripping.
   Do NOT reuse the A-D regex for GSM8K. Write a new extraction function
   + tests before running.
3. Commitment logging (Yeom et al. style): `answer_committed`,
   `committed_reasoning_fraction`, stability-commitment lag. History gives
   us what we need (a position is committed once it leaves `mask_token_id`
   and never returns). Add it as new fields, do not break existing JSONL.
4. Forced deliberation on the smoke set, with K + extra budget pre-registered
   in `results/baselines.md` BEFORE looking at accuracy.
5. LLaDA-8B-Instruct + LLaDA-1.5 adapters for the main text table.
   MMaDA stays as holdout after the text trio.

## 5. Rules (violating these = revert)

- No weights in tests. Ever. Tests use fakes/stubs only.
- No torch / transformers imports at module import time in `src/traj_eval/`.
  Lazy imports inside `run_*` functions only.
- Do not change core metric definitions without updating docstring + tests
  + README metrics table + run log note.
- Do not change the null rule (`None` = no verdict, not a flip) without
  updating `metrics.py`, `adapters/dream.py`, this handoff, and the paper
  outline's method section together.
- Small commits, conventional messages (`feat:`, `fix:`, `docs:`, `test:`).
  One logical change per commit. Update `results/baselines.md` in the same
  commit as the run it describes, never in a later "cleanup" commit.
- Do NOT claim synthetic numbers are model results. Do NOT claim the
  rationalization heuristic proves unfaithfulness. Label it heuristic.
- If GPUs are full, do NOT start a download-and-eval job. Wait, or run
  `--fake` + tests only, and say so in your update.

## 6. Definition of done for the smoke (completed 2026-10-29)

- [x] `python -m pytest -q` passes on your machine, count reported (38 passed at pull; 46 passed after post-smoke metrics + GSM8K + LLaDA adapter additions).
- [x] `results/dream_arcc_smoke5_input.json` + `ids.json` exist, ids fixed before run (`4b60d44`).
- [x] `results/dream_arcc_smoke5.jsonl` exists, 5 trajectories, one step per history entry (`7831ea7`).
- [x] `results/dream_arcc_smoke5_run_log.json` exists with full sampler + env + git rev (`7831ea7`).
- [x] `results/baselines.md` updated with aggregate (pasted) + undecided count (`7831ea7`, `5fb1735`).
- [x] Commit pushed with message starting `feat(smoke):` and a body that states `smoke clean` (`7831ea7`).

Post-smoke additions also landed and pushed (2026-10-29):
- `5fb1735`: `rationalization_components`, `answer_commitment_step` (`T_commit`), `stability_commitment_lag` (`T_commit - T_stab`), `unwritten_reasoning_at_convergence` (`U`), and per-step commitment tracking in `history_to_trajectory`.
- `9d38423`: GSM8K numeric extraction (`extract_gsm8k_verdict`, `strip_gsm8k_exemplars`, `normalize_numeric_answer`) and `--force-deliberation-steps` hook.
- `df94f94`: Pre-registered fixed inputs and IDs for ARC-C CoT 5-item (`dream_arcc_cot_smoke5_*`), GSM8K 5-item (`dream_gsm8k_smoke5_*`), and 64-item suites (`arcc_smoke64_*`, `smoke64_ids.json`, `gsm8k_smoke64_input.json`).
- `c668e55`: LLaDA adapter (`src/traj_eval/adapters/llada.py`) for `GSAI-ML/LLaDA-8B-Instruct` and `GSAI-ML/LLaDA-1.5` (`mask_id=126336`, EOS/EoT confidence suppression) + CLI `--model-family` support + unit tests (46 total passing).

## 7. If you get stuck

- Adapter import error -> check `pip install -e ".[dev]"` and `PYTHONPATH=src`.
- `torch.cuda.is_available() == False` -> you are in base or `.venv`, not `cdr-gpu`.
- `len(history) != steps` -> print `len(history)`, `steps`, model id, transformers version; do not patch the assert.
- Final verdict mismatch -> print both verdicts + decoded `sequences` text (first 500 chars) in the log; do not force them to match by re-parsing.
- GPU OOM -> pick a different card with >=25 GB free, reduce `max_new_tokens`/`steps`
  ONLY in a new logged run, never silently mid-run.

Questions: open an issue with a minimal repro (fake history is fine). Do not
DM the user for a decision the spec already makes.
