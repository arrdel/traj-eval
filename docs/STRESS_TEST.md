# Stress Test and Red-Team Report (`traj-eval`)

Date: 2026-11-10
Git revision tested: `eb7c070` (`master` after PR #33)
Tracking issue: [#34](https://github.com/arrdel/traj-eval/issues/34)

## 1. Machine Specifications and Environment

| Field | Value |
|---|---|
| Hostname | `titan3` |
| OS / Kernel | Linux `4.18.0-553.40.1.el8_10.x86_64` |
| CPU | Dual-socket Intel(R) Xeon(R) Gold 6346 CPU @ 3.10 GHz (32 physical cores / 64 logical threads, 36 MiB L3 cache per socket) |
| System RAM | 1.0 TiB (`872 GiB` available at run) |
| Python | `3.13.11` (`.venv/bin/python`, single-threaded CPU execution) |
| Storage for stress artifacts | `/tmp/traj_eval_stress/` (ephemeral, removed after run; zero stress artifacts written to `results/`) |

## 2. Documented Performance Budget and Verdict

- **Pre-registered CPU budget:** Score `1,000` trajectories (`256` denoising steps each, including `load_trajectories`, `evaluate_trajectories`, `stratified_evaluation`, and `compare_conditions` with `n_boot = 1000` across 5 metrics) in **under 10 minutes (`600.0s`)** of single-core CPU wall time and **under 4 GiB (`4,096 MiB`)** peak resident memory (`VmHWM`).
- **Verdict: PASS (by a wide margin).**
  - On `1,000` trajectories (`256` steps each), total CPU wall time across all stages (`load + evaluate + stratified + compare`) is **`4.30s`** (`0.72s` load + `0.265s` evaluate + `0.399s` stratified + `1.891s` compare) and resident memory is **`< 400 MiB`** (`139x` faster than the 600s time budget and `10x` below the 4 GiB memory budget).
  - Even at the **`10,000`-trajectory ceiling** (`2,560,000` total denoising steps, `530.0 MB` JSONL), total CPU wall time across all stages is **`34.18s`** (`7.268s` load + `2.644s` evaluate + `4.034s` stratified + `20.230s` compare) with **`1,877.4 MiB`** peak `VmHWM` (`1,355.4 MiB` steady RSS), still well inside the 10-minute / 4 GiB budget.

## 3. Part 1: Full-Set and Ceiling Benchmarks on CPU

### 3.1 Full Benchmark Test Splits (`ARC-C` 1,172 items and `GSM8K` 1,319 items, 256 steps each)

Full test-set prompts and gold answers were loaded from the cached Hugging Face test splits (`allenai/ai2_arc` `ARC-Challenge` `test` = `1,172` items; `openai/gsm8k` `main` `test` = `1,319` items) paired with 256-step denoising trajectories in `/tmp/traj_eval_stress/`:

| Suite | Items ($N$) | Steps ($T$) | JSONL Size | `load_trajectories` Wall Time | `load_trajectories` RSS Delta (Total RSS) | `evaluate_trajectories` Wall Time (Peak Alloc) | `stratified_evaluation` Wall Time (Peak Alloc) | `compare_conditions` Wall Time ($n_{\text{boot}}=1000$) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `ARC-C` full test (`direct`) | 1,172 | 256 | 48.0 MB | `0.689s` | `+115.1 MiB` (`129.9 MiB`) | `0.399s` (`1.5 MiB`) | `0.619s` (`0.2 MiB`) | `2.362s` (paired vs. `reasoning`, `362.8 MiB` RSS) |
| `ARC-C` full test (`canonical reasoning`) | 1,172 | 256 | 193.0 MB | `1.331s` | `+232.0 MiB` (`362.1 MiB`) | `0.198s` (`1.5 MiB`) | `0.320s` (`0.2 MiB`) | `2.362s` (paired vs. `direct`, `362.8 MiB` RSS) |
| `GSM8K` full test (`numeric`) | 1,319 | 256 | 164.0 MB | `1.157s` | `+154.2 MiB` (`380.4 MiB`) | `0.265s` (`1.7 MiB`) | `0.422s` (`0.2 MiB`) | `2.415s` (`380.4 MiB` RSS) |

### 3.2 Trajectory Count Scaling ($N \in \{64, 250, 1000, 2500, 10000\}$ at $T = 256$ steps)

Generated a `530.0 MB` synthetic JSONL file (`10,000` trajectories $\times$ `256` steps = `2,560,000` steps across `genuine`, `rationalization`, and `oscillation` archetypes):
- `load_trajectories` (`10,000 x 256`): **`7.268s`**, `+1,078.6 MiB` RSS delta (`1,347.7 MiB` steady RSS; `1,877.4 MiB` peak `VmHWM` during `p.read_text().splitlines()`).

| Trajectories ($N$) | Scale Factor | `evaluate_trajectories` | `stratified_evaluation` | `compare_conditions` ($n_{\text{boot}}=1000$) | Steady RSS |
|---:|---:|---:|---:|---:|---:|
| 64 | $1.00\times$ | `0.018s` | `0.025s` | `0.128s` | `1,347.7 MiB` |
| 250 | $3.91\times$ | `0.067s` ($3.72\times$) | `0.097s` ($3.88\times$) | `0.442s` ($3.45\times$) | `1,347.7 MiB` |
| 1,000 | $4.00\times$ | `0.265s` ($3.96\times$) | `0.399s` ($4.11\times$) | `1.891s` ($4.28\times$) | `1,348.4 MiB` |
| 2,500 | $2.50\times$ | `0.659s` ($2.49\times$) | `1.006s` ($2.52\times$) | `5.066s` ($2.68\times$) | `1,349.5 MiB` |
| 10,000 | $4.00\times$ | `2.644s` ($4.01\times$) | `4.034s` ($4.01\times$) | `20.230s` ($3.99\times$) | `1,355.4 MiB` |

### 3.3 Step Horizon Scaling ($T \in \{256, 1024, 4096\}$ at $N = 64$ trajectories)

#### A. Bounded-length step text (realistic fixed canvas length)

| Trajectories ($N$) | Steps ($T$) | Step Factor | JSONL Size | `load_trajectories` | `evaluate_trajectories` | `compare_conditions` |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 256 | $1.0\times$ | 3.0 MB | `0.035s` | `0.018s` | `0.127s` |
| 64 | 1,024 | $4.0\times$ | 12.1 MB | `0.143s` ($4.09\times$) | `0.075s` ($4.17\times$) | `0.275s` ($2.17\times$) |
| 64 | 4,096 | $4.0\times$ | 48.7 MB | `0.576s` ($4.03\times$) | `0.354s` ($4.72\times$) | `1.017s` ($3.70\times$) |

#### B. Repo synthetic generator `_rationalization(idx, n_steps=T)` in `src/traj_eval/synthetic.py` (Quadratic $O(T^2)$ blowup)

| Trajectories ($N$) | Steps ($T$) | Step Factor | Generation Time | Total `reasoning_text` Characters | Memory Growth Factor | `evaluate_trajectories` |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 64 | $1.0\times$ | `0.005s` | `717,536` (`0.72 MB`) | $1.00\times$ | `0.002s` |
| 16 | 256 | $4.0\times$ | `0.061s` ($12.2\times$) | `11,714,624` (`11.71 MB`) | **$16.33\times$** | `0.005s` |
| 16 | 1,024 | $4.0\times$ | `0.987s` ($16.2\times$) | `191,326,976` (`191.33 MB`) | **$16.33\times$** | `0.023s` |

### 3.4 Scaling Verdict per Function

| Function / Module | Asymptotic Complexity | Scaling Verdict | Notes |
|---|---|---|---|
| `load_trajectories` (`io.py`) | $O(N \cdot T)$ time and space | Linear | Transient peak `VmHWM` is $\sim 1.4\times$ steady RSS because `p.read_text().splitlines()` holds the full file string and line list in memory simultaneously. |
| `evaluate_trajectories` (`metrics.py`) | $O(N \cdot T)$ time, $O(N)$ output space | Linear (`0.264 ms / traj` at $T=256$) | Calls `_convergence_index(t)` 7 times and `rationalization_components(t)` 2 times per trajectory (constant-factor redundancy, filed in [#39](https://github.com/arrdel/traj-eval/issues/39)). |
| `stratified_evaluation` (`metrics.py`) | $O(N \cdot T)$ time | Linear (`0.403 ms / traj` at $T=256$) | Recomputes per-trajectory metrics across `all`, `correct`/`incorrect`, and `cotsel_gaps` ($1.53\times$ slower than `evaluate_trajectories`; filed in [#39](https://github.com/arrdel/traj-eval/issues/39)). |
| `paired_bootstrap_delta` (`metrics.py`) | $O(N \cdot T + N \cdot B)$ time ($B = n_{\text{boot}}$) | Linear in $N$, high Python loop constant | Draws $N \cdot B$ pure-Python `rng.randrange(n_paired)` samples per metric ($5 \times 10^7$ calls across 5 metrics at $N=10,000, B=1000$, taking `20.23s`; filed in [#39](https://github.com/arrdel/traj-eval/issues/39)). |
| `compare_conditions` (`metrics.py`) | $O(N \cdot T + N \cdot B)$ time | Linear (`2.023 ms / pair` at $T=256$) | Recomputes `_per_item(t)` 5 times in `paired_bootstrap_delta` after already running `evaluate_trajectories` on both conditions (filed in [#39](https://github.com/arrdel/traj-eval/issues/39)). |
| `_rationalization` (`synthetic.py`) | **$O(T^2)$ time and space** | **Quadratic in $T$ (FAIL)** | Appends $i$ copies of `"Supporting detail j."` at step $i$, allocating $\Theta(T^2)$ characters per trajectory ($16.3\times$ memory jump per $4\times$ increase in $T$). Also `_genuine` and `_oscillation` silently cap at 8 steps (filed in [#38](https://github.com/arrdel/traj-eval/issues/38)). |

## 4. Part 2 & Part 3: Property Tests, Red-Team Corpus, and Follow-Up Issues

All 22 public functions in `src/traj_eval/metrics.py` are property-tested in `tests/test_stress.py`, and the adversarial corpus + 10,000-trajectory seeded fuzz test are implemented in `tests/test_adversarial.py`. Per Issue #34 rules, zero metric or library definitions were modified in this PR; every failure uncovered during stress testing and red-teaming has been filed as its own follow-up issue:

| Issue | Component | Summary of Failure Found |
|---|---|---|
| [#37](https://github.com/arrdel/traj-eval/issues/37) | `src/traj_eval/io.py`, `src/traj_eval/cli.py` | 1. Passing a directory to `load_trajectories` / `traj-eval run` raises unhandled `IsADirectoryError` traceback instead of exiting with code `2`.<br>2. `"step": Infinity` in JSON raises unhandled `OverflowError`; `"step": true` silently parses as `1`.<br>3. `"committed_reasoning_fraction": [1, 2]` raises unhandled `TypeError` in CLI; `"committed_reasoning_fraction": NaN` is accepted and causes `unwritten_reasoning_at_convergence` to silently return `1.0`.<br>4. `"answer_committed": "false"` silently evaluates to `True` via `bool("false")`, and non-string predictions (`42`, `["A"]`) are stringified via `str(raw_pred)`.<br>5. Negative (`-5`), duplicate (`0, 0`), and huge (`10^9`) step numbers are accepted without validation, causing `verdict_convergence_step` and `answer_commitment_step` to return values outside `[0, len(steps))`. |
| [#38](https://github.com/arrdel/traj-eval/issues/38) | `src/traj_eval/synthetic.py` | 1. `_rationalization(idx, n_steps=T)` has a quadratic $O(T^2)$ string memory and runtime blowup.<br>2. `_genuine(idx, n_steps)` and `_oscillation(idx, n_steps)` slice 8-element lists and silently cap at 8 steps when `n_steps > 8`.<br>3. `generate_synthetic_trajectories` does not expose an `n_steps` parameter. |
| [#39](https://github.com/arrdel/traj-eval/issues/39) | `src/traj_eval/metrics.py` | 1. `paired_bootstrap_delta` has an IEEE-754 quantile index asymmetry at `ci = 0.90, n_boot = 1000` (`alpha = (1.0 - 0.90)/2.0 = 0.04999999999999999`, so `lo_idx = floor(49.999999999999986) = 49` while `hi_idx = 949` corresponds to index `50` from the top), causing `ci_low(A, B)` to differ slightly from `-ci_high(B, A)`.<br>2. `paired_bootstrap_delta` silently overwrites duplicate trajectory IDs in `base_map` / `target_map` and depends on `base_trajectories` ID ordering.<br>3. Redundant per-trajectory metric recomputation across `evaluate_trajectories`, `stratified_evaluation`, and `compare_conditions`. |
| [#40](https://github.com/arrdel/traj-eval/issues/40) | `src/traj_eval/adapters/dream.py`, `src/traj_eval/adapters/llada.py`, `experiments/run_dream_arcc_smoke.py` | 1. Missing keys in `items` dicts raise raw `KeyError` after `out_ids` is already written.<br>2. Non-positive `steps` / `max_new_tokens` in `DreamSmokeConfig` are not validated upfront.<br>3. Passing a directory or malformed JSON to `experiments/run_dream_arcc_smoke.py --items` raises an unhandled traceback instead of exiting cleanly with code `2`. |
