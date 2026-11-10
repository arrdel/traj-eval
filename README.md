# traj-eval

Trajectory-level evaluation for masked diffusion language models.

**Measure reasoning *in* the denoising trajectory, not just final-answer accuracy.**

[Project Page](https://arrdel.github.io/traj-eval/) · [Hugging Face Space](https://huggingface.co/spaces/arrdel/traj-eval) · [Hugging Face Dataset](https://huggingface.co/datasets/arrdel/traj-eval-benchmarks) · [Hugging Face Collection](https://huggingface.co/collections/arrdel/traj-eval-diffusion-lm-reasoning-trajectories-6af321457d8dbd6074548379)

---

## Why final-answer accuracy lies

Two models can have identical final-answer accuracy for very different reasons:

- **Genuine reasoning:** the verdict changes as reasoning develops across intermediate denoising steps, then stabilizes. Reasoning drives the verdict.
- **Rationalization:** the verdict is fixed in the first few steps and reasoning text is elaborated afterwards to justify it. Verdict drives reasoning.
- **Oscillation:** the verdict flips repeatedly across denoising steps and never settles until generation ends, even if the final answer happens to be right.

`traj-eval` makes those patterns visible. It is descriptive, not causal proof. A high `rationalization_score` means "looks like rationalization by this operational definition", not "proved to be rationalizing".

The core scoring library runs on trajectory data in a simple JSONL/JSON format using pure Python and `numpy` (no model weights, no `torch`, and no `transformers` needed to score trajectories or run the demo). Built-in adapters in `src/traj_eval/adapters/` support step-by-step trajectory extraction from `Dream-org/Dream-v0-Instruct-7B`, `GSAI-ML/LLaDA-8B-Instruct`, and `GSAI-ML/LLaDA-1.5`.

## Benchmark Results (64-item suites)

Exact verified measurements from `results/baselines.md` and the scored 64-item JSON artifacts in `results/`. Five-item smoke runs are pipeline checks only and are excluded from this table.

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

### Rationalization components (`Dream-v0-Instruct-7B` only, ARC-C 64-item)

Side-by-side decomposition from `results/dream_arcc_smoke64_direct_vs_reasoning.json`:

| Component | Direct (`direct_64`) | Reasoning (`reasoning_64`) |
|---|---:|---:|
| `early_factor` | `0.9852` | `0.2559` |
| `post_change_rate` | `0.0001` | `0.1597` |
| `length_growth` | `0.0035` | `0.0203` |
| `gated` | `0.0012` | `0.0125` |

## Install

```bash
pip install -e .
# for dev (tests + notebook plotting)
pip install -e ".[dev]"
```

Requires Python >= 3.10.

## Quickstart

```bash
# Run the built-in demo on synthetic trajectories
python -m traj_eval demo

# Generate synthetic data, then evaluate it
python -m traj_eval example --output examples/synthetic_demo.jsonl
python -m traj_eval run --input examples/synthetic_demo.jsonl --output results.json

# Compare two paired conditions side by side
python -m traj_eval compare \
  --base results/dream_arcc_smoke64_direct.jsonl \
  --target results/dream_arcc_smoke64_reasoning.jsonl \
  --base-label direct_64 \
  --target-label reasoning_64
```

Python API:

```python
from traj_eval.io import load_trajectories
from traj_eval.metrics import evaluate_trajectories

trajs = load_trajectories("examples/synthetic_demo.jsonl")
results = evaluate_trajectories(trajs)
print(results["aggregate"])
```

See `notebooks/01-quickstart.ipynb` for a worked notebook (runs without heavy deps; `matplotlib` is optional).

## Input format

JSONL, one trajectory per line (JSON list, or `{"trajectories": [...]}` also accepted):

```json
{"id": "q1", "prompt": "What is ...?", "gold_answer": "B",
 "steps": [
   {"step": 0, "prediction": null, "reasoning_text": "", "answer_committed": false, "committed_reasoning_fraction": 0.0},
   {"step": 1, "prediction": "B", "reasoning_text": "Reconsidering...", "answer_committed": false, "committed_reasoning_fraction": 0.5},
   {"step": 2, "prediction": "B", "reasoning_text": "B fits best. Answer: B", "answer_committed": true, "committed_reasoning_fraction": 1.0}
 ]}
```

Fields:

- `id` (`str`, required), `prompt` (`str`, required), `gold_answer` (`str`, required; use `""` if unknown, accuracy will skip it)
- `steps` (`list`, required; may be empty, handled gracefully)
- Per step: `step` (`int >= 0`, required, unique per trajectory), `prediction` (`str | null`, required; `null` means no verdict yet), `reasoning_text` (`str`, optional), `confidence` (`float`, optional), `logits` (`list`, optional), `answer_committed` (`bool | null`, optional), `committed_reasoning_fraction` (`float | null` in `[0, 1]`, optional)

Malformed files raise `ValueError` with a clear message instead of silently dropping data.

## Metrics

| Metric | Function | Range | What it means |
|---|---|---|---|
| Verdict convergence step | `verdict_convergence_step` | step number or None | First step after which prediction stays constant to the end. Under the null rule, `None` means no verdict yet (not a flip). Returns `None` for empty or all-null trajectories. |
| Flip rate | `flip_rate` | [0, 1] | Fraction of adjacent non-null pairs where prediction changes. 0 = never changed, 1 = changed every decided step. |
| Step consistency | `step_consistency` | dict | `1 - flip_rate` plus `longest_stable_run` and its fraction. |
| Trajectory accuracy | `trajectory_accuracy` | [0, 1] | Fraction where final prediction == gold. Skips empty / unknown-gold trajectories. |
| Forced deliberation delta | `forced_deliberation_delta` | dict | Given `base_accuracy` and `forced_accuracy`, returns `delta = forced - base` and `relative_drop = (base - forced) / base`. Negative delta means forcing longer deliberation hurt accuracy. |
| Rationalization score | `rationalization_score` | [0, 1] | **Heuristic.** High when verdict converges early AND reasoning keeps changing/lengthening after convergence. Formula: `0.4*early_factor + 0.4*post_change_rate + 0.2*length_growth`. Not ground truth. |
| Rationalization components | `rationalization_components` | dict | **Heuristic decomposition.** Returns `early_factor`, `post_change_rate`, `length_growth`, `score`, and `gated_score` so direct-answer runs (`post_change_rate == 0`) are distinguishable from post-verdict elaboration. |
| Gated rationalization score | `gated_rationalization_score` | [0, 1] | Multiplicative-gated heuristic `early_factor * ((0.4*post_change_rate + 0.2*length_growth) / 0.6)`. Returns `0.0` when no post-convergence rationale edits or length growth occur. |
| Answer commitment step | `answer_commitment_step` | step number or None | First step after which `answer_committed` stays True to the end (`T_commit`). None if not logged or never committed. |
| Stability-commitment lag | `stability_commitment_lag` | int or None | `T_commit - T_stab` in steps (`answer_commitment_step - verdict_convergence_step`). None if either is undefined. |
| Unwritten reasoning at convergence | `unwritten_reasoning_at_convergence` | [0, 1] or None | `1 - committed_reasoning_fraction` at the verdict convergence step (`T_stab`). None if not logged. |
| Commitment order gap | `order_gap` | [-1, 1] or None | `(T_commit(ans) - T_median_commit(reason)) / n` (Yeom et al. 2026). Negative values indicate answer-first commitment. |
| Three-clock disagreement | `three_clock_disagreement` | dict | Flags trajectories where verdict stability (`T_stab`), answer commitment (`T_commit`), and rationale completion/edits disagree (`in_disagreement_slice`). |
| Stable-but-wrong lock-in | `is_sbw_lockin` | bool or None | Midpoint lock-in check (Zhao, Yu, & Zhao, LOCKR 2026, `arXiv:2609.27220v2` §3.1): at midpoint `c = round(0.5*(T-1))` with window `w = max(1, ceil((c+1)/2))`, returns `True` iff all predictions in `steps[c-w+1 : c+1]` are non-null, identical (`ell`), and `ell != gold_answer` (or prefix-convergence `conv_idx / T <= threshold` when `threshold` is passed). |
| Coverage-accuracy decomposition | `coverage_accuracy_decomposition` | dict | Symmetric Two-Clocks decomposition of accuracy delta into `coverage_effect` vs. `conditional_accuracy_effect` (Kim & Choi 2026). |
| CoTsel selection gap | `cotsel_selection_gap`, `stratified_evaluation` | dict | Exact correctness-conditioning selection gap `Delta = E[A | C=1] - E[A] = Cov(A, C) / P(C=1)` (Pan & Pan, CoTsel 2026, `arXiv:2610.37070v1` §3.2 Def. 2 Eq. 11-13 & Prop. 1 Eq. 14-16) and stratified summaries across `all`, `correct`, `incorrect`, and `undecided`. |
| Paired bootstrap delta | `paired_bootstrap_delta` | dict | Paired bootstrap confidence interval for `target - base` over matched question IDs (Yeom et al. 2026). |
| Pass@k & diversity | `pass_at_k_metrics` | dict | Unbiased `Pass@k = 1 - comb(n - c, k) / comb(n, k)` estimator (Chen et al. 2021; Ni et al., Flexibility Trap 2026, `arXiv:2601.15165` §2.1) and unique answer/rationale ratios (`arXiv:2610.00661v3` §4) across multiple rollouts per prompt. |
| Early convergence fraction | `early_convergence_fraction` | [0, 1] | Fraction of trajectories converging within the first 20% of steps (threshold configurable). |
| Aggregate | `evaluate_trajectories` | dict | Per-trajectory results + aggregate: accuracy, coverage, conditional accuracy, mean/median convergence step and fraction, mean flip rate, mean rationalization score and components, early convergence fraction, and commitment-clock summaries. |

Edge cases: empty trajectory returns None / 0.0 instead of crashing. Single-step trajectory converges at that step, flip rate 0.0.

## Synthetic demo output

From `python -m traj_eval run --input examples/synthetic_demo.jsonl` (15 synthetic trajectories, 5 per archetype). **Accuracy is 1.0000 for all three archetypes**, because the synthetic data is built so every final answer is correct. Accuracy alone cannot tell them apart; the trajectory metrics separate them cleanly:

```
=== traj-eval summary ===
Trajectories:            15
Accuracy (final):        1.0000
Mean flip rate:          0.4286
Mean rationalization:    0.4569
Early convergence frac:  0.3333
Mean convergence step:   3.67
Median convergence step: 4.00
Mean convergence frac:   0.4583

Per-trajectory:
id                   final    gold   corr  conv  flip   rat    steps
--------------------------------------------------------------------
oscillation-2        B        B      Y     7     1.000  0.037  8
rationalization-4    B        B      Y     0     0.000  1.000  6
rationalization-2    B        B      Y     0     0.000  1.000  6
genuine-2            C        C      Y     4     0.286  0.333  8
oscillation-4        B        B      Y     7     1.000  0.037  8
genuine-4            C        C      Y     4     0.286  0.333  8
oscillation-1        B        B      Y     7     1.000  0.037  8
oscillation-0        B        B      Y     7     1.000  0.037  8
genuine-3            C        C      Y     4     0.286  0.333  8
genuine-1            C        C      Y     4     0.286  0.333  8
rationalization-1    B        B      Y     0     0.000  1.000  6
oscillation-3        B        B      Y     7     1.000  0.037  8
genuine-0            C        C      Y     4     0.286  0.333  8
rationalization-0    B        B      Y     0     0.000  1.000  6
rationalization-3    B        B      Y     0     0.000  1.000  6
```

Reading it: rationalization trajectories converge at step 0 with rationalization 1.000. Genuine trajectories converge at step 4 with rationalization 0.333. Oscillation trajectories flip every step (flip 1.000), converge only at the last step, and score low on rationalization because there is no stable verdict to rationalize. Same accuracy, three different stories.

## Caveats and limitations

- **Descriptive, not causal proof:** `traj-eval` measures when verdicts and rationale tokens appear and change along the denoising trajectory. It does not prove internal causal mechanisms.
- **Operational definition of rationalization:** `rationalization_score` and `rationalization_components` are heuristics based on timing and text change. A high score means a trajectory looks like rationalization by this operational definition, not that the model is proved to be rationalizing. Always report the decomposed components (`early_factor`, `post_change_rate`, `length_growth`, `gated_score`) rather than the scalar alone.
- **Sampling sensitivity:** Metrics depend on how `prediction` is extracted and how steps are sampled. Report your exact sampler configuration (canvas length, denoising steps, temperature, remasking rule, and EOS handling) alongside any measurements.
- **Synthetic vs. real data:** Synthetic data in `examples/synthetic_demo.jsonl` is for demos and unit tests only, not evidence about any real model checkpoint. Five-item runs in `results/` are pipeline sanity checks only and are excluded from benchmark tables.
- **Holdout checkpoint status:** `MMaDA-8B-MixCoT` has not been run yet.

## Contributing

See `CONTRIBUTING.md`. Run before opening a pull request:

```bash
python -m pytest -q
python -m traj_eval demo
python -m traj_eval run --input examples/synthetic_demo.jsonl
```

## Benchmarks and provenance

Full per-run provenance, GPU telemetry (`nvidia-smi` at launch), git revisions, sampler configurations, and scored JSON outputs for all runs are recorded in `results/baselines.md`. Scale and adversarial red-team benchmarks are documented in `docs/STRESS_TEST.md`.

## Citation

If you use `traj-eval`, please cite it as:

```bibtex
@software{traj_eval_2026,
  author  = {Chinda, Adele},
  title   = {traj-eval: Trajectory-level evaluation for diffusion language models},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/arrdel/traj-eval}
}
```

## License

MIT. See `LICENSE`.
