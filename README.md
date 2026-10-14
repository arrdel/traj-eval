# traj-eval

Trajectory-level evaluation for diffusion language models.

**Measure reasoning *in* the denoising trajectory, not just final-answer accuracy.**

## Why final-answer accuracy lies

Two models can have identical final-answer accuracy for very different reasons:

- **Genuine reasoning:** the verdict changes as reasoning develops, then stabilizes. Reasoning drives the verdict.
- **Rationalization:** the verdict is fixed in the first few steps and reasoning is elaborated afterwards to justify it. Verdict drives reasoning.
- **Oscillation:** the verdict flips repeatedly and never settles, even if the final answer happens to be right.

`traj-eval` makes those patterns visible. It is descriptive, not causal proof. A high `rationalization_score` means "looks like rationalization by this operational definition", not "proved to be rationalizing".

This is v0.1. It works on trajectory data in a simple JSONL/JSON format, with pure Python + numpy. No model weights, no torch, no transformers needed. Adapters for LLaDA, Dream, and other diffusion LMs are on the roadmap, not in v0.1.

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
```

Python API:

```python
from traj_eval.io import load_trajectories
from traj_eval.metrics import evaluate_trajectories

trajs = load_trajectories("examples/synthetic_demo.jsonl")
results = evaluate_trajectories(trajs)
print(results["aggregate"])
```

See `notebooks/01-quickstart.ipynb` for a worked notebook (runs without heavy deps; matplotlib is optional).

## Input format

JSONL, one trajectory per line (JSON list, or `{"trajectories": [...]}` also accepted):

```json
{"id": "q1", "prompt": "What is ...?", "gold_answer": "B",
 "steps": [
   {"step": 0, "prediction": "A", "reasoning_text": "Initial guess..."},
   {"step": 1, "prediction": "B", "reasoning_text": "Reconsidering..."},
   {"step": 2, "prediction": "B", "reasoning_text": "B fits best."}
 ]}
```

Fields:

- `id` (str, required), `prompt` (str, required), `gold_answer` (str, required; use `""` if unknown, accuracy will skip it)
- `steps` (list, required; may be empty, handled gracefully)
- Per step: `step` (int, required), `prediction` (str, required), `reasoning_text` (str, optional), `confidence` (float, optional), `logits` (list, optional)

Malformed files raise `ValueError` with a clear message instead of silently dropping data.

## Metrics

| Metric | Function | Range | What it means |
|---|---|---|---|
| Verdict convergence step | `verdict_convergence_step` | step number or None | First step after which prediction stays constant to the end. None only for empty trajectory. |
| Flip rate | `flip_rate` | [0, 1] | Fraction of adjacent pairs where prediction changes. 0 = never changed, 1 = changed every step. |
| Step consistency | `step_consistency` | dict | `1 - flip_rate` plus `longest_stable_run` and its fraction. |
| Trajectory accuracy | `trajectory_accuracy` | [0, 1] | Fraction where final prediction == gold. Skips empty / unknown-gold trajectories. |
| Forced deliberation delta | `forced_deliberation_delta` | dict | Given `base_accuracy` and `forced_accuracy`, returns `delta = forced - base` and `relative_drop = (base - forced) / base`. Negative delta means forcing longer deliberation hurt accuracy. |
| Rationalization score | `rationalization_score` | [0, 1] | **Heuristic.** High when verdict converges early AND reasoning keeps changing/lengthening after convergence. Formula: `0.4*early_factor + 0.4*post_change_rate + 0.2*length_growth`. Not ground truth. |
| Early convergence fraction | `early_convergence_fraction` | [0, 1] | Fraction of trajectories converging within the first 20% of steps (threshold configurable). |
| Aggregate | `evaluate_trajectories` | dict | Per-trajectory results + aggregate: accuracy, mean/median convergence step and fraction, mean flip rate, mean rationalization score, early convergence fraction. |

Edge cases: empty trajectory returns None / 0.0 instead of crashing. Single-step trajectory converges at that step, flip rate 0.0.

## Real example output

From `python -m traj_eval run --input examples/synthetic_demo.jsonl` (15 synthetic trajectories, 5 per archetype). Note the key lesson: **accuracy is 1.0000 for all three archetypes**, because the synthetic data is built so every final answer is correct. Accuracy alone cannot tell them apart. The trajectory metrics can:

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

## Limitations

- v0.1 does not include model adapters. You bring trajectories; it measures them. LLaDA / Dream adapters are roadmap.
- `rationalization_score` is a heuristic based on timing and text change. It does not detect intent, and it can be gamed by paraphrasing. Use it to flag trajectories for inspection, not as a verdict.
- Metrics depend on how you define `prediction` and how often you sample steps. Sample consistently, and report your sampling in any paper or post that uses this tool.
- Synthetic data is for demos and tests only, not evidence about any real model.
- Confidence and logits are stored but not yet used by metrics in v0.1.

## Roadmap

- [ ] Adapters: `src/traj_eval/adapters/` for LLaDA-8B, Dream-7B, and other open diffusion LMs (return standard `Trajectory` objects; tests use stub models, never download weights in tests)
- [ ] More metrics: calibration across steps, semantic stability (not just exact prediction match), per-step confidence curves
- [ ] Forced-deliberation experiment helper: run base vs forced on the same prompts and report the delta table
- [ ] Visualization helpers: convergence histograms, flip-rate distributions
- [ ] Larger real-trajectory example dataset (with clear provenance)

## Contributing

See `CONTRIBUTING.md`. Run before a PR:

```bash
python -m pytest -q
python -m traj_eval demo
python -m traj_eval run --input examples/synthetic_demo.jsonl
```

## Citation

If you use `traj-eval` in research, please cite it as:

```
@software{traj_eval_2026,
  author = {Chinda, Adele},
  title = {traj-eval: Trajectory-level evaluation for diffusion language models},
  year = {2026},
  version = {0.1.0}
}
```

## License

MIT. See `LICENSE`.
