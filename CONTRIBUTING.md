# Contributing to traj-eval

Thanks for contributing. This repo is built to be friendly to Cursor agents
and human contributors alike. Keep changes small, tested, and honest about
what is a heuristic vs. what is measured.

## Quickstart for contributors

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest -q
python -m traj_eval demo
```

## Workflow

1. Open an issue or note what you are changing and why.
2. Make the smallest change that adds value. Do not refactor unrelated code
   in the same PR.
3. Add or update tests for any new metric, loader, or CLI behaviour.
4. Run the full check before opening a PR:
   ```bash
   python -m pytest -q
   python -m traj_eval demo
   python -m traj_eval run --input examples/synthetic_demo.jsonl
   ```
5. Update the README if you change user-facing behaviour, metrics, or
   the file format.

## Style

- Python 3.10+, typed, with docstrings on all public functions.
- Keep dependencies light. Do not add `torch`, `transformers`, or other
  heavy deps without discussion in an issue first. v0.1 is pure
  Python + numpy on purpose.
- Format with clear, readable code. No formatter is enforced yet; keep
  diffs clean and consistent with surrounding code.

## How to add a new metric

1. Implement it in `src/traj_eval/metrics.py` with a docstring that states:
   - what it measures,
   - its range and edge cases (empty, single step),
   - whether it is descriptive or a heuristic.
2. Wire it into `evaluate_trajectories()` if it should appear in
   per-trajectory / aggregate output.
3. Add tests in `tests/test_metrics.py` covering at least: a normal case,
   an edge case (empty / single step), and the expected range.
4. Document it in the README metrics table.

## How to add a model adapter (roadmap)

Adapters for real models (LLaDA, Dream, etc.) live in a future
`src/traj_eval/adapters/` package. An adapter should:

- Take a model / generation config and return a list of `Trajectory`
  objects in the standard format.
- Not change the metric definitions.
- Ship with a small test that uses a fake / stub model, not real weights.

Do not submit an adapter that downloads weights in tests.

## Honesty rules

- Label `rationalization_score` and any future heuristic as a heuristic.
- Do not claim a metric proves reasoning or its absence.
- Report limitations in the README when you add a feature.

## Questions

Open an issue. For research discussion, include a small reproducible
example (synthetic trajectories are fine) so others can check the claim.
