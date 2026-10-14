# Examples

## synthetic_demo.jsonl

Generated synthetic trajectories for quick testing. Three archetypes,
5 of each by default (15 total):

- `genuine-*`: verdict changes gradually, then stabilizes. Reasoning
  stabilizes after convergence.
- `rationalization-*`: verdict fixed from step 0, reasoning keeps
  elaborating afterwards.
- `oscillation-*`: verdict flips repeatedly, never settles.

Regenerate it with:

```bash
python -m traj_eval example --output examples/synthetic_demo.jsonl
```

Then evaluate:

```bash
python -m traj_eval run --input examples/synthetic_demo.jsonl --output results.json
```

## Format

Each line is one trajectory:

```json
{"id": "genuine-0", "prompt": "...", "gold_answer": "C",
 "steps": [{"step": 0, "prediction": "A", "reasoning_text": "..."}, ...]}
```

See `README.md` for the full format and metric definitions.
