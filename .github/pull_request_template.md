## What changed and why


## Checklist (fill in every item, do not delete items)

- [ ] One logical change only; branch is not `master`
- [ ] `python -m pytest -q` tail pasted below (verbatim, after this change)
- [ ] Tests added or updated for any metric, adapter, or extractor change
- [ ] No weights in tests; no torch/transformers import at module import time in `src/`
- [ ] Null rule unchanged (`None` = no verdict, not a flip), or the change is documented in metrics, adapter, handoff, and paper outline together
- [ ] Prompt template unchanged, or the new prompt is pre-registered with its own output files (never overwriting an existing condition)
- [ ] Sampler config unchanged mid-run; any GPU run includes its run log path
- [ ] No paper claim beyond what the artifacts in this PR prove (5 items = pipeline check only)
- [ ] Scope check: this PR does not add a new adapter, metric, credit scheme, intervention hook, or prompt condition unless the current follow-up doc explicitly allows it

## Test output (verbatim tail)

```
```

## GPU run (if any)

- Run log path:
- GPU index and free memory at launch:
- Sampler config (temperature, top_p, alg, alg_temp, steps, max_new_tokens):
- Gate result (pass / fail, with reason if fail):
