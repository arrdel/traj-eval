# Scoping spec — Dream adapter for traj-eval

Status: **spec only. No adapter code is written.** This file scopes tomorrow's first real-checkpoint run and records what was verified today (19 Oct 2026). Nothing here scores a model yet.

## Three confirmations requested

1. **Dream-first, not LLaDA-8B.** Confirmed as the plan. Start model is `Dream-org/Dream-v0-Instruct-7B` (the Instruct variant). Do **not** use `Dream-org/DreamOn-v0-7B` (infilling); its page is a code-pattern reference only.
2. **ARC-C first.** Confirmed as the plan. Unambiguous A–D option-letter verdict via regex. Expand to ARC-C + GSM8K only after the ARC-C smoke is clean.
3. **Regex + null-for-undecided verdict rule.** Confirmed as the plan. Parse the final option letter A–D; if no letter is decodable at a step, record `prediction: null` (no-verdict), never a guess.

These three are the design. The spec below is built on them.

## Two uncertainties — now checked

### U1. Does `diffusion_generate` with history exist on Dream-v0-Instruct-7B itself?

**Confirmed, directly on the Instruct repo**, not inferred from DreamOn. `Dream-org/Dream-v0-Instruct-7B/generation_utils.py` defines:

- `DreamModelOutput(sequences, history)`.
- The sampling loop `for i in range(steps):` and, when `return_dict_in_generate and output_history`, `histories.append(x.clone())` at the end of every step, returning `history=histories`.

So the per-step mechanism is real on the Instruct model. The HKUNLP README shows the exact call on `Dream-org/Dream-v0-Instruct-7B`.

### U2. Local requirements (20 GB VRAM, transformers 4.46.2, torch 2.5.1)

**Checked on this machine:**

| Requirement | Status here |
|---|---|
| transformers 4.46.2 | Present in the `cdr-gpu` conda env (`/home/achinda1/miniconda3/envs/cdr-gpu`). Exactly 4.46.2. |
| torch 2.5.1 | `cdr-gpu` has `torch 2.5.1+cu121`, and `torch.cuda.is_available()` is **True** there. |
| ≥ 20 GB VRAM | 8× RTX A6000, 49 GB each. Fine per card. |
| Base env | Unusable for GPU: `torch 2.11.0+cu130`, `transformers 5.5.3`, CUDA not available (driver 12010 too old for cu130). The repo `.venv` has no transformers at all. |

**Conclusion:** run from the `cdr-gpu` env, not base and not `.venv`. It already matches Dream's pinned stack.

GPU occupancy is still high (today: 4 cards ~33 GB used, 4 cards ~16 GB used, all 78–100% utilization). ~16 GB free on the lighter cards is below the 20 GB headroom for a bf16 7B plus activations. **Pick a card with ≥ ~25 GB free at run time**; do not assume one is free. Set `CUDA_VISIBLE_DEVICES` to that single card.

## One correction to the proposed call

The proposed signature listed `number_transfer_tokens=1` as a user argument. In `generation_utils.py`, **`number_transfer_tokens` is computed internally** from the masking schedule:

```
number_transfer_tokens = int(num_mask_token * (1 - s / t))   # i < steps-1
number_transfer_tokens = int(num_mask_token)                 # last step
```

It is **not** a kwarg. The user-facing control is `steps`: roughly `max_new_tokens / steps` tokens are unmasked per step. To approximate one position per step, set `steps == max_new_tokens`. Record `steps`, not a transfer-token count, in the run log.

So the confirmed call for the adapter is:

```
output = model.diffusion_generate(
    input_ids, attention_mask=attention_mask,
    max_new_tokens=M, steps=S,
    output_history=True, return_dict_in_generate=True,
    temperature=T, top_p=P, alg="entropy", alg_temp=0.0,
)
history = output.history      # list, one cloned full-length tensor per step
sequences = output.sequences  # final tokens
```

## History semantics the adapter must respect

Each `history[i]` is `x.clone()`: the **full** sequence of length `input_len + max_new_tokens`, padded with `mask_token_id`, containing both the prompt and the generation region, with still-masked positions equal to `mask_token_id`. Consequences:

- **Slice off the prompt** (`input_ids.shape[1]`) before decoding the generation region.
- **Strip mask tokens** from the region before regex. A step early in the trajectory may be almost all masks → regex finds nothing → `prediction: null`. That is the intended no-verdict case, not an error.
- `len(history) == steps`. Emit one traj-eval step per entry: `step = i`, `prediction = <A-D or null>`, `reasoning_text = <decoded, mask-stripped generation region>`.
- The final parsed verdict should match the verdict parsed from `sequences`. Assert this as a sanity check during the smoke run.

## Verdict rule (ARC-C)

- Decode the mask-stripped generation region at step `i`.
- Regex the **final** standalone option letter A–D (e.g. the last match of `\b([A-D])\b`, with answer-prefix patterns like `answer is (A)` taking precedence, mirroring how the eval prompt asks for the letter).
- No decodable letter → `prediction: null`. Null steps still count toward the trajectory length and flip-rate denominator semantics as defined in traj-eval; a flip is only counted between two non-null, differing verdicts (confirm this against `metrics.flip_rate` when wiring — v0.1 compares raw `prediction` strings, so decide whether null is a distinct value or skipped, and write the decision into the adapter, not the metrics).
- `gold_answer` is the ARC-C correct letter.

Open wiring decision to settle in the adapter (not in this spec): whether a `null → "B"` transition is a flip. Recommendation: treat null as "no verdict yet" and count the first flip only between two emitted letters, so early all-mask steps do not inflate flip rate. Document whichever is chosen in the run log.

## Prompt and length

- Chat template via `tokenizer.apply_chat_template(..., add_generation_prompt=True)`.
- Dream context (input + output) is **2048 tokens**. ARC-C prompts are short; a small `max_new_tokens` (e.g. 256) is plenty and keeps `steps` cheap.
- `max_new_tokens` must be ≥ the mask count being denoised (it defines the canvas). For ARC-C a letter-plus-brief-rationale canvas is small; do not over-allocate, since `steps` scales cost with it.

## Smoke protocol (run unchanged tomorrow)

1. Env: `cdr-gpu`. One GPU with ≥ 25 GB free, pinned via `CUDA_VISIBLE_DEVICES`.
2. Data: **5 ARC-C items**, ids written to `results/dream_arcc_smoke5_ids.json` when the run starts.
3. Sampler: `alg="entropy"`, `alg_temp=0.0`, `temperature=0.2`, `top_p=0.95`, `max_new_tokens=256`, `steps=256`. (Start from Dream's documented defaults; temperature 0.0 is also acceptable and more deterministic — pick one and log it.)
4. For each item: run `diffusion_generate` with history, emit one traj-eval JSONL trajectory (one step per `history` entry).
5. Score with `python -m traj_eval run`. Paste the aggregate into `results/baselines.md` with: git revision, env, GPU index, and the full sampler config (`temperature`, `top_p`, `alg`, `alg_temp`, `steps`, `max_new_tokens`).
6. Sanity asserts in the smoke: `len(history) == steps`; final parsed verdict == verdict parsed from `sequences`; at least the last step is non-null on items the model answers.

Only after the 5-item smoke is clean: expand to the full ARC-C set, then add GSM8K. **GSM8K caveat:** 8-shot CoT makes verdict extraction harder (the numeric answer is embedded in a long rationale, and few-shot exemplars contain their own answers). Needs a separate numeric-answer extraction rule and exemplar-stripping before it enters the harness. Do not reuse the A–D regex for it.

## Reference numbers (context, not our results)

Dream's paper reports ARC-C 59.8 (0-shot) vs LLaDA-8B 47.5. That is their number, for orientation on whether our extraction is sane. It is **not** a traj-eval result and does not go in the baselines table as one.

## What is explicitly out of scope for tomorrow

- Token-commitment logging (Yeom et al. style `answer_committed`). The history gives us what we need to add it later (a position is committed once it leaves `mask_token_id` and never returns), but the first run is parsed-verdict only.
- LLaDA and MMaDA adapters.
- Forced deliberation.
