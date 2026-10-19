# Paper 1 outline — traj-eval

Working title: **Same accuracy, different trajectory: measuring reasoning regimes in masked diffusion language models.**

Target: ICML 2027 or NeurIPS 2027. This file is the outline, not the draft. Claims below are split into what v0.1 already supports and what is still an experiment.

## The claim we can still own

Final-answer accuracy does not say whether a masked diffusion language model reasoned, rationalized, or oscillated. A shared trajectory protocol can separate those regimes, and the separation is the measurement that later training work has to condition on.

That is narrower than "masked diffusion models rationalize." That phenomenon is already published. The paper lives or dies on a reusable instrument plus a result those papers do not report: on the same trajectories, parsed-verdict stability, irreversible token commitment, and post-stability reasoning-text change are different events, and an outcome-only RL advantage copied onto every denoising step cannot tell them apart.

## 0. What v0.1 is

`traj-eval` 0.1.0 scores JSONL trajectories. It does not load weights. Public entry points:

- `verdict_convergence_step` — first step after which the parsed prediction stays constant.
- `flip_rate` — fraction of adjacent steps whose parsed prediction changes.
- `rationalization_score` — heuristic in [0, 1]. High when the verdict converges early and `reasoning_text` keeps changing or lengthening afterwards. Formula: `0.4 * early_factor + 0.4 * post_change_rate + 0.2 * length_growth`.
- `trajectory_accuracy` — final prediction versus gold.
- `forced_deliberation_delta` — arithmetic on two accuracies the caller supplies. It does not run a forced-deliberation experiment.
- `early_convergence_fraction` — share of trajectories whose convergence index is within the first 20% of steps.

The only numbers in the repo today are synthetic. On 15 hand-built trajectories (5 genuine, 5 rationalization, 5 oscillation), accuracy is 1.0 for every archetype. Mean rationalization is 1.0, 0.3333, and 0.0375 respectively. See `results/baselines.md`. Those numbers validate the metric, not any model.

## 1. Problem

Autoregressive chain-of-thought writes the rationale before the answer, so display order and conditioning order coincide. A masked diffusion language model starts from a fully masked canvas and unmasks many positions per step. The string that looks like a rationale can be filled in after the answer span is already stable.

Two models, or two prompts, can therefore post the same accuracy for different reasons:

- **Genuine revision.** The parsed verdict moves as the canvas fills, then holds. Reasoning text settles with it.
- **Rationalization, operationally.** The parsed verdict is fixed in the first few steps, and the reasoning text keeps changing or growing afterwards.
- **Oscillation.** The parsed verdict flips through the trajectory and is stable only because generation ended.

Final-answer benchmarks collapse all three to one bit. A training method that broadcasts one outcome reward onto every denoising step (diffu-GRPO) then treats those steps as equally responsible for the answer.

The paper's job is to measure the regimes. It is not to prove intent. `rationalization_score` means "looks like rationalization under this definition."

## 2. Related work

Write this section as a difference, not a survey. The closest papers already measured pieces of the phenomenon on overlapping models.

| Paper | What it measures | Models | What it does not give us |
|---|---|---|---|
| Yeom et al., [Answer First, Reason Later](https://www.alphaxiv.org/abs/2608.05687) (Aug 2026) | Irreversible token commitment. Answer-first if the answer span commits before the median reasoning token. Order gap, backfill, frontier gating, a timing-matched delay of a reserved answer slot. | LLaDA-8B-Instruct, LLaDA-1.5, Dream-7B. GSM8K, MATH-500, a 4-option control, HumanEval/MBPP as a negative control. | A reusable harness. Parsed-verdict flip rate. A text-change rationalization score. Oscillation. MMaDA. An implication for credit assignment. |
| Kim and Choi, [Two Clocks](https://www.alphaxiv.org/abs/2610.00953) (Oct 2026, NeurIPS workshop) | `T_stab`: logged answer candidate matches the final answer and never leaves it. `T_commit`: the answer position is permanently written. Unwritten fraction of the rationale-side canvas at stabilization. | LaViDa, Nemotron, DiffusionGemma. V\*Bench, M³CoT, ScienceQA. | Text diffusion checkpoints. A distinction between an unwritten canvas and reasoning text that is already written but still changing. |
| Li et al., arXiv:2508.19982, as summarized by Kim and Choi | Early answer convergence and the Prophet early-commit decoder. Their main analysis tracks the first ground-truth token match on finally correct outputs. | Not re-read for this outline. | Confirm against the PDF before this paragraph is frozen. |
| Zhao et al., [d1](https://www.alphaxiv.org/abs/2504.12216) | diffu-GRPO: one outcome advantage, unnormalized (`r - mean`), copied onto every token of the sampled sequence. SFT + RL on LLaDA-8B-Instruct. | LLaDA-8B-Instruct. Math and planning. | Any use of when the verdict stabilized. This is the naive-RL baseline for Paper 2. |
| Shao et al., [DeepSeekMath](https://www.alphaxiv.org/abs/2402.03300); DeepSeek-AI, [DeepSeek-R1](https://www.alphaxiv.org/abs/2501.12948) | GRPO. Group-relative advantage, no learned value head. R1-Zero uses a verifiable outcome reward and no reasoning SFT. | Autoregressive models. | Diffusion trajectories. Per-step credit. |
| Turpin et al. 2023; Lanham et al. 2023 | Unfaithful chain-of-thought in autoregressive models. | AR LMs. | A denoising trajectory. Yeom et al. already connect to these. |

Faithfulness probes, process reward models, and block-diffusion decoders belong in one paragraph each. Do not let them become the paper.

**Positioning sentence.** Yeom et al. show that commitment order can cost accuracy, and that the cost depends on model and interface. Kim and Choi separate candidate stabilization from commitment in multimodal diffusion models. traj-eval is the protocol that puts parsed-verdict dynamics, commitment, and post-stability text change in one file, on the text checkpoints a training paper will actually update.

Do not write "we show that diffusion models answer first." Yeom et al. showed a version of that, with interventions, on the same first two model families.

## 3. Method

### 3.1 Trajectory schema

One JSON object per prompt. Required: `id`, `prompt`, `gold_answer`, `steps[]` with `step` and `prediction`. Optional and already stored: `reasoning_text`, `confidence`, `logits`.

Paper extension, not in v0.1 yet. Each step also records:

- `canvas` — decoded text at that step, masks explicit.
- `answer_committed` — whether the answer span is visible and was never remasked afterwards. This is Yeom et al.'s commitment, logged forward.
- `committed_reasoning_fraction` — fraction of the reasoning region already committed. This is the complement of Kim and Choi's unwritten fraction, on text.

`prediction` stays the parsed verdict (the letter or the normalized number), not a token id. That is the point. Token commitment and the parsed answer can move on different clocks.

Sampling has to be reported with every number: canvas length, denoising steps, remasking rule, temperature, EOS handling. Yeom et al. show EOS suppression changes answer-first rates from 17% to 77% on their GSM8K setup. An unreported sampler makes the table incomparable.

### 3.2 Metrics

Keep the v0.1 metrics. Add, once the log exists:

- **Commitment step** of the answer span, and **order gap** against the median reasoning-token commitment. Same definitions as Yeom et al., implemented in this repo so the number is reproducible from the JSONL.
- **Stability–commitment lag.** Parsed verdict has entered its stable suffix, but the answer span is not yet irreversibly committed. This is the gap between v0.1's convergence step and `T_commit`.
- **Post-stability edit rate.** Fraction of post-convergence steps whose `reasoning_text` changes. Already inside `rationalization_score`; report the component, not only the weighted sum.
- **Disagreement slice.** Trajectories where the three clocks disagree: verdict string stable while the canvas is still being written, or answer tokens committed while the parsed verdict later flips because the span was re-parsed. The size of this slice is a main result. If it is empty, say so and shrink the claim.

`rationalization_score` stays labeled heuristic. Do not replace the components with the scalar in the main table.

### 3.3 Forced deliberation

A separate condition, not a new metric formula. After the parsed verdict has been stable for K steps, continue denoising for a fixed extra budget with the answer span still eligible for remasking. Report `forced_deliberation_delta` against the matched unrestricted run.

This is not frontier gating. Frontier gating changes which positions may commit. Forced deliberation changes how long a stable verdict is allowed to keep being revised. Both can be run. Only the second one is about "more thinking." Yeom et al. already showed that later commitment is not monotonically better. The paper should be ready for a null or a negative delta.

## 4. Experiments

### Checkpoints

Text trio, chosen so the table can sit next to Yeom et al.:

| Checkpoint | Hugging Face id | Why this one |
|---|---|---|
| LLaDA-8B-Instruct | `GSAI-ML/LLaDA-8B-Instruct` | Masked diffusion trained from scratch, then SFT. The d1 baseline. |
| LLaDA-1.5 | `GSAI-ML/LLaDA-1.5` | Same family plus VRPO. Yeom et al. find a similar backfill pattern to LLaDA-8B. Tests whether preference optimization moves the trajectory or only the accuracy. |
| Dream-v0-Instruct-7B | `Dream-org/Dream-v0-Instruct-7B` | Initialized from an autoregressive model, then adapted. Yeom et al. find little free-form MATH backfill, then very early commitment once an answer slot is reserved. |

MMaDA-8B-MixCoT (`Gen-Verse/MMaDA-8B-MixCoT`) is the modality holdout, not the third text model. Run it after the text trio is comparable. Putting it in the first table would mix a multimodal mixed-CoT checkpoint into a commitment-order comparison that has no published MMaDA numbers to check against.

### Tasks

1. **Smoke (first GPU window).** 64 GSM8K test questions, fixed ids, written into the repo before the run. Canvas 256, 256 steps, temperature 0, low-confidence remasking, EOS suppression on. One seed. Goal: adapters emit valid JSONL and the metric code runs.
2. **Main text table.** GSM8K test, reporting the 1,069-question split Yeom et al. treat as the held-out estimate, and MATH-500. Same sampler family as the smoke run, with canvas length as a factor on MATH only (256 and 512) if the smoke run is cheap enough. Do not start at 512 × 1,069 × 3 models.
3. **Forced deliberation.** On the smoke set first. K and the extra budget are pre-registered in `results/baselines.md` before looking at accuracy.
4. **Disagreement slice.** On the main set. Count trajectories where verdict convergence, answer commitment, and post-stability text change do not move together. Stratify by correct versus incorrect. Kim and Choi find high unwritten fractions in both correctness groups; do not assume the slice is only the errors.
5. **Negative control, if time.** A short HumanEval subset. Yeom et al. find no resolved frontier-gating gain where the answer is not a compact span. If our rationalization score lights up there anyway, the heuristic is too coarse and the paper has to say so.

### What gets reported

Per checkpoint and task: accuracy, mean flip rate, early-convergence fraction, mean convergence fraction, the three components of the rationalization heuristic, answer-commitment step, order gap, and the disagreement rate. Paired bootstrap over questions, following Yeom et al., once the main set exists. No leaderboard prose around a 64-question smoke run.

### Compute

Eight RTX A6000s are on this machine. On 17 Oct 2026 they were full (about 43 GB of 49 GB, utilization at 100% on every device). The system PyTorch build is CUDA 13 and does not see the driver; the working stack elsewhere is `torch 2.5.1+cu121`. Do not start a download-and-eval job against occupied GPUs. The first real numbers go in `results/baselines.md` with the sampler config and the git revision that scored them.

## 5. Claims

Write only the rows whose status is "measured" at submission time. The others stay in this outline.

| # | Claim | Status |
|---|---|---|
| C0 | On synthetic trajectories built so that every final answer is correct, accuracy is 1.0 and the trajectory metrics separate genuine revision, rationalization, and oscillation. | Measured. Not evidence about a model. |
| C1 | On real checkpoints, accuracy is a weak description of the trajectory: the same accuracy band contains different flip rates and convergence times. | Not measured. |
| C2 | Parsed-verdict stability, answer-span commitment, and post-stability reasoning-text change disagree on a non-trivial fraction of trajectories. | Not measured. This is the result that is still open after Yeom et al. and Kim and Choi. |
| C3 | Forcing extra denoising after the parsed verdict has stabilized changes accuracy by a reported delta, which may be negative. | Not measured. |
| C4 | LLaDA-8B-Instruct, LLaDA-1.5, and Dream-7B do not share one trajectory signature, so a single "diffusion reasoning" number is the wrong summary. | Expected from Yeom et al. Our job is to recover it with this harness, not to cite it as our discovery. |
| C5 | An RL method that copies one outcome advantage onto every denoising step assigns credit to steps after the verdict has already stabilized, on the trajectories where C2 holds. | Hypothesis. It is the bridge to Paper 2. It is not a theorem, and it is false on trajectories that revise the verdict late. |

Explicit non-claims:

- The rationalization heuristic is not a proof of unfaithfulness.
- Synthetic numbers are not a baseline for any checkpoint.
- Frontier-gating gains reported by Yeom et al. are not traj-eval results.
- A null disagreement slice is a real result. It would mean v0.1's parsed verdict is redundant with commitment, and the paper becomes a consolidation plus the forced-deliberation delta.

## 6. Implications for training (one section, then stop)

If C2 holds, diffu-GRPO's broadcast advantage is the wrong inductive bias on rationalizing trajectories and a reasonable one on trajectories that revise late. Paper 2 should compare:

- broadcast outcome advantage (d1 / diffu-GRPO),
- advantage only on steps up to verdict convergence,
- a credit rule that can still update after convergence when the verdict later flips.

That comparison is not this paper. This paper's last section states the measurement those three conditions need, and stops.

## 7. Figure list

1. One trajectory from each regime, synthetic, as the method figure. The README table is the sketch.
2. Scatter of accuracy versus mean convergence fraction, one point per checkpoint × task. The failure of accuracy is visual only if the x-axis actually spreads.
3. Disagreement slice: a 3-way count of (verdict stable, answer committed, reasoning text still changing).
4. Forced-deliberation delta with a paired interval. Include the null if that is what happens.

## 8. Nearest writing order

1. Problem and the non-claim paragraph. One page.
2. Method, including the heuristic formula and the sampler fields. The README is the source.
3. Related work, after Li et al. 2508.19982 has been read in the PDF.
4. Experiments, filled only from `results/`.
5. Claims, copied from the measured rows of the table above.
