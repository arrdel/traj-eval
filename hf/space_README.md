---
title: traj-eval
emoji: 🧭
colorFrom: yellow
colorTo: red
sdk: static
app_file: index.html
pinned: true
license: mit
short_description: Denoising trajectory evaluation for diffusion LMs
models:
  - Dream-org/Dream-v0-Instruct-7B
  - GSAI-ML/LLaDA-8B-Instruct
  - GSAI-ML/LLaDA-1.5
datasets:
  - arrdel/traj-eval-benchmarks
tags:
  - masked-diffusion
  - diffusion-language-models
  - reasoning-trajectories
  - evaluation
---

# traj-eval: Measure reasoning in the denoising trajectory, not just final-answer accuracy

Static Hugging Face Space for `traj-eval`, an open-source toolkit and benchmark suite for evaluating denoising trajectories in masked diffusion language models.

Two models can have identical final-answer accuracy for very different reasons: genuine reasoning, rationalization, and oscillation. Final-step evaluation collapses the entire denoising process into a single endpoint prediction, hiding when a verdict stabilized and how the intermediate rationale developed. `traj-eval` instruments and scores the full step-by-step denoising trajectory so these reasoning regimes can be distinguished.

## What is on this page

1. **Why accuracy lies:** Plain-language definitions of genuine reasoning, rationalization, and oscillation, plus the v0.1 synthetic benchmark split (marked synthetic, not model evidence).
2. **One trajectory, two conditions:** Interactive step-by-step chart over denoising steps `0` to `255` (sampled every 16 steps) for ARC-Challenge items `Mercury_LBS10610` (reasoning condition) and `Mercury_7175875` (direct condition) on `Dream-v0-Instruct-7B` (shown to illustrate the trajectory pattern under each condition, not a paired comparison on the same item).
3. **Results (64-item suites only):** Verified measurements across `Dream-v0-Instruct-7B`, `LLaDA-8B-Instruct`, and `LLaDA-1.5` on 64-item ARC-Challenge (`direct` and `reasoning`) and GSM8K (`numeric`) suites, plus the side-by-side rationalization component decomposition for `Dream-v0-Instruct-7B`.
4. **Metrics, install, quickstart, and caveats:** Operational definitions from `src/traj_eval/metrics.py`, zero-weight CLI/Python quickstart (`numpy` only, Python >= 3.10), and explicit caveats.

## Links

- **GitHub Repository:** [https://github.com/arrdel/traj-eval](https://github.com/arrdel/traj-eval)
- **GitHub Pages Site:** [https://arrdel.github.io/traj-eval/](https://arrdel.github.io/traj-eval/)
- **Companion Benchmark Dataset:** [https://huggingface.co/datasets/arrdel/traj-eval-benchmarks](https://huggingface.co/datasets/arrdel/traj-eval-benchmarks)
- **Hugging Face Collection:** [https://huggingface.co/collections/arrdel/traj-eval-diffusion-lm-reasoning-trajectories-6af321457d8dbd6074548379](https://huggingface.co/collections/arrdel/traj-eval-diffusion-lm-reasoning-trajectories-6af321457d8dbd6074548379)
- **Author:** Adele Chinda (PhD Candidate in NLP, Georgia State University)
