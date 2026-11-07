---
title: traj-eval Trajectory Explorer
emoji: 🔬
colorFrom: indigo
colorTo: blue
sdk: static
pinned: true
license: mit
short_description: Interactive denoising trajectory explorer for diffusion LMs
---

# `traj-eval`: Interactive Denoising Trajectory Explorer & Benchmark Leaderboard

Interactive companion for ***"Same accuracy, different trajectory: measuring reasoning regimes in masked diffusion language models"*** (Adele Chinda, Georgia State University).

- **Scrub step-by-step ($t = 1 \dots 256$)** through real denoising trajectories from `Dream-org/Dream-v0-Instruct-7B`, `GSAI-ML/LLaDA-8B-Instruct`, and `GSAI-ML/LLaDA-1.5`.
- **Compare reasoning clocks:** Verdict convergence ($T_{\text{stab}}$), token commitment ($T_{\text{commit}}$), unwritten reasoning fraction ($U = 1 - F_R(T_{\text{stab}})$), order gap, and multiplicative gated rationalization score.
- **Inspect your own `.jsonl` files:** Drag and drop any `traj-eval` `.jsonl` file directly in the browser to visualize and audit custom runs client-side.
