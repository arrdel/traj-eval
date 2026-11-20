"""Generate Figures 1, 2, and 3 for the P1 demo ('Watch a diffusion model change its mind')."""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from traj_eval.io import load_trajectories
from traj_eval.metrics import flip_rate, verdict_convergence_step

OUT_DIR = Path("results/p1_figures")


def build_fig1() -> Path:
    llada_d = load_trajectories("results/llada_instruct_arcc_smoke64_direct.jsonl")
    llada_r = load_trajectories("results/llada_instruct_arcc_smoke64_reasoning.jsonl")
    dream_d = load_trajectories("results/dream_arcc_smoke64_direct.jsonl")
    dream_r = load_trajectories("results/dream_arcc_smoke64_reasoning.jsonl")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    bins = np.linspace(0, 256, 33)

    for ax, (title, d_trajs, r_trajs, sampler_note) in zip(
        axes,
        [
            ("LLaDA-8B-Instruct", llada_d, llada_r, "temp=0.0, low-conf remasking, 256 steps"),
            ("Dream-v0-Instruct-7B", dream_d, dream_r, "temp=0.2, top_p=0.95, entropy, 256 steps"),
        ],
    ):
        d_steps = [s for t in d_trajs if (s := verdict_convergence_step(t)) is not None]
        r_steps = [s for t in r_trajs if (s := verdict_convergence_step(t)) is not None]
        d_med = statistics.median(d_steps)
        r_med = statistics.median(r_steps)

        ax.hist(d_steps, bins=bins, alpha=0.65, color="#2b6cb0", label=f"Direct (n={len(d_steps)}, med={d_med:.0f})")
        ax.hist(r_steps, bins=bins, alpha=0.65, color="#dd6b20", label=f"Reasoning (n={len(r_steps)}, med={r_med:.0f})")
        ax.axvline(d_med, color="#1a365d", linestyle="--", linewidth=1.8)
        ax.axvline(r_med, color="#9c4221", linestyle="--", linewidth=1.8)
        ax.set_title(f"{title}\n({sampler_note})", fontsize=11)
        ax.set_xlabel("Convergence step (0 to 256)")
        ax.set_xlim(0, 256)
        ax.legend(frameon=True, fontsize=9)
        ax.grid(axis="y", alpha=0.25)

    axes[0].set_ylabel("Question count (ARC-C 64-item paired)")
    fig.suptitle("Figure 1: Answer commitment timing, direct vs. reasoning prompts", fontsize=13, y=1.02)
    fig.tight_layout()
    out = OUT_DIR / "fig1_commitment_distribution.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def build_fig2() -> Path:
    trajs = load_trajectories("results/llada_instruct_arcc_smoke64_reasoning.jsonl")
    flips = [t for t in trajs if flip_rate(t) > 0][:3]
    no_flips = [t for t in trajs if flip_rate(t) == 0 and (verdict_convergence_step(t) or 0) >= 15][:3]
    selected = flips + no_flips

    letter_to_y = {None: 0, "A": 1, "B": 2, "C": 3, "D": 4}
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.2), sharex=True, sharey=True)
    for ax, traj in zip(axes.flat, selected):
        xs = [s.step for s in traj.steps]
        ys = [letter_to_y.get(s.prediction, 0) for s in traj.steps]
        ax.step(xs, ys, where="post", color="#2d3748", linewidth=1.8)
        conv = verdict_convergence_step(traj)
        if conv is not None:
            ax.axvline(conv, color="#dd6b20", linestyle="--", linewidth=1.2, label=f"conv={conv}")
        gold_y = letter_to_y.get(traj.gold_answer, 0)
        ax.axhline(gold_y, color="#38a169", linestyle=":", linewidth=1.2, label=f"gold={traj.gold_answer}")
        tag = "flip" if flip_rate(traj) > 0 else "no-flip"
        ax.set_title(f"{traj.id} ({tag}, final={traj.final_prediction})", fontsize=9.5)
        ax.set_yticks([0, 1, 2, 3, 4])
        ax.set_yticklabels(["None", "A", "B", "C", "D"])
        ax.set_xlim(0, 256)
        ax.legend(fontsize=7.5, loc="lower right")
        ax.grid(alpha=0.25)

    for ax in axes[1]:
        ax.set_xlabel("Denoising step")
    fig.suptitle("Figure 2: Per-step verdict curves (LLaDA-8B-Instruct, ARC-C reasoning)", fontsize=13, y=1.01)
    fig.tight_layout()
    out = OUT_DIR / "fig2_verdict_curves.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def build_fig3(summary_path: Path) -> Path | None:
    if not summary_path.exists():
        return None
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    patch_conds = [c for c in data["conditions"] if c["name"].startswith("patch_k")]
    noise_conds = [c for c in data["conditions"] if c["name"].startswith("noise_k")]
    if not patch_conds:
        return None

    patch_conds.sort(key=lambda c: int(c["name"].replace("patch_k", "")))
    ks = [int(c["name"].replace("patch_k", "")) for c in patch_conds]
    flips = [c["verdict_flip_rate"] for c in patch_conds]
    flip_ses = [c["verdict_flip_se"] for c in patch_conds]
    deltas = [c["mean_commit_step_delta"] for c in patch_conds]
    delta_ses = [c["se_commit_step_delta"] for c in patch_conds]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    axes[0].errorbar(ks, flips, yerr=flip_ses, marker="o", capsize=4, color="#c53030", label="Partner hidden patch")
    if noise_conds:
        noise_conds.sort(key=lambda c: int(c["name"].replace("noise_k", "")))
        n_ks = [int(c["name"].replace("noise_k", "")) for c in noise_conds]
        n_flips = [c["verdict_flip_rate"] for c in noise_conds]
        n_ses = [c["verdict_flip_se"] for c in noise_conds]
        axes[0].errorbar(n_ks, n_flips, yerr=n_ses, marker="s", linestyle="--", capsize=4, color="#718096", label="Matched-norm noise")
    axes[0].set_xlabel("Patch step k")
    axes[0].set_ylabel("Verdict flip rate vs. baseline")
    axes[0].set_title("Verdict flip rate by patch step k")
    axes[0].legend(frameon=True, fontsize=9)
    axes[0].grid(alpha=0.25)

    axes[1].errorbar(ks, deltas, yerr=delta_ses, marker="o", capsize=4, color="#2b6cb0", label="Partner hidden patch")
    if noise_conds:
        n_deltas = [c["mean_commit_step_delta"] for c in noise_conds]
        n_d_ses = [c["se_commit_step_delta"] for c in noise_conds]
        axes[1].errorbar(n_ks, n_deltas, yerr=n_d_ses, marker="s", linestyle="--", capsize=4, color="#718096", label="Matched-norm noise")
    axes[1].axhline(0.0, color="#4a5568", linestyle=":", linewidth=1.0)
    axes[1].set_xlabel("Patch step k")
    axes[1].set_ylabel("Mean commitment-step delta (patched - base)")
    axes[1].set_title("Commitment-step shift by patch step k")
    axes[1].legend(frameon=True, fontsize=9)
    axes[1].grid(alpha=0.25)

    fig.suptitle("Figure 3: Causal effect of layer-16 hidden-state patching (LLaDA-8B-Instruct, n=64)", fontsize=12.5, y=1.02)
    fig.tight_layout()
    out = OUT_DIR / "fig3_patching_effect.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    f1 = build_fig1()
    f2 = build_fig2()
    f3 = build_fig3(Path("results/p1_patching/p1_llada_patching_summary.json"))
    print("Generated:", f1, f2, f3)


if __name__ == "__main__":
    main()
