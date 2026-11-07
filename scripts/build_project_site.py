"""Build static project site data bundle and Hugging Face Dataset/Space cards.

Compiles `results/*_results.json`, `results/*_direct_vs_reasoning.json`, and
step-by-step trajectory keyframes from `results/*.jsonl` into:
- `site/data/trajectories_bundle.json` (used by `site/index.html` & HF Static Space)
- `hf/dataset_README.md` (Hugging Face Dataset card for all benchmark artifacts)
- `hf/space_README.md` (Hugging Face Static Space card)
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
SITE_DIR = REPO_ROOT / "site"
SITE_DATA_DIR = SITE_DIR / "data"
HF_DIR = REPO_ROOT / "hf"


SUITE_SPECS: list[dict[str, Any]] = [
    {
        "suite_id": "dream_arcc_smoke64_direct",
        "model_family": "dream",
        "model_label": "Dream-v0-Instruct-7B",
        "hf_model_id": "Dream-org/Dream-v0-Instruct-7B",
        "task": "arcc",
        "task_label": "ARC-Challenge",
        "condition": "direct",
        "condition_label": "Direct (Answer-First)",
        "scale": 64,
        "input_file": "arcc_smoke64_direct_input.json",
    },
    {
        "suite_id": "dream_arcc_smoke64_reasoning",
        "model_family": "dream",
        "model_label": "Dream-v0-Instruct-7B",
        "hf_model_id": "Dream-org/Dream-v0-Instruct-7B",
        "task": "arcc",
        "task_label": "ARC-Challenge",
        "condition": "reasoning",
        "condition_label": "Canonical Reasoning (Rationale-First)",
        "scale": 64,
        "input_file": "arcc_smoke64_reasoning_input.json",
    },
    {
        "suite_id": "llada_instruct_arcc_smoke64_direct",
        "model_family": "llada-instruct",
        "model_label": "LLaDA-8B-Instruct",
        "hf_model_id": "GSAI-ML/LLaDA-8B-Instruct",
        "task": "arcc",
        "task_label": "ARC-Challenge",
        "condition": "direct",
        "condition_label": "Direct (Answer-First)",
        "scale": 64,
        "input_file": "arcc_smoke64_direct_input.json",
    },
    {
        "suite_id": "llada_instruct_arcc_smoke64_reasoning",
        "model_family": "llada-instruct",
        "model_label": "LLaDA-8B-Instruct",
        "hf_model_id": "GSAI-ML/LLaDA-8B-Instruct",
        "task": "arcc",
        "task_label": "ARC-Challenge",
        "condition": "reasoning",
        "condition_label": "Canonical Reasoning (Rationale-First)",
        "scale": 64,
        "input_file": "arcc_smoke64_reasoning_input.json",
    },
    {
        "suite_id": "llada_15_arcc_smoke64_direct",
        "model_family": "llada-1.5",
        "model_label": "LLaDA-1.5 (VRPO)",
        "hf_model_id": "GSAI-ML/LLaDA-1.5",
        "task": "arcc",
        "task_label": "ARC-Challenge",
        "condition": "direct",
        "condition_label": "Direct (Answer-First)",
        "scale": 64,
        "input_file": "arcc_smoke64_direct_input.json",
    },
    {
        "suite_id": "llada_15_arcc_smoke64_reasoning",
        "model_family": "llada-1.5",
        "model_label": "LLaDA-1.5 (VRPO)",
        "hf_model_id": "GSAI-ML/LLaDA-1.5",
        "task": "arcc",
        "task_label": "ARC-Challenge",
        "condition": "reasoning",
        "condition_label": "Canonical Reasoning (Rationale-First)",
        "scale": 64,
        "input_file": "arcc_smoke64_reasoning_input.json",
    },
    {
        "suite_id": "dream_gsm8k_smoke64",
        "model_family": "dream",
        "model_label": "Dream-v0-Instruct-7B",
        "hf_model_id": "Dream-org/Dream-v0-Instruct-7B",
        "task": "gsm8k",
        "task_label": "GSM8K",
        "condition": "cot_numeric",
        "condition_label": "0-Shot Step-by-Step Numeric",
        "scale": 64,
        "input_file": "gsm8k_smoke64_input.json",
    },
    {
        "suite_id": "llada_instruct_gsm8k_smoke64",
        "model_family": "llada-instruct",
        "model_label": "LLaDA-8B-Instruct",
        "hf_model_id": "GSAI-ML/LLaDA-8B-Instruct",
        "task": "gsm8k",
        "task_label": "GSM8K",
        "condition": "cot_numeric",
        "condition_label": "0-Shot Step-by-Step Numeric",
        "scale": 64,
        "input_file": "gsm8k_smoke64_input.json",
    },
    {
        "suite_id": "llada_15_gsm8k_smoke64",
        "model_family": "llada-1.5",
        "model_label": "LLaDA-1.5 (VRPO)",
        "hf_model_id": "GSAI-ML/LLaDA-1.5",
        "task": "gsm8k",
        "task_label": "GSM8K",
        "condition": "cot_numeric",
        "condition_label": "0-Shot Step-by-Step Numeric",
        "scale": 64,
        "input_file": "gsm8k_smoke64_input.json",
    },
    {
        "suite_id": "dream_arcc_smoke5",
        "model_family": "dream",
        "model_label": "Dream-v0-Instruct-7B",
        "hf_model_id": "Dream-org/Dream-v0-Instruct-7B",
        "task": "arcc",
        "task_label": "ARC-Challenge",
        "condition": "direct",
        "condition_label": "Direct (5-item Smoke)",
        "scale": 5,
        "input_file": "dream_arcc_smoke5_input.json",
    },
    {
        "suite_id": "dream_arcc_smoke5_reasoning",
        "model_family": "dream",
        "model_label": "Dream-v0-Instruct-7B",
        "hf_model_id": "Dream-org/Dream-v0-Instruct-7B",
        "task": "arcc",
        "task_label": "ARC-Challenge",
        "condition": "reasoning",
        "condition_label": "Canonical Reasoning (5-item Smoke)",
        "scale": 5,
        "input_file": "dream_arcc_smoke5_reasoning_input.json",
    },
    {
        "suite_id": "dream_arcc_eos_suppression_k64_smoke5",
        "model_family": "dream",
        "model_label": "Dream-v0-Instruct-7B",
        "hf_model_id": "Dream-org/Dream-v0-Instruct-7B",
        "task": "arcc",
        "task_label": "ARC-Challenge",
        "condition": "eos_suppression_k64",
        "condition_label": "EOS Suppression (k=64, 5-item Intervention)",
        "scale": 5,
        "input_file": "dream_arcc_smoke5_input.json",
    },
    {
        "suite_id": "llada_instruct_arcc_smoke5_direct",
        "model_family": "llada-instruct",
        "model_label": "LLaDA-8B-Instruct",
        "hf_model_id": "GSAI-ML/LLaDA-8B-Instruct",
        "task": "arcc",
        "task_label": "ARC-Challenge",
        "condition": "direct",
        "condition_label": "Direct (5-item Smoke)",
        "scale": 5,
        "input_file": "dream_arcc_smoke5_input.json",
    },
    {
        "suite_id": "llada_instruct_arcc_smoke5_reasoning",
        "model_family": "llada-instruct",
        "model_label": "LLaDA-8B-Instruct",
        "hf_model_id": "GSAI-ML/LLaDA-8B-Instruct",
        "task": "arcc",
        "task_label": "ARC-Challenge",
        "condition": "reasoning",
        "condition_label": "Canonical Reasoning (5-item Smoke)",
        "scale": 5,
        "input_file": "dream_arcc_smoke5_reasoning_input.json",
    },
    {
        "suite_id": "llada_15_arcc_smoke5_direct",
        "model_family": "llada-1.5",
        "model_label": "LLaDA-1.5 (VRPO)",
        "hf_model_id": "GSAI-ML/LLaDA-1.5",
        "task": "arcc",
        "task_label": "ARC-Challenge",
        "condition": "direct",
        "condition_label": "Direct (5-item Smoke)",
        "scale": 5,
        "input_file": "dream_arcc_smoke5_input.json",
    },
    {
        "suite_id": "llada_15_arcc_smoke5_reasoning",
        "model_family": "llada-1.5",
        "model_label": "LLaDA-1.5 (VRPO)",
        "hf_model_id": "GSAI-ML/LLaDA-1.5",
        "task": "arcc",
        "task_label": "ARC-Challenge",
        "condition": "reasoning",
        "condition_label": "Canonical Reasoning (5-item Smoke)",
        "scale": 5,
        "input_file": "dream_arcc_smoke5_reasoning_input.json",
    },
    {
        "suite_id": "dream_gsm8k_smoke5",
        "model_family": "dream",
        "model_label": "Dream-v0-Instruct-7B",
        "hf_model_id": "Dream-org/Dream-v0-Instruct-7B",
        "task": "gsm8k",
        "task_label": "GSM8K",
        "condition": "cot_numeric",
        "condition_label": "0-Shot Step-by-Step Numeric (5-item Smoke)",
        "scale": 5,
        "input_file": "dream_gsm8k_smoke5_input.json",
    },
    {
        "suite_id": "llada_instruct_gsm8k_smoke5",
        "model_family": "llada-instruct",
        "model_label": "LLaDA-8B-Instruct",
        "hf_model_id": "GSAI-ML/LLaDA-8B-Instruct",
        "task": "gsm8k",
        "task_label": "GSM8K",
        "condition": "cot_numeric",
        "condition_label": "0-Shot Step-by-Step Numeric (5-item Smoke)",
        "scale": 5,
        "input_file": "dream_gsm8k_smoke5_input.json",
    },
    {
        "suite_id": "llada_15_gsm8k_smoke5",
        "model_family": "llada-1.5",
        "model_label": "LLaDA-1.5 (VRPO)",
        "hf_model_id": "GSAI-ML/LLaDA-1.5",
        "task": "gsm8k",
        "task_label": "GSM8K",
        "condition": "cot_numeric",
        "condition_label": "0-Shot Step-by-Step Numeric (5-item Smoke)",
        "scale": 5,
        "input_file": "dream_gsm8k_smoke5_input.json",
    },
]

COMPARISON_SPECS: list[dict[str, Any]] = [
    {
        "comparison_id": "dream_arcc_smoke64_direct_vs_reasoning",
        "model_label": "Dream-v0-Instruct-7B",
        "task_label": "ARC-Challenge (64 items)",
        "scale": 64,
        "file": "dream_arcc_smoke64_direct_vs_reasoning.json",
    },
    {
        "comparison_id": "llada_instruct_arcc_smoke64_direct_vs_reasoning",
        "model_label": "LLaDA-8B-Instruct",
        "task_label": "ARC-Challenge (64 items)",
        "scale": 64,
        "file": "llada_instruct_arcc_smoke64_direct_vs_reasoning.json",
    },
    {
        "comparison_id": "llada_15_arcc_smoke64_direct_vs_reasoning",
        "model_label": "LLaDA-1.5 (VRPO)",
        "task_label": "ARC-Challenge (64 items)",
        "scale": 64,
        "file": "llada_15_arcc_smoke64_direct_vs_reasoning.json",
    },
    {
        "comparison_id": "dream_arcc_smoke5_base_vs_eos_suppression_k64",
        "model_label": "Dream-v0-Instruct-7B",
        "task_label": "ARC-Challenge EOS Suppression Intervention (5 items)",
        "scale": 5,
        "file": "dream_arcc_smoke5_base_vs_eos_suppression_k64.json",
    },
]


_MASK_RUN_RE = re.compile(r"(?:<\|(?:mdm_)?mask\|>){2,}")
_SINGLE_MASK_RE = re.compile(r"<\|(?:mdm_)?mask\|>")


def pack_mask_runs(text: str) -> str:
    """Pack consecutive `<|mask|>` / `<|mdm_mask|>` tokens into `<<M:count>>`."""
    if not text:
        return ""

    def _repl(match: re.Match[str]) -> str:
        s = match.group(0)
        count = len(_SINGLE_MASK_RE.findall(s))
        return f"<<M:{count}>>"

    packed = _MASK_RUN_RE.sub(_repl, text)
    return _SINGLE_MASK_RE.sub("<<M:1>>", packed)


def unpack_mask_runs(packed: str, mask_token: str = "<|mask|>") -> str:
    """Unpack `<<M:count>>` back into repeated mask tokens."""
    return re.sub(
        r"<<M:(\d+)>>",
        lambda m: mask_token * int(m.group(1)),
        packed,
    )


def compress_steps_lossless(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compress trajectory steps losslessly by keeping state-change keyframes
    and run-length packing repeated mask tokens (`<<M:k>>`).

    Any step `t` in `[1..n_steps]` has the exact state of the latest keyframe
    with `step <= t`.
    """
    keyframes: list[dict[str, Any]] = []
    prev_sig: tuple[Any, ...] | None = None
    n = len(steps)
    for idx, s in enumerate(steps):
        crf = s.get("committed_reasoning_fraction")
        crf_round = round(float(crf), 4) if crf is not None else None
        conf = s.get("confidence")
        conf_round = round(float(conf), 4) if conf is not None else None
        packed_text = pack_mask_runs(s.get("reasoning_text", ""))
        sig = (
            s.get("prediction"),
            packed_text,
            s.get("answer_committed"),
            crf_round,
        )
        if prev_sig is None or sig != prev_sig or idx == n - 1:
            keyframes.append(
                {
                    "step": int(s["step"]),
                    "prediction": s.get("prediction"),
                    "reasoning_text": packed_text,
                    "confidence": conf_round,
                    "answer_committed": s.get("answer_committed"),
                    "committed_reasoning_fraction": crf_round,
                }
            )
            prev_sig = sig
    return keyframes


def _load_prompt_map(input_filename: str, results_dir: Path) -> dict[str, str]:
    path = results_dir / input_filename
    if not path.exists():
        return {}
    items = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for item in items:
        item_id = str(item.get("id", ""))
        prompt = str(item.get("question") or item.get("prompt") or "")
        if item_id:
            out[item_id] = prompt
    return out


def select_showcase_ids(per_trajectory: list[dict[str, Any]], scale: int) -> set[str]:
    """Select showcase trajectory IDs for a suite so users can scrub rich examples."""
    if scale <= 5:
        return {str(r["id"]) for r in per_trajectory}

    selected: list[str] = []
    for r in per_trajectory[:5]:
        selected.append(str(r["id"]))

    by_flips = sorted(
        per_trajectory,
        key=lambda r: (float(r.get("flip_rate") or 0.0), float(r.get("gated_rationalization_score") or 0.0)),
        reverse=True,
    )
    for r in by_flips[:2]:
        selected.append(str(r["id"]))

    for special_id in ("Mercury_402216", "Mercury_412681"):
        if any(str(r["id"]) == special_id for r in per_trajectory):
            selected.append(special_id)

    return set(selected)


def build_bundle(results_dir: Path = RESULTS_DIR) -> dict[str, Any]:
    """Compile all suites, comparisons, and lossless trajectory keyframes into a bundle."""
    suites_out: list[dict[str, Any]] = []
    trajectories_out: dict[str, list[dict[str, Any]]] = {}

    for spec in SUITE_SPECS:
        suite_id = spec["suite_id"]
        res_path = results_dir / f"{suite_id}_results.json"
        log_path = results_dir / f"{suite_id}_run_log.json"
        jsonl_path = results_dir / f"{suite_id}.jsonl"
        if not res_path.exists():
            continue

        res_data = json.loads(res_path.read_text(encoding="utf-8"))
        log_data = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else {}
        prompts_by_id = _load_prompt_map(spec["input_file"], results_dir)

        per_traj = res_data.get("per_trajectory", [])
        metrics_by_id = {str(r["id"]): r for r in per_traj}
        showcase_ids = select_showcase_ids(per_traj, int(spec["scale"]))

        suite_trajs: list[dict[str, Any]] = []
        if jsonl_path.exists():
            with jsonl_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    tid = str(row["id"])
                    if tid not in showcase_ids:
                        continue
                    steps = row.get("steps", [])
                    suite_trajs.append(
                        {
                            "id": tid,
                            "prompt": prompts_by_id.get(tid, ""),
                            "gold_answer": row.get("gold_answer"),
                            "final_prediction": row.get("final_prediction"),
                            "n_steps": len(steps),
                            "metrics": metrics_by_id.get(tid, {}),
                            "keyframes": compress_steps_lossless(steps),
                        }
                    )

        trajectories_out[suite_id] = suite_trajs
        suites_out.append(
            {
                **spec,
                "git_revision": log_data.get("git_revision", ""),
                "gpu_device": log_data.get("env", {}).get("cuda_visible_devices", ""),
                "sampler_config": log_data.get("sampler_config", {}),
                "aggregate": res_data.get("aggregate", {}),
                "per_trajectory": per_traj,
                "showcase_ids": [t["id"] for t in suite_trajs],
            }
        )

    comparisons_out: list[dict[str, Any]] = []
    for cspec in COMPARISON_SPECS:
        cpath = results_dir / cspec["file"]
        if not cpath.exists():
            continue
        cdata = json.loads(cpath.read_text(encoding="utf-8"))
        comparisons_out.append({**cspec, "data": cdata})

    return {
        "project": {
            "title": "Same Accuracy, Different Trajectory: Measuring Reasoning Regimes in Masked Diffusion Language Models",
            "short_title": "traj-eval",
            "author": "Adele Chinda",
            "affiliation": "Department of Computer Science, Georgia State University",
            "github_url": "https://github.com/arrdel/traj-eval",
            "version": "0.1.0",
        },
        "suites": suites_out,
        "comparisons": comparisons_out,
        "trajectories": trajectories_out,
    }


def render_hf_dataset_card(bundle: dict[str, Any]) -> str:
    """Render the Hugging Face Dataset Card (`README.md`) with YAML frontmatter."""
    suites_64 = [s for s in bundle["suites"] if s["scale"] == 64]
    rows_md: list[str] = []
    for s in suites_64:
        agg = s["aggregate"]
        rows_md.append(
            f"| `{s['suite_id']}` | `{s['hf_model_id']}` | {s['task_label']} | "
            f"{s['condition_label']} | {agg.get('accuracy', 0.0):.4f} | "
            f"{agg.get('coverage', 0.0):.4f} | {agg.get('mean_convergence_step', 0.0):.1f} | "
            f"{agg.get('mean_answer_commitment_step', 0.0):.1f} | "
            f"{agg.get('mean_unwritten_reasoning_at_convergence', 0.0):.4f} | "
            f"{agg.get('mean_order_gap', 0.0):+.4f} | "
            f"{agg.get('mean_gated_rationalization_score', 0.0):.4f} |"
        )
    table_str = "\n".join(rows_md)

    return f"""---
license: mit
task_categories:
  - question-answering
  - text-generation
language:
  - en
tags:
  - masked-diffusion
  - diffusion-language-models
  - reasoning-trajectories
  - evaluation
  - dream-7b
  - llada
  - arc-challenge
  - gsm8k
pretty_name: "traj-eval: Masked Diffusion Reasoning Trajectory Benchmarks"
size_categories:
  - n<1K
---

# `traj-eval`: Step-by-Step Denoising Trajectory Benchmarks for Masked Diffusion LMs

**Paper:** *Same accuracy, different trajectory: measuring reasoning regimes in masked diffusion language models*  
**Author:** Adele Chinda (Georgia State University)  
**Code & Evaluation Toolkit:** [https://github.com/arrdel/traj-eval](https://github.com/arrdel/traj-eval)

## Overview

Final-step accuracy treats a model that commits at step 3 and backfills a post-hoc explanation identically to a model that deliberates until step 220 before locking in its answer. **`traj-eval`** records and scores the full step-by-step denoising trajectory $t = 1 \\dots T$ of open masked diffusion language models (`Dream-org/Dream-v0-Instruct-7B`, `GSAI-ML/LLaDA-8B-Instruct`, and `GSAI-ML/LLaDA-1.5`) across **ARC-Challenge** (`direct` vs. canonical `reasoning` prompts, plus `EOS suppression` interventions) and **GSM8K**.

Each `.jsonl` file in this repository contains complete step-by-step denoising trajectories with:
- `step`: 1-indexed denoising step $t \\in [1, T]$
- `prediction`: extracted verdict at step $t$ (`"A"`–`"D"` on ARC-C, normalized number on GSM8K, or `null` while still masked/undecided)
- `reasoning_text`: decoded intermediate canvas at step $t$ (preserving `<|mask|>` / `<|mdm_mask|>` tokens)
- `answer_committed`: boolean indicating whether the answer token span in the canvas is unmasked at step $t$
- `committed_reasoning_fraction`: fraction $F_R(t) \\in [0, 1]$ of non-answer rationale tokens unmasked at step $t$

## 64-Item Benchmark Summary

| Suite ID | Checkpoint | Task | Condition | Accuracy | Coverage | Mean $T_{{\\text{{stab}}}}$ | Mean $T_{{\\text{{commit}}}}$ | Unwritten $U$ | Order Gap | Gated Rat. Score |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
{table_str}

## Quickstart

```python
from huggingface_hub import hf_hub_download
from traj_eval.io import load_trajectories
from traj_eval.metrics import evaluate_trajectories

jsonl_path = hf_hub_download(
    repo_id="arrdel/traj-eval-benchmarks",
    filename="results/llada_15_arcc_smoke64_reasoning.jsonl",
    repo_type="dataset",
)
trajs = load_trajectories(jsonl_path)
summary = evaluate_trajectories(trajs)
print(summary["aggregate"])
```

## License

MIT License.
"""


def render_hf_space_card() -> str:
    """Render the Hugging Face Static Space Card (`README.md`)."""
    return """---
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

- **Scrub step-by-step ($t = 1 \\dots 256$)** through real denoising trajectories from `Dream-org/Dream-v0-Instruct-7B`, `GSAI-ML/LLaDA-8B-Instruct`, and `GSAI-ML/LLaDA-1.5`.
- **Compare reasoning clocks:** Verdict convergence ($T_{\\text{stab}}$), token commitment ($T_{\\text{commit}}$), unwritten reasoning fraction ($U = 1 - F_R(T_{\\text{stab}})$), order gap, and multiplicative gated rationalization score.
- **Inspect your own `.jsonl` files:** Drag and drop any `traj-eval` `.jsonl` file directly in the browser to visualize and audit custom runs client-side.
"""


def write_all_artifacts(
    results_dir: Path = RESULTS_DIR,
    site_data_dir: Path = SITE_DATA_DIR,
    hf_dir: Path = HF_DIR,
) -> dict[str, Path]:
    """Build and write `summary.json`, per-suite keyframe files, and HF cards."""
    bundle = build_bundle(results_dir=results_dir)
    site_data_dir.mkdir(parents=True, exist_ok=True)
    suites_dir = site_data_dir / "suites"
    suites_dir.mkdir(parents=True, exist_ok=True)
    hf_dir.mkdir(parents=True, exist_ok=True)

    summary_payload = {
        "project": bundle["project"],
        "suites": bundle["suites"],
        "comparisons": bundle["comparisons"],
    }
    summary_path = site_data_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary_payload, separators=(",", ":")),
        encoding="utf-8",
    )

    for suite_id, trajs in bundle["trajectories"].items():
        suite_path = suites_dir / f"{suite_id}.json"
        suite_path.write_text(
            json.dumps({"suite_id": suite_id, "trajectories": trajs}, separators=(",", ":")),
            encoding="utf-8",
        )

    bundle_path = site_data_dir / "trajectories_bundle.json"
    if bundle_path.exists():
        bundle_path.unlink()

    dataset_card_path = hf_dir / "dataset_README.md"
    dataset_card_path.write_text(render_hf_dataset_card(bundle), encoding="utf-8")

    space_card_path = hf_dir / "space_README.md"
    space_card_path.write_text(render_hf_space_card(), encoding="utf-8")

    return {
        "summary_path": summary_path,
        "suites_dir": suites_dir,
        "dataset_card_path": dataset_card_path,
        "space_card_path": space_card_path,
    }


if __name__ == "__main__":
    paths = write_all_artifacts()
    for label, p in paths.items():
        if p.is_file():
            size_kb = p.stat().st_size / 1024.0
            print(f"{label}: {p} ({size_kb:.1f} KB)")
        else:
            files = list(p.glob("*.json"))
            total_kb = sum(f.stat().st_size for f in files) / 1024.0
            print(f"{label}: {p} ({len(files)} files, {total_kb:.1f} KB total)")
