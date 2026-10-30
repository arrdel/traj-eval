#!/usr/bin/env python3
# Copyright (C) 2026 Lenovo
# All Rights Reserved.
# Lenovo Confidential Restricted.

"""Sequential GPU watcher and PR-gated auto-queue runner for traj-eval.

Why a sequential watcher instead of a raw cron job
--------------------------------------------------
A standard ``cron`` job triggers on wall-clock time (e.g. every 5 minutes)
with no awareness of whether a previous GPU job is still running or whether
a protocol gate passed. This module implements a stateful, lockfile-guarded
sequential job queue that:

1. Optionally waits until a scheduled reservation start time (``--not-before``)
   and inspects active bookings on ``amai-gsu/amai-gpu-scheduler``.
2. Polls ``nvidia-smi`` until at least one GPU satisfies both free-VRAM
   (``>= 25,000 MiB``) and utilization (``<= 25%``) thresholds across two
   consecutive checks.
3. Runs the next pending job in ``cdr-gpu`` pinned to that single GPU via
   ``CUDA_VISIBLE_DEVICES``.
4. Scores the resulting JSONL trajectory file with ``traj_eval``, writes the
   scored JSON + optional side-by-side condition comparison JSON, and appends
   a complete provenance & results section to ``results/baselines.md``.
5. Creates a clean feature branch directly off ``origin/master``
   (``feat/smoke-<job-id>``), commits the explicit artifact files, pushes the
   branch, opens a Pull Request against ``master`` using
   ``.github/pull_request_template.md``, and waits for the managing agent
   (Muse) to review and merge that PR before advancing to the next stage.
6. Enforces the hard protocol gate in ``docs/CURSOR_FOLLOWUP_2026-10-30.md``
   §5: if the 5-item reasoning smoke fails ``check_reasoning_smoke_gate``,
   it opens the diagnostic PR and halts the queue instead of scaling to
   64-item or multi-checkpoint runs.
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from traj_eval.cli import _print_summary  # noqa: E402
from traj_eval.io import load_trajectories, save_results  # noqa: E402
from traj_eval.metrics import (  # noqa: E402
    check_reasoning_smoke_gate,
    compare_conditions,
    evaluate_trajectories,
)

DEFAULT_CDR_GPU_PYTHON = "/home/achinda1/miniconda3/envs/cdr-gpu/bin/python"
LOCK_FILE = REPO_ROOT / ".gpu_queue.lock"
STATE_FILE = REPO_ROOT / "results" / "gpu_queue_state.json"
SCHEDULER_REPO = "amai-gsu/amai-gpu-scheduler"


@dataclass
class GPUStatus:
    """Parsed snapshot of a single GPU from ``nvidia-smi``."""

    index: int
    name: str
    memory_used_mib: int
    memory_free_mib: int
    memory_total_mib: int
    utilization_pct: int


@dataclass
class QueuedJob:
    """One sequential experiment in the GPU queue."""

    job_id: str
    title: str
    model_family: str
    task: str
    items_path: str
    out_jsonl: str
    out_ids: str
    out_log: str
    out_results: str
    temperature: float = 0.2
    top_p: float = 0.95
    steps: int = 256
    max_new_tokens: int = 256
    block_length: int = 256
    force_deliberation_steps: int = 0
    frontier_width: int | None = None
    require_reasoning_gate: bool = False
    compare_base_jsonl: str | None = None
    compare_base_label: str = "direct"
    compare_target_label: str = "reasoning"
    compare_out_json: str | None = None

    @property
    def branch_name(self) -> str:
        """Dedicated PR branch name for this job."""
        return f"feat/smoke-{self.job_id.replace('_', '-')}"


DEFAULT_JOB_QUEUE: list[QueuedJob] = [
    # Stage 1: Mandatory next experiment per docs/CURSOR_FOLLOWUP_2026-10-30.md §5.
    # Halts the entire queue if check_reasoning_smoke_gate fails.
    QueuedJob(
        job_id="dream_arcc_smoke5_reasoning",
        title="Dream-v0-Instruct-7B — ARC-C 5-item reasoning-condition smoke",
        model_family="dream",
        task="arcc",
        items_path="results/dream_arcc_smoke5_reasoning_input.json",
        out_jsonl="results/dream_arcc_smoke5_reasoning.jsonl",
        out_ids="results/dream_arcc_smoke5_reasoning_ids.json",
        out_log="results/dream_arcc_smoke5_reasoning_run_log.json",
        out_results="results/dream_arcc_smoke5_reasoning_results.json",
        require_reasoning_gate=True,
        compare_base_jsonl="results/dream_arcc_smoke5.jsonl",
        compare_base_label="direct",
        compare_target_label="reasoning",
        compare_out_json="results/dream_arcc_smoke5_direct_vs_reasoning.json",
    ),
    # Stage 2: Paired 64-item ARC-C direct + canonical reasoning runs on Dream-7B.
    QueuedJob(
        job_id="dream_arcc_smoke64_direct",
        title="Dream-v0-Instruct-7B — ARC-C 64-item direct-answer suite",
        model_family="dream",
        task="arcc",
        items_path="results/arcc_smoke64_direct_input.json",
        out_jsonl="results/dream_arcc_smoke64_direct.jsonl",
        out_ids="results/arcc_smoke64_ids.json",
        out_log="results/dream_arcc_smoke64_direct_run_log.json",
        out_results="results/dream_arcc_smoke64_direct_results.json",
    ),
    QueuedJob(
        job_id="dream_arcc_smoke64_reasoning",
        title="Dream-v0-Instruct-7B — ARC-C 64-item canonical reasoning suite",
        model_family="dream",
        task="arcc",
        items_path="results/arcc_smoke64_reasoning_input.json",
        out_jsonl="results/dream_arcc_smoke64_reasoning.jsonl",
        out_ids="results/arcc_smoke64_ids.json",
        out_log="results/dream_arcc_smoke64_reasoning_run_log.json",
        out_results="results/dream_arcc_smoke64_reasoning_results.json",
        compare_base_jsonl="results/dream_arcc_smoke64_direct.jsonl",
        compare_base_label="direct_64",
        compare_target_label="reasoning_64",
        compare_out_json="results/dream_arcc_smoke64_direct_vs_reasoning.json",
    ),
    # Stage 3: GSM8K 5-item numeric smoke on Dream-7B.
    QueuedJob(
        job_id="dream_gsm8k_smoke5",
        title="Dream-v0-Instruct-7B — GSM8K 5-item numeric smoke",
        model_family="dream",
        task="gsm8k",
        items_path="results/dream_gsm8k_smoke5_input.json",
        out_jsonl="results/dream_gsm8k_smoke5.jsonl",
        out_ids="results/dream_gsm8k_smoke5_ids.json",
        out_log="results/dream_gsm8k_smoke5_run_log.json",
        out_results="results/dream_gsm8k_smoke5_results.json",
    ),
    # Stage 4: Pre-registered EOS-suppression K=64 5-item smoke (named per CURSOR_FOLLOWUP_2026-10-30 §3c).
    QueuedJob(
        job_id="dream_arcc_eos_suppression_k64_smoke5",
        title="Dream-v0-Instruct-7B — ARC-C 5-item EOS-suppression K=64 smoke",
        model_family="dream",
        task="arcc",
        items_path="results/dream_arcc_smoke5_input.json",
        out_jsonl="results/dream_arcc_eos_suppression_k64_smoke5.jsonl",
        out_ids="results/dream_arcc_smoke5_ids.json",
        out_log="results/dream_arcc_eos_suppression_k64_smoke5_run_log.json",
        out_results="results/dream_arcc_eos_suppression_k64_smoke5_results.json",
        force_deliberation_steps=64,
        compare_base_jsonl="results/dream_arcc_smoke5.jsonl",
        compare_base_label="unrestricted",
        compare_target_label="eos_suppression_k64",
        compare_out_json="results/dream_arcc_smoke5_base_vs_eos_suppression_k64.json",
    ),
    # Stage 5: LLaDA-8B-Instruct and LLaDA-1.5 paired 5-item ARC-C smokes.
    QueuedJob(
        job_id="llada_instruct_arcc_smoke5_direct",
        title="LLaDA-8B-Instruct — ARC-C 5-item direct-answer smoke",
        model_family="llada-instruct",
        task="arcc",
        items_path="results/dream_arcc_smoke5_input.json",
        out_jsonl="results/llada_instruct_arcc_smoke5_direct.jsonl",
        out_ids="results/dream_arcc_smoke5_ids.json",
        out_log="results/llada_instruct_arcc_smoke5_direct_run_log.json",
        out_results="results/llada_instruct_arcc_smoke5_direct_results.json",
        temperature=0.0,
    ),
    QueuedJob(
        job_id="llada_instruct_arcc_smoke5_reasoning",
        title="LLaDA-8B-Instruct — ARC-C 5-item reasoning-condition smoke",
        model_family="llada-instruct",
        task="arcc",
        items_path="results/dream_arcc_smoke5_reasoning_input.json",
        out_jsonl="results/llada_instruct_arcc_smoke5_reasoning.jsonl",
        out_ids="results/dream_arcc_smoke5_reasoning_ids.json",
        out_log="results/llada_instruct_arcc_smoke5_reasoning_run_log.json",
        out_results="results/llada_instruct_arcc_smoke5_reasoning_results.json",
        temperature=0.0,
        compare_base_jsonl="results/llada_instruct_arcc_smoke5_direct.jsonl",
        compare_base_label="direct",
        compare_target_label="reasoning",
        compare_out_json="results/llada_instruct_arcc_smoke5_direct_vs_reasoning.json",
    ),
    QueuedJob(
        job_id="llada_15_arcc_smoke5_direct",
        title="LLaDA-1.5 — ARC-C 5-item direct-answer smoke",
        model_family="llada-1.5",
        task="arcc",
        items_path="results/dream_arcc_smoke5_input.json",
        out_jsonl="results/llada_15_arcc_smoke5_direct.jsonl",
        out_ids="results/dream_arcc_smoke5_ids.json",
        out_log="results/llada_15_arcc_smoke5_direct_run_log.json",
        out_results="results/llada_15_arcc_smoke5_direct_results.json",
        temperature=0.0,
    ),
    QueuedJob(
        job_id="llada_15_arcc_smoke5_reasoning",
        title="LLaDA-1.5 — ARC-C 5-item reasoning-condition smoke",
        model_family="llada-1.5",
        task="arcc",
        items_path="results/dream_arcc_smoke5_reasoning_input.json",
        out_jsonl="results/llada_15_arcc_smoke5_reasoning.jsonl",
        out_ids="results/dream_arcc_smoke5_reasoning_ids.json",
        out_log="results/llada_15_arcc_smoke5_reasoning_run_log.json",
        out_results="results/llada_15_arcc_smoke5_reasoning_results.json",
        temperature=0.0,
        compare_base_jsonl="results/llada_15_arcc_smoke5_direct.jsonl",
        compare_base_label="direct",
        compare_target_label="reasoning",
        compare_out_json="results/llada_15_arcc_smoke5_direct_vs_reasoning.json",
    ),
]


def parse_scheduler_bookings(
    bookings_payload: dict[str, Any], uid: str = "arrdel"
) -> list[dict[str, Any]]:
    """Filter active (non-cancelled, non-closed) bookings for ``uid`` from ``bookings.json``."""
    raw_list = bookings_payload.get("bookings", [])
    if not isinstance(raw_list, list):
        return []
    active: list[dict[str, Any]] = []
    for item in raw_list:
        if not isinstance(item, dict):
            continue
        if item.get("uid") != uid:
            continue
        if item.get("cancelled") or item.get("closedAt"):
            continue
        active.append(item)
    active.sort(key=lambda b: (str(b.get("start", "")), str(b.get("gpuId", ""))))
    return active


def fetch_scheduler_bookings(uid: str = "arrdel") -> list[dict[str, Any]]:
    """Fetch active bookings for ``uid`` from ``amai-gsu/amai-gpu-scheduler`` via ``gh api``."""
    try:
        raw = subprocess.check_output(
            ["gh", "api", f"repos/{SCHEDULER_REPO}/contents/bookings.json", "-q", ".content"],
            text=True,
            timeout=10,
        )
        payload = json.loads(base64.b64decode(raw))
        return parse_scheduler_bookings(payload, uid=uid)
    except Exception:
        return []


def parse_nvidia_smi_csv(csv_text: str) -> list[GPUStatus]:
    """Parse ``nvidia-smi --query-gpu=index,name,memory.used,memory.free,memory.total,utilization.gpu --format=csv,noheader,nounits``."""
    gpus: list[GPUStatus] = []
    for raw_line in csv_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        try:
            gpus.append(
                GPUStatus(
                    index=int(parts[0]),
                    name=parts[1],
                    memory_used_mib=int(float(parts[2])),
                    memory_free_mib=int(float(parts[3])),
                    memory_total_mib=int(float(parts[4])),
                    utilization_pct=int(float(parts[5])),
                )
            )
        except ValueError:
            continue
    return gpus


def select_available_gpu(
    gpus: Sequence[GPUStatus],
    *,
    min_free_mib: int = 25000,
    max_util_pct: int = 25,
) -> GPUStatus | None:
    """Return the eligible GPU with the most free VRAM, or None if all are busy."""
    candidates = [
        g
        for g in gpus
        if g.memory_free_mib >= min_free_mib and g.utilization_pct <= max_util_pct
    ]
    if not candidates:
        return None
    # Prefer lowest utilization first, then highest free memory, then lowest index.
    candidates.sort(key=lambda g: (g.utilization_pct, -g.memory_free_mib, g.index))
    return candidates[0]


def query_local_gpus() -> tuple[list[GPUStatus], str]:
    """Query ``nvidia-smi`` and return ``(parsed_gpus, formatted_table_text)``."""
    raw_csv = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.used,memory.free,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=15,
    )
    table_csv = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
            "--format=csv",
        ],
        text=True,
        timeout=15,
    ).strip()
    return parse_nvidia_smi_csv(raw_csv), table_csv


def has_local_artifacts(job: QueuedJob, repo_root: Path = REPO_ROOT) -> bool:
    """Return True if the job's trajectory JSONL and scored results JSON exist on disk."""
    jsonl = repo_root / job.out_jsonl
    res = repo_root / job.out_results
    return jsonl.exists() and jsonl.stat().st_size > 0 and res.exists() and res.stat().st_size > 0


def query_pr_state(branch_name: str, repo_root: Path = REPO_ROOT) -> str | None:
    """Return ``'OPEN'``, ``'MERGED'``, ``'CLOSED'``, or ``None`` for ``branch_name`` via ``gh pr view``."""
    proc = subprocess.run(
        ["gh", "pr", "view", branch_name, "--json", "state", "-q", ".state"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout.strip().upper() or None


def is_job_completed(
    job: QueuedJob,
    repo_root: Path = REPO_ROOT,
    *,
    require_merged_pr: bool = False,
    pr_state_fn: Callable[[str], str | None] | None = None,
) -> bool:
    """Return True if the job's artifacts exist and (when required) its PR is merged."""
    if not has_local_artifacts(job, repo_root=repo_root):
        return False
    if not require_merged_pr:
        return True
    checker = pr_state_fn or (lambda b: query_pr_state(b, repo_root=repo_root))
    state = checker(job.branch_name)
    # If no PR exists for the branch (None), only treat as completed if the
    # artifact is already tracked on origin/master; otherwise require MERGED.
    if state == "MERGED":
        return True
    if state in ("OPEN", "CLOSED"):
        return False
    tracked = subprocess.run(
        ["git", "cat-file", "-e", f"origin/master:{job.out_results}"],
        cwd=repo_root,
        capture_output=True,
        check=False,
    )
    return tracked.returncode == 0


def format_sampler_summary(job: QueuedJob) -> str:
    """Format model-family-accurate sampler settings for PR descriptions and baselines."""
    if job.model_family in ("llada-instruct", "llada-1.5"):
        return (
            f"model_family={job.model_family}, temperature={job.temperature}, "
            f"remasking=low_confidence, steps={job.steps}, gen_length={job.max_new_tokens}, "
            f"block_length={job.block_length}, confidence_eos_eot_inf=True, "
            f"frontier_width={job.frontier_width}"
        )
    return (
        f"model_family={job.model_family}, temperature={job.temperature}, "
        f"top_p={job.top_p}, alg=entropy, alg_temp=0.0, steps={job.steps}, "
        f"max_new_tokens={job.max_new_tokens}, "
        f"force_deliberation_steps={job.force_deliberation_steps}, "
        f"frontier_width={job.frontier_width}"
    )


def format_baselines_entry(
    *,
    job: QueuedJob,
    gpu: GPUStatus,
    nvidia_smi_table: str,
    git_rev: str,
    summary_text: str,
    eval_results: dict[str, Any],
    gate_result: dict[str, Any] | None,
    comparison: dict[str, Any] | None,
) -> str:
    """Format a provenance-complete Markdown section for ``results/baselines.md``."""
    agg = eval_results["aggregate"]
    per = eval_results["per_trajectory"]
    n_total = len(per)
    n_non_null = sum(1 for r in per if r.get("final_prediction") is not None)
    n_undecided = n_total - n_non_null
    status_label = (
        gate_result["status"]
        if gate_result is not None
        else f"smoke clean ({n_non_null}/{n_total} decided)"
    )
    today = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")

    lines = [
        "",
        f"## {job.title} ({status_label})",
        "",
        f"- **Date (UTC):** {today}",
        f"- **Job ID:** `{job.job_id}`",
        f"- **Git revision (`git rev-parse HEAD` at run):** `{git_rev}`",
        "- **Environment:** conda env `cdr-gpu` (`torch 2.5.1+cu121`, `transformers 4.46.2`)",
        f"- **GPU:** index `{gpu.index}` (`CUDA_VISIBLE_DEVICES={gpu.index}`, `{gpu.name}`, `{gpu.memory_free_mib} MiB` free at launch)",
        "- **`nvidia-smi` at launch:**",
        "  ```",
    ]
    for row in nvidia_smi_table.splitlines():
        lines.append(f"  {row}")
    lines.extend(
        [
            "  ```",
            f"- **Sampler & adapter config:** `{format_sampler_summary(job)}`",
            "- **Artifacts:**",
            f"  - Input: `{job.items_path}`",
            f"  - Fixed IDs: `{job.out_ids}`",
            f"  - Trajectories: `{job.out_jsonl}`",
            f"  - Run log: `{job.out_log}`",
            f"  - Scored output: `{job.out_results}`",
            f"- **Undecided count:** **{n_undecided} undecided** ({n_non_null} / {n_total} non-null at final step).",
        ]
    )
    if gate_result is not None:
        lines.append(
            f"- **Reasoning smoke gate (`CURSOR_FOLLOWUP_2026-10-30` §5):** `{gate_result['status']}` "
            f"(commitment clocks on `{gate_result['n_with_commitment_clocks']}/{gate_result['n_decided']}`, "
            f"post-verdict change on `{gate_result['n_with_post_change']}`, "
            f"pre-verdict rationale on `{gate_result['n_with_pre_verdict_rationale']}`)."
        )
    lines.extend(
        [
            "",
            "### Pasted `traj-eval run` output",
            "",
            "```",
            summary_text.strip(),
            "```",
            "",
            f"### Pasted aggregate JSON (`{job.out_results}`)",
            "",
            "```json",
            json.dumps(agg, indent=2),
            "```",
        ]
    )
    if comparison is not None:
        b_lbl = comparison["base_label"]
        t_lbl = comparison["target_label"]
        sbs = comparison["rationalization_components_side_by_side"]
        lines.extend(
            [
                "",
                f"### Side-by-side rationalization components (`{b_lbl}` vs `{t_lbl}`)",
                "",
                f"| Component | `{b_lbl}` | `{t_lbl}` | Delta (`{t_lbl} - {b_lbl}`) |",
                "|---|---:|---:|---:|",
            ]
        )
        for comp_name, vals in sbs.items():
            lines.append(
                f"| `{comp_name}` | {vals[b_lbl]:.4f} | {vals[t_lbl]:.4f} | {vals['delta']:+.4f} |"
            )
    lines.append("")
    return "\n".join(lines)


def format_pr_body(
    *,
    job: QueuedJob,
    gpu: GPUStatus,
    pytest_out: str,
    eval_results: dict[str, Any],
    gate_result: dict[str, Any] | None,
) -> str:
    """Format a GitHub Pull Request body matching ``.github/pull_request_template.md``."""
    agg = eval_results["aggregate"]
    per = eval_results["per_trajectory"]
    n_total = len(per)
    n_non_null = sum(1 for r in per if r.get("final_prediction") is not None)
    n_undecided = n_total - n_non_null
    gate_str = (
        f"pass ({gate_result['status']})"
        if (gate_result is not None and gate_result.get("passed"))
        else (
            f"fail ({gate_result['status']})"
            if gate_result is not None
            else f"n/a ({n_non_null}/{n_total} decided, {n_undecided} undecided)"
        )
    )
    claim_note = (
        "5 items = pipeline check only"
        if n_total <= 5
        else f"{n_total} items = pre-registered benchmark slice; descriptive metrics only"
    )
    sampler_str = format_sampler_summary(job)
    return "\n".join(
        [
            "## What changed and why",
            "",
            f"Runs pre-registered GPU stage `{job.job_id}` ({job.title}) on GPU `{gpu.index}` (`{gpu.name}`, `{gpu.memory_free_mib} MiB` free at launch), scores the trajectories into `{job.out_results}`, and appends the full provenance and decomposed rationalization components (`early_factor={agg.get('mean_early_factor')}`, `post_change_rate={agg.get('mean_post_change_rate')}`, `length_growth={agg.get('mean_length_growth')}`, `gated_score={agg.get('mean_gated_rationalization_score')}`) to `results/baselines.md`. No metric definitions, null rules, or sampler configs were changed.",
            "",
            "## Checklist (fill in every item, do not delete items)",
            "",
            "- [x] One logical change only; branch is not `master`",
            "- [x] `python -m pytest -q` tail pasted below (verbatim, after this change)",
            "- [x] Tests added or updated for any metric, adapter, or extractor change (n/a: experiment artifact PR only)",
            "- [x] No weights in tests; no torch/transformers import at module import time in `src/`",
            "- [x] Null rule unchanged (`None` = no verdict, not a flip), or the change is documented in metrics, adapter, handoff, and paper outline together",
            f"- [x] Prompt template unchanged, or the new prompt is pre-registered with its own output files (`{job.items_path}` -> `{job.out_jsonl}`, never overwriting an existing condition)",
            f"- [x] Sampler config unchanged mid-run; any GPU run includes its run log path (`{job.out_log}`)",
            f"- [x] No paper claim beyond what the artifacts in this PR prove ({claim_note})",
            "- [x] Scope check: this PR does not add a new adapter, metric, credit scheme, intervention hook, or prompt condition unless the current follow-up doc explicitly allows it",
            "",
            "## Test output (verbatim tail)",
            "",
            "```",
            pytest_out.strip(),
            "```",
            "",
            "## GPU run (if any)",
            "",
            f"- Run log path: `{job.out_log}`",
            f"- GPU index and free memory at launch: GPU `{gpu.index}` (`{gpu.name}`, `{gpu.memory_free_mib} MiB` free, `{gpu.utilization_pct}%` util)",
            f"- Sampler config: `{sampler_str}`",
            f"- Gate result (pass / fail, with reason if fail): `{gate_str}`",
            "",
        ]
    )


def _capture_summary_text(results: dict[str, Any]) -> str:
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        _print_summary(results)
    return buf.getvalue()


def _write_queue_state(state: dict[str, Any], state_path: Path = STATE_FILE) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def sync_master_and_checkout_branch(
    branch_name: str | None = None, repo_root: Path = REPO_ROOT
) -> str:
    """Fetch ``origin/master``, update local ``master``, and optionally create ``branch_name`` from ``origin/master``."""
    subprocess.run(["git", "fetch", "origin", "master"], cwd=repo_root, check=True)
    subprocess.run(["git", "checkout", "master"], cwd=repo_root, check=True)
    subprocess.run(["git", "merge", "--ff-only", "origin/master"], cwd=repo_root, check=True)
    if branch_name:
        subprocess.run(
            ["git", "checkout", "-B", branch_name, "origin/master"],
            cwd=repo_root,
            check=True,
        )
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()


def build_smoke_command(
    job: QueuedJob,
    *,
    python_bin: str = DEFAULT_CDR_GPU_PYTHON,
    fake: bool = False,
) -> list[str]:
    """Build the command line for ``experiments/run_dream_arcc_smoke.py``."""
    cmd = [
        python_bin,
        str(REPO_ROOT / "experiments" / "run_dream_arcc_smoke.py"),
        "--items",
        job.items_path,
        "--out-jsonl",
        job.out_jsonl,
        "--out-ids",
        job.out_ids,
        "--out-log",
        job.out_log,
        "--model-family",
        job.model_family,
        "--task",
        job.task,
        "--temperature",
        str(job.temperature),
        "--top-p",
        str(job.top_p),
        "--steps",
        str(job.steps),
        "--max-new-tokens",
        str(job.max_new_tokens),
        "--block-length",
        str(job.block_length),
        "--force-deliberation-steps",
        str(job.force_deliberation_steps),
    ]
    if job.frontier_width is not None:
        cmd.extend(["--frontier-width", str(job.frontier_width)])
    if fake:
        cmd.extend(["--fake", "--n-items", "5"])
    return cmd


def run_single_queued_job(
    job: QueuedJob,
    gpu: GPUStatus,
    nvidia_smi_table: str,
    *,
    repo_root: Path = REPO_ROOT,
    python_bin: str = DEFAULT_CDR_GPU_PYTHON,
    fake: bool = False,
    auto_commit_push: bool = True,
) -> dict[str, Any]:
    """Execute one queued job, score it, update baselines.md, and open a PR."""
    branch_name = job.branch_name
    if auto_commit_push:
        git_rev = sync_master_and_checkout_branch(branch_name=branch_name, repo_root=repo_root)
    else:
        git_rev = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip()

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu.index)
    env["OMP_NUM_THREADS"] = "2"
    env["MKL_NUM_THREADS"] = "2"
    env["OPENBLAS_NUM_THREADS"] = "2"
    env["PYTHONPATH"] = str(repo_root / "src")

    cmd = build_smoke_command(job, python_bin=python_bin, fake=fake)
    print(f"[gpu-queue] Launching {job.job_id} on GPU {gpu.index} ({gpu.memory_free_mib} MiB free)...", flush=True)
    proc = subprocess.run(cmd, cwd=repo_root, env=env, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Job {job.job_id} exited with code {proc.returncode}")

    trajs = load_trajectories(repo_root / job.out_jsonl)
    eval_results = evaluate_trajectories(trajs)
    gate_result: dict[str, Any] | None = None
    if job.require_reasoning_gate:
        gate_result = check_reasoning_smoke_gate(trajs)
        eval_results["reasoning_smoke_gate"] = gate_result

    save_results(eval_results, repo_root / job.out_results)
    summary_text = _capture_summary_text(eval_results)

    comparison: dict[str, Any] | None = None
    if job.compare_base_jsonl and job.compare_out_json:
        base_path = repo_root / job.compare_base_jsonl
        if base_path.exists():
            base_trajs = load_trajectories(base_path)
            comparison = compare_conditions(
                base_trajs,
                trajs,
                base_label=job.compare_base_label,
                target_label=job.compare_target_label,
            )
            save_results(comparison, repo_root / job.compare_out_json)

    baselines_path = repo_root / "results" / "baselines.md"
    entry_md = format_baselines_entry(
        job=job,
        gpu=gpu,
        nvidia_smi_table=nvidia_smi_table,
        git_rev=git_rev,
        summary_text=summary_text,
        eval_results=eval_results,
        gate_result=gate_result,
        comparison=comparison,
    )
    if baselines_path.exists():
        baselines_path.write_text(
            baselines_path.read_text(encoding="utf-8").rstrip() + "\n" + entry_md,
            encoding="utf-8",
        )

    pr_url: str | None = None
    if auto_commit_push:
        files_to_stage = [
            job.out_jsonl,
            job.out_ids,
            job.out_log,
            job.out_results,
            "results/baselines.md",
        ]
        if job.compare_out_json and (repo_root / job.compare_out_json).exists():
            files_to_stage.append(job.compare_out_json)
        pytest_out = subprocess.check_output(
            [str(repo_root / ".venv" / "bin" / "python"), "-m", "pytest", "-q"],
            cwd=repo_root,
            text=True,
        ).strip()
        subprocess.run(["git", "add", "--", *files_to_stage], cwd=repo_root, check=True)
        gate_line = (
            gate_result["status"]
            if gate_result is not None
            else "smoke clean."
        )
        pr_title = f"feat(smoke): run {job.job_id} on GPU {gpu.index}"
        commit_msg = (
            f"{pr_title}\n\n"
            f"{gate_line}\n\n"
            f"- Model family: {job.model_family} ({job.task})\n"
            f"- GPU: {gpu.index} ({gpu.name}, {gpu.memory_free_mib} MiB free at launch)\n"
            f"- Scored results: {job.out_results} and updated results/baselines.md.\n\n"
            f"pytest -q:\n{pytest_out}"
        )
        subprocess.run(["git", "commit", "-m", commit_msg], cwd=repo_root, check=True)
        subprocess.run(
            ["git", "push", "-u", "origin", branch_name, "--force-with-lease"],
            cwd=repo_root,
            check=True,
        )
        pr_body = format_pr_body(
            job=job,
            gpu=gpu,
            pytest_out=pytest_out,
            eval_results=eval_results,
            gate_result=gate_result,
        )
        pr_proc = subprocess.run(
            [
                "gh",
                "pr",
                "create",
                "--base",
                "master",
                "--head",
                branch_name,
                "--title",
                pr_title,
                "--body",
                pr_body,
            ],
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
        if pr_proc.returncode == 0 and pr_proc.stdout.strip():
            pr_url = pr_proc.stdout.strip()
        else:
            view_proc = subprocess.run(
                ["gh", "pr", "view", branch_name, "--json", "url", "-q", ".url"],
                cwd=repo_root,
                text=True,
                capture_output=True,
                check=False,
            )
            if view_proc.returncode == 0 and view_proc.stdout.strip():
                pr_url = view_proc.stdout.strip()
            else:
                raise RuntimeError(
                    f"Failed to create or locate PR for {branch_name}: "
                    f"create_stderr={pr_proc.stderr.strip()!r}, "
                    f"view_stderr={view_proc.stderr.strip()!r}"
                )
        print(f"[gpu-queue] Opened PR for {job.job_id}: {pr_url}", flush=True)

    return {
        "job_id": job.job_id,
        "branch": branch_name,
        "pr_url": pr_url,
        "gpu_index": gpu.index,
        "accuracy": eval_results["aggregate"]["accuracy"],
        "gate": gate_result,
    }


def wait_for_branch_pr_merge(
    *,
    job_id: str,
    branch_name: str,
    pr_url: str | None,
    poll_interval_sec: float,
    completed_runs: list[dict[str, Any]],
    repo_root: Path = REPO_ROOT,
    state_path: Path = STATE_FILE,
    pr_state_fn: Callable[[str], str | None] | None = None,
    sync_master_fn: Callable[[], str] | None = None,
    max_lookup_failures: int = 5,
) -> str | None:
    """Wait until ``branch_name`` PR is ``MERGED`` and ``master`` is synced, or return a halt reason."""
    checker = pr_state_fn or (lambda b: query_pr_state(b, repo_root=repo_root))
    syncer = sync_master_fn or (lambda: sync_master_and_checkout_branch(None, repo_root=repo_root))
    consecutive_none = 0
    while True:
        pr_state = checker(branch_name)
        if pr_state == "MERGED":
            try:
                syncer()
            except Exception as exc:
                return f"Failed to sync master after {branch_name} merged: {exc}"
            print(
                f"[gpu-queue] PR for {job_id} ({branch_name}) MERGED and master synced.",
                flush=True,
            )
            return None
        if pr_state == "CLOSED":
            return f"PR for {job_id} ({branch_name}) was closed without merging"
        if pr_state is None:
            consecutive_none += 1
            if consecutive_none >= max_lookup_failures:
                return (
                    f"Failed to query PR state for {job_id} ({branch_name}) "
                    f"after {consecutive_none} attempts"
                )
        else:
            consecutive_none = 0

        _write_queue_state(
            {
                "status": "waiting_for_pr_merge",
                "pid": os.getpid(),
                "awaiting_job_id": job_id,
                "awaiting_pr_branch": branch_name,
                "pr_url": pr_url,
                "pr_state": pr_state,
                "completed_runs": completed_runs,
            },
            state_path=state_path,
        )
        time.sleep(poll_interval_sec)


def run_queue_loop(
    jobs: Sequence[QueuedJob] = DEFAULT_JOB_QUEUE,
    *,
    min_free_mib: int = 25000,
    max_util_pct: int = 25,
    poll_interval_sec: float = 45.0,
    confirm_delay_sec: float = 10.0,
    max_jobs: int | None = None,
    fake: bool = False,
    auto_commit_push: bool = True,
    wait_for_pr_merge: bool = True,
    not_before_utc: str | None = None,
    repo_root: Path = REPO_ROOT,
    state_path: Path = STATE_FILE,
) -> dict[str, Any]:
    """Poll for a free GPU and execute pending jobs in strict PR-gated sequence."""
    completed_runs: list[dict[str, Any]] = []
    halted_reason: str | None = None
    require_pr = bool(auto_commit_push and wait_for_pr_merge and not fake)

    # Step 0: If a previous daemon run left an open PR awaiting review/merge,
    # recover and wait for that PR to merge before selecting any new job.
    if require_pr:
        for job in jobs:
            if has_local_artifacts(job, repo_root=repo_root):
                pr_state = query_pr_state(job.branch_name, repo_root=repo_root)
                if pr_state == "OPEN":
                    print(
                        f"[gpu-queue] Recovering merge wait for open PR on {job.branch_name}...",
                        flush=True,
                    )
                    halted_reason = wait_for_branch_pr_merge(
                        job_id=job.job_id,
                        branch_name=job.branch_name,
                        pr_url=None,
                        poll_interval_sec=poll_interval_sec,
                        completed_runs=completed_runs,
                        repo_root=repo_root,
                        state_path=state_path,
                    )
                    if halted_reason:
                        break
                elif pr_state == "CLOSED":
                    halted_reason = (
                        f"Existing PR for {job.job_id} ({job.branch_name}) was closed without merging"
                    )
                    break

    if halted_reason:
        summary = {
            "status": "halted_on_gate",
            "pid": os.getpid(),
            "halted_reason": halted_reason,
            "completed_runs": completed_runs,
        }
        _write_queue_state(summary, state_path=state_path)
        return summary

    pending = [
        j
        for j in jobs
        if not is_job_completed(j, repo_root=repo_root, require_merged_pr=require_pr)
    ]
    if max_jobs is not None:
        pending = pending[:max_jobs]

    _write_queue_state(
        {
            "status": "watching",
            "pid": os.getpid(),
            "min_free_mib": min_free_mib,
            "max_util_pct": max_util_pct,
            "wait_for_pr_merge": wait_for_pr_merge,
            "not_before_utc": not_before_utc,
            "pending_jobs": [j.job_id for j in pending],
            "completed_runs": completed_runs,
        },
        state_path=state_path,
    )

    for job in pending:
        if is_job_completed(job, repo_root=repo_root, require_merged_pr=require_pr):
            continue

        # Optional reservation start gate (--not-before YYYY-MM-DDTHH:MM:SSZ).
        if not_before_utc and not fake:
            target_dt = dt.datetime.fromisoformat(not_before_utc.replace("Z", "+00:00"))
            while dt.datetime.now(dt.timezone.utc) < target_dt:
                _write_queue_state(
                    {
                        "status": "waiting_for_reservation_window",
                        "pid": os.getpid(),
                        "not_before_utc": not_before_utc,
                        "next_job": job.job_id,
                        "completed_runs": completed_runs,
                    },
                    state_path=state_path,
                )
                time.sleep(poll_interval_sec)

        # Wait until a GPU is available on two consecutive checks.
        while True:
            if fake:
                chosen = GPUStatus(
                    index=0,
                    name="FAKE-GPU",
                    memory_used_mib=0,
                    memory_free_mib=49140,
                    memory_total_mib=49140,
                    utilization_pct=0,
                )
                table_csv = "index, name, memory.used [MiB], memory.total [MiB], utilization.gpu [%]\n0, FAKE-GPU, 0 MiB, 49140 MiB, 0 %"
                break

            gpus_1, table_1 = query_local_gpus()
            cand_1 = select_available_gpu(
                gpus_1, min_free_mib=min_free_mib, max_util_pct=max_util_pct
            )
            if cand_1 is not None:
                if confirm_delay_sec > 0:
                    time.sleep(confirm_delay_sec)
                gpus_2, table_2 = query_local_gpus()
                cand_2 = select_available_gpu(
                    gpus_2, min_free_mib=min_free_mib, max_util_pct=max_util_pct
                )
                if cand_2 is not None:
                    chosen = cand_2
                    table_csv = table_2
                    break

            max_free = max((g.memory_free_mib for g in gpus_1), default=0)
            print(
                f"[gpu-queue] Waiting for GPU (>={min_free_mib} MiB free, <={max_util_pct}% util) "
                f"for next job {job.job_id!r}; current max free = {max_free} MiB.",
                flush=True,
            )
            _write_queue_state(
                {
                    "status": "waiting_for_gpu",
                    "pid": os.getpid(),
                    "next_job": job.job_id,
                    "current_max_free_mib": max_free,
                    "pending_jobs": [
                        j.job_id
                        for j in pending
                        if not is_job_completed(j, repo_root=repo_root, require_merged_pr=require_pr)
                    ],
                    "completed_runs": completed_runs,
                },
                state_path=state_path,
            )
            time.sleep(poll_interval_sec)

        _write_queue_state(
            {
                "status": "running_job",
                "pid": os.getpid(),
                "current_job": job.job_id,
                "gpu_index": chosen.index,
                "gpu_free_mib": chosen.memory_free_mib,
                "completed_runs": completed_runs,
            },
            state_path=state_path,
        )

        res = run_single_queued_job(
            job,
            chosen,
            table_csv,
            repo_root=repo_root,
            python_bin=sys.executable if fake else DEFAULT_CDR_GPU_PYTHON,
            fake=fake,
            auto_commit_push=auto_commit_push,
        )
        completed_runs.append(res)

        gate = res.get("gate")
        if gate is not None and not gate.get("passed", False):
            halted_reason = str(gate.get("status", "reasoning smoke gate failed"))
            print(f"[gpu-queue] HALTING QUEUE per protocol gate: {halted_reason}", flush=True)
            break

        # Per docs/PR_WORKFLOW.md §§5-6, wait for the managing agent (Muse) to
        # review and merge this job's PR before starting the next scope/job.
        if require_pr:
            halted_reason = wait_for_branch_pr_merge(
                job_id=job.job_id,
                branch_name=str(res["branch"]),
                pr_url=res.get("pr_url"),
                poll_interval_sec=poll_interval_sec,
                completed_runs=completed_runs,
                repo_root=repo_root,
                state_path=state_path,
            )
            if halted_reason:
                print(f"[gpu-queue] HALTING QUEUE: {halted_reason}", flush=True)
                break

    final_status = "halted_on_gate" if halted_reason else "completed"
    summary = {
        "status": final_status,
        "pid": os.getpid(),
        "halted_reason": halted_reason,
        "completed_runs": completed_runs,
    }
    _write_queue_state(summary, state_path=state_path)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Sequential GPU watcher & job queue for traj-eval")
    ap.add_argument("--min-free-mib", type=int, default=25000, help="Minimum free VRAM in MiB (default: 25000)")
    ap.add_argument("--max-util-pct", type=int, default=25, help="Maximum GPU utilization %% at launch (default: 25)")
    ap.add_argument("--poll-interval", type=float, default=45.0, help="Seconds between nvidia-smi polls")
    ap.add_argument("--confirm-delay", type=float, default=10.0, help="Seconds between double-check polls")
    ap.add_argument("--max-jobs", type=int, default=None, help="Stop after running at most N pending jobs")
    ap.add_argument("--not-before", default=None, help="ISO-8601 UTC start timestamp before which no GPU jobs run")
    ap.add_argument("--fake", action="store_true", help="Dry-run with stub model and fake GPU")
    ap.add_argument("--no-git", action="store_true", help="Do not run git branch/commit/push/PR")
    ap.add_argument(
        "--no-wait-for-pr-merge",
        action="store_true",
        help="Do not wait for each job's PR to be merged before running the next job",
    )
    ap.add_argument("--status", action="store_true", help="Print current queue + GPU + scheduler booking status and exit")
    args = ap.parse_args(argv)

    if args.status:
        gpus, table = query_local_gpus()
        cand = select_available_gpu(gpus, min_free_mib=args.min_free_mib, max_util_pct=args.max_util_pct)
        print(table)
        print(f"\nEligible GPU right now (>={args.min_free_mib} MiB free, <={args.max_util_pct}% util): {cand}")
        bookings = fetch_scheduler_bookings("arrdel")
        if bookings:
            print(f"\nActive Lab GPU Scheduler reservations ({SCHEDULER_REPO}, user=arrdel):")
            for b in bookings:
                print(
                    f"  - [{b.get('id')}] node={b.get('gpuId')} cards={b.get('slots')} "
                    f"{b.get('start')} -> {b.get('end')} ({b.get('modelSize')}, {b.get('vramGb')} GB/card)"
                )
        print("\nQueued traj-eval stages:")
        for j in DEFAULT_JOB_QUEUE:
            done = is_job_completed(j)
            print(f"  [{'DONE' if done else 'PEND'}] {j.job_id} ({j.branch_name}): {j.title}")
        if STATE_FILE.exists():
            print("\nLast daemon state:")
            print(STATE_FILE.read_text(encoding="utf-8"))
        return 0

    # Single-instance lockfile guard so duplicate daemons never collide.
    if LOCK_FILE.exists():
        try:
            old_pid = int(LOCK_FILE.read_text(encoding="utf-8").strip())
            os.kill(old_pid, 0)
            print(f"[gpu-queue] Queue daemon already running (PID {old_pid}). Exiting.", file=sys.stderr)
            return 0
        except (ValueError, ProcessLookupError, PermissionError):
            LOCK_FILE.unlink(missing_ok=True)

    LOCK_FILE.write_text(f"{os.getpid()}\n", encoding="utf-8")
    try:
        summary = run_queue_loop(
            min_free_mib=args.min_free_mib,
            max_util_pct=args.max_util_pct,
            poll_interval_sec=args.poll_interval,
            confirm_delay_sec=args.confirm_delay,
            max_jobs=args.max_jobs,
            fake=args.fake,
            auto_commit_push=not args.no_git,
            wait_for_pr_merge=not args.no_wait_for_pr_merge,
            not_before_utc=args.not_before,
        )
        return 1 if summary["status"] == "halted_on_gate" else 0
    finally:
        LOCK_FILE.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
