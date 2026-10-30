# Copyright (C) 2026 Lenovo
# All Rights Reserved.
# Lenovo Confidential Restricted.

"""Unit tests for the sequential GPU watcher and job queue."""

from pathlib import Path

import pytest

from scripts.gpu_job_queue import (
    DEFAULT_JOB_QUEUE,
    GPUStatus,
    QueuedJob,
    build_smoke_command,
    format_baselines_entry,
    is_job_completed,
    parse_nvidia_smi_csv,
    select_available_gpu,
)


def test_parse_nvidia_smi_csv_and_select_gpu():
    sample_csv = (
        "0, NVIDIA RTX A6000, 32891, 15785, 49140, 100\n"
        "4, NVIDIA RTX A6000, 16899, 32241, 49140, 0\n"
        "5, NVIDIA RTX A6000, 12000, 37140, 49140, 95\n"
        "6, NVIDIA RTX A6000, 10000, 39140, 49140, 0\n"
    )
    gpus = parse_nvidia_smi_csv(sample_csv)
    assert len(gpus) == 4
    # Cards 4 and 6 both have >= 25000 MiB free and <= 25% util; card 6 has more free VRAM.
    chosen = select_available_gpu(gpus, min_free_mib=25000, max_util_pct=25)
    assert chosen is not None
    assert chosen.index == 6
    assert chosen.memory_free_mib == 39140

    # When all cards are above utilization threshold or below free memory threshold, returns None.
    busy_only = [gpus[0], gpus[2]]
    assert select_available_gpu(busy_only, min_free_mib=25000, max_util_pct=25) is None


def test_default_queue_starts_with_reasoning_smoke_and_enforces_gate():
    assert len(DEFAULT_JOB_QUEUE) >= 5
    first = DEFAULT_JOB_QUEUE[0]
    assert first.job_id == "dream_arcc_smoke5_reasoning"
    assert first.require_reasoning_gate is True
    assert first.items_path == "results/dream_arcc_smoke5_reasoning_input.json"
    assert first.out_jsonl == "results/dream_arcc_smoke5_reasoning.jsonl"


def test_is_job_completed_and_build_command(tmp_path: Path):
    job = QueuedJob(
        job_id="test_job",
        title="Test Job",
        model_family="llada-instruct",
        task="arcc",
        items_path="results/in.json",
        out_jsonl="results/out.jsonl",
        out_ids="results/ids.json",
        out_log="results/log.json",
        out_results="results/res.json",
        frontier_width=16,
    )
    assert is_job_completed(job, repo_root=tmp_path) is False
    (tmp_path / "results").mkdir(parents=True)
    (tmp_path / "results" / "out.jsonl").write_text('{"id":"1"}\n')
    assert is_job_completed(job, repo_root=tmp_path) is False
    (tmp_path / "results" / "res.json").write_text('{"aggregate":{}}\n')
    assert is_job_completed(job, repo_root=tmp_path) is True

    cmd = build_smoke_command(job, python_bin="/usr/bin/python3", fake=True)
    assert "--model-family" in cmd
    assert "llada-instruct" in cmd
    assert "--frontier-width" in cmd
    assert "16" in cmd
    assert "--fake" in cmd


def test_format_baselines_entry_includes_provenance_and_side_by_side():
    job = DEFAULT_JOB_QUEUE[0]
    gpu = GPUStatus(
        index=4,
        name="NVIDIA RTX A6000",
        memory_used_mib=16899,
        memory_free_mib=32241,
        memory_total_mib=49140,
        utilization_pct=0,
    )
    eval_results = {
        "per_trajectory": [{"id": "q1", "final_prediction": "C"}],
        "aggregate": {
            "n_trajectories": 5,
            "accuracy": 1.0,
            "mean_early_factor": 0.85,
            "mean_post_change_rate": 0.4,
            "mean_length_growth": 0.5,
            "mean_rationalization_score": 0.6,
            "mean_gated_rationalization_score": 0.3683,
        },
    }
    gate_result = {
        "passed": True,
        "status": "reasoning smoke clean",
        "n_decided": 5,
        "n_with_commitment_clocks": 5,
        "n_with_post_change": 4,
        "n_with_pre_verdict_rationale": 5,
    }
    comparison = {
        "base_label": "direct",
        "target_label": "reasoning",
        "rationalization_components_side_by_side": {
            "early_factor": {"direct": 0.9875, "reasoning": 0.85, "delta": -0.1375},
            "post_change_rate": {"direct": 0.0, "reasoning": 0.4, "delta": 0.4},
            "length_growth": {"direct": 0.0, "reasoning": 0.5, "delta": 0.5},
            "score": {"direct": 0.395, "reasoning": 0.6, "delta": 0.205},
            "gated_score": {"direct": 0.0, "reasoning": 0.3683, "delta": 0.3683},
        },
    }
    md = format_baselines_entry(
        job=job,
        gpu=gpu,
        nvidia_smi_table="index, name\n4, NVIDIA RTX A6000",
        git_rev="abc1234",
        summary_text="=== traj-eval summary ===",
        eval_results=eval_results,
        gate_result=gate_result,
        comparison=comparison,
    )
    assert "reasoning smoke clean" in md
    assert "CUDA_VISIBLE_DEVICES=4" in md
    assert "0 undecided" in md
    assert "Side-by-side rationalization components (`direct` vs `reasoning`)" in md
    assert "| `post_change_rate` | 0.0000 | 0.4000 | +0.4000 |" in md
