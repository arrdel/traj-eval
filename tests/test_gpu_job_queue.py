# Copyright (C) 2026 Lenovo
# All Rights Reserved.
# Lenovo Confidential Restricted.

"""Unit tests for the sequential GPU watcher and PR-gated job queue."""

from pathlib import Path

from scripts.gpu_job_queue import (
    DEFAULT_JOB_QUEUE,
    GPUStatus,
    QueuedJob,
    build_smoke_command,
    format_baselines_entry,
    format_pr_body,
    format_sampler_summary,
    is_job_completed,
    parse_nvidia_smi_csv,
    parse_scheduler_bookings,
    select_available_gpu,
    wait_for_branch_pr_merge,
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
    chosen = select_available_gpu(gpus, min_free_mib=25000, max_util_pct=25)
    assert chosen is not None
    assert chosen.index == 6
    assert chosen.memory_free_mib == 39140

    busy_only = [gpus[0], gpus[2]]
    assert select_available_gpu(busy_only, min_free_mib=25000, max_util_pct=25) is None


def test_default_queue_starts_with_reasoning_smoke_and_enforces_gate():
    assert len(DEFAULT_JOB_QUEUE) >= 5
    first = DEFAULT_JOB_QUEUE[0]
    assert first.job_id == "dream_arcc_smoke5_reasoning"
    assert first.require_reasoning_gate is True
    assert first.items_path == "results/dream_arcc_smoke5_reasoning_input.json"
    assert first.out_jsonl == "results/dream_arcc_smoke5_reasoning.jsonl"


def test_is_job_completed_respects_pr_merge_state(tmp_path: Path):
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
    assert is_job_completed(job, repo_root=tmp_path, require_merged_pr=False) is True

    # When require_merged_pr=True, an OPEN or CLOSED PR is NOT treated as completed.
    assert (
        is_job_completed(
            job,
            repo_root=tmp_path,
            require_merged_pr=True,
            pr_state_fn=lambda _: "OPEN",
        )
        is False
    )
    assert (
        is_job_completed(
            job,
            repo_root=tmp_path,
            require_merged_pr=True,
            pr_state_fn=lambda _: "CLOSED",
        )
        is False
    )
    assert (
        is_job_completed(
            job,
            repo_root=tmp_path,
            require_merged_pr=True,
            pr_state_fn=lambda _: "MERGED",
        )
        is True
    )

    cmd = build_smoke_command(job, python_bin="/usr/bin/python3", fake=True)
    assert "--model-family" in cmd
    assert "llada-instruct" in cmd
    assert "--frontier-width" in cmd
    assert "16" in cmd
    assert "--fake" in cmd


def test_wait_for_branch_pr_merge_handles_merged_closed_and_lookup_failure(tmp_path: Path):
    state_file = tmp_path / "state.json"
    synced: list[bool] = []
    states = iter(["OPEN", "MERGED"])
    halt = wait_for_branch_pr_merge(
        job_id="j1",
        branch_name="feat/smoke-j1",
        pr_url="https://example.com/pr/1",
        poll_interval_sec=0.0,
        completed_runs=[],
        repo_root=tmp_path,
        state_path=state_file,
        pr_state_fn=lambda _: next(states),
        sync_master_fn=lambda: (synced.append(True) or "abc"),
    )
    assert halt is None
    assert synced == [True]

    # Closed PR halts immediately.
    halt_closed = wait_for_branch_pr_merge(
        job_id="j2",
        branch_name="feat/smoke-j2",
        pr_url="https://example.com/pr/2",
        poll_interval_sec=0.0,
        completed_runs=[],
        repo_root=tmp_path,
        state_path=state_file,
        pr_state_fn=lambda _: "CLOSED",
        sync_master_fn=lambda: "abc",
    )
    assert halt_closed is not None
    assert "closed without merging" in halt_closed

    # Persistent lookup failure halts after max_lookup_failures.
    halt_none = wait_for_branch_pr_merge(
        job_id="j3",
        branch_name="feat/smoke-j3",
        pr_url=None,
        poll_interval_sec=0.0,
        completed_runs=[],
        repo_root=tmp_path,
        state_path=state_file,
        pr_state_fn=lambda _: None,
        sync_master_fn=lambda: "abc",
        max_lookup_failures=2,
    )
    assert halt_none is not None
    assert "Failed to query PR state" in halt_none


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


def test_format_pr_body_matches_pr_template_and_model_family():
    job_dream5 = DEFAULT_JOB_QUEUE[0]
    gpu = GPUStatus(
        index=4,
        name="NVIDIA RTX A6000",
        memory_used_mib=16899,
        memory_free_mib=32241,
        memory_total_mib=49140,
        utilization_pct=0,
    )
    eval_5 = {
        "per_trajectory": [{"id": f"q{i}", "final_prediction": "C"} for i in range(5)],
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
    }
    body_5 = format_pr_body(
        job=job_dream5,
        gpu=gpu,
        pytest_out="65 passed in 0.35s",
        eval_results=eval_5,
        gate_result=gate_result,
    )
    assert "## What changed and why" in body_5
    assert "## Checklist (fill in every item, do not delete items)" in body_5
    assert "- [x] One logical change only; branch is not `master`" in body_5
    assert "(5 items = pipeline check only)" in body_5
    assert "alg=entropy" in body_5
    assert "pass (reasoning smoke clean)" in body_5

    # 64-item and LLaDA formatting check.
    job_64 = DEFAULT_JOB_QUEUE[1]
    eval_64 = {
        "per_trajectory": [{"id": f"q{i}", "final_prediction": "C"} for i in range(64)],
        "aggregate": eval_5["aggregate"],
    }
    body_64 = format_pr_body(
        job=job_64,
        gpu=gpu,
        pytest_out="65 passed in 0.35s",
        eval_results=eval_64,
        gate_result=None,
    )
    assert "(64 items = pre-registered benchmark slice; descriptive metrics only)" in body_64

    llada_job = DEFAULT_JOB_QUEUE[5]
    assert "remasking=low_confidence" in format_sampler_summary(llada_job)
    assert "confidence_eos_eot_inf=True" in format_sampler_summary(llada_job)


def test_parse_scheduler_bookings():
    payload = {
        "bookings": [
            {
                "id": "b_1",
                "uid": "arrdel",
                "gpuId": "ada6000",
                "start": "2026-11-16",
                "end": "2026-11-30",
                "slots": [0, 1, 2],
            },
            {
                "id": "b_2",
                "uid": "arrdel",
                "gpuId": "lab2",
                "start": "2026-11-01",
                "end": "2026-11-15",
                "slots": [0, 1, 2, 3],
            },
            {
                "id": "b_3",
                "uid": "arrdel",
                "gpuId": "lab2",
                "start": "2026-10-20",
                "end": "2026-10-21",
                "closedAt": "2026-10-21T00:00:00Z",
            },
            {
                "id": "b_4",
                "uid": "other",
                "gpuId": "ada6000",
                "start": "2026-11-01",
                "end": "2026-11-15",
            },
        ]
    }
    active = parse_scheduler_bookings(payload, uid="arrdel")
    assert [b["id"] for b in active] == ["b_2", "b_1"]
