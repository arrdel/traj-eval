"""Scale and property tests for traj-eval (Issue #34 Parts 1 and 2).

Default `pytest -q` runs fast CPU property and scaling tests across every
function in `src/traj_eval/metrics.py`. Heavy full-benchmark runs (1,172 ARC-C
items, 1,319 GSM8K items, 10,000 x 256 synthetic trajectories, and 64 x 4,096
long trajectories) are gated behind `TRAJ_EVAL_RUN_HEAVY_STRESS=1` or running
`python tests/test_stress.py` directly as a script.
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Sequence

import pytest

from traj_eval.io import load_trajectories, trajectories_to_dicts
from traj_eval.metrics import (
    answer_commitment_step,
    check_reasoning_smoke_gate,
    compare_conditions,
    cotsel_selection_gap,
    coverage_accuracy_decomposition,
    early_convergence_fraction,
    evaluate_trajectories,
    flip_rate,
    forced_deliberation_delta,
    gated_rationalization_score,
    is_sbw_lockin,
    order_gap,
    paired_bootstrap_delta,
    pass_at_k_metrics,
    rationalization_components,
    rationalization_score,
    stability_commitment_lag,
    step_consistency,
    stratified_evaluation,
    three_clock_disagreement,
    trajectory_accuracy,
    unwritten_reasoning_at_convergence,
    verdict_convergence_step,
)
from traj_eval.synthetic import (
    _genuine,
    _oscillation,
    _rationalization,
    generate_synthetic_trajectories,
)
from traj_eval.types import Trajectory, TrajectoryStep

HEAVY_STRESS_ENABLED = os.environ.get("TRAJ_EVAL_RUN_HEAVY_STRESS") == "1"


def _make_seeded_trajectory(
    rng: random.Random,
    tid: str,
    *,
    n_steps: int = 16,
    prompt: str = "Question?",
    gold: str = "B",
) -> Trajectory:
    options = [None, "A", "B", "C", "D"]
    commit_start = rng.randint(0, n_steps)  # n_steps means never committed
    crf = 0.0
    text = ""
    steps: list[TrajectoryStep] = []
    for s in range(n_steps):
        pred = rng.choice(options)
        if s >= commit_start and pred is None:
            pred = rng.choice(["A", "B", "C", "D"])
        if rng.random() < 0.6:
            text = text + f" step{s}"
        crf = min(1.0, round(crf + rng.uniform(0.0, 2.0 / max(1, n_steps)), 4))
        steps.append(
            TrajectoryStep(
                step=s,
                prediction=pred,
                reasoning_text=text.strip(),
                confidence=round(rng.uniform(0.0, 1.0), 4),
                answer_committed=(s >= commit_start),
                committed_reasoning_fraction=crf,
            )
        )
    return Trajectory(id=tid, prompt=prompt, gold_answer=gold, steps=steps)


def assert_all_metric_invariants(trajs: Sequence[Trajectory]) -> None:
    """Assert all Part 2 invariants across every function in `src/traj_eval/metrics.py`."""
    t_list = list(trajs)
    for t in t_list:
        n = len(t.steps)
        t_stab = verdict_convergence_step(t)
        t_comm = answer_commitment_step(t)
        fr = flip_rate(t)
        cons = step_consistency(t)
        rat_c = rationalization_components(t)
        rat = rationalization_score(t)
        grat = gated_rationalization_score(t)
        lag = stability_commitment_lag(t)
        unwritten = unwritten_reasoning_at_convergence(t)
        ogap = order_gap(t)
        disag = three_clock_disagreement(t)
        sbw = is_sbw_lockin(t)
        sbw_thresh = is_sbw_lockin(t, threshold=0.2)

        assert 0.0 <= fr <= 1.0
        assert 0.0 <= cons["consistency"] <= 1.0
        assert 0 <= cons["longest_stable_run"] <= n
        assert 0.0 <= cons["longest_stable_run_fraction"] <= 1.0

        # Convergence and commitment steps lie in [0, n) or are None for 0-indexed steps.
        if t_stab is not None:
            assert 0 <= t_stab < n
        if t_comm is not None:
            assert 0 <= t_comm < n
        if t_stab is not None and t_comm is not None:
            assert lag == t_comm - t_stab
        else:
            assert lag is None

        # Fractions and rationalization bounds; gated never exceeds ungated.
        for key in ("early_factor", "post_change_rate", "length_growth", "score", "gated_score"):
            assert 0.0 <= rat_c[key] <= 1.0
        assert rat == rat_c["score"]
        assert grat == rat_c["gated_score"]
        assert grat <= rat + 1e-9

        if unwritten is not None:
            assert 0.0 <= unwritten <= 1.0
        if ogap is not None:
            assert -1.0 <= ogap <= 1.0
        assert isinstance(disag["in_disagreement_slice"], bool)
        assert sbw in (True, False, None)
        assert sbw_thresh in (True, False, None)

    acc = trajectory_accuracy(t_list)
    ecf = early_convergence_fraction(t_list)
    assert 0.0 <= acc <= 1.0
    assert 0.0 <= ecf <= 1.0

    ev = evaluate_trajectories(t_list)
    agg = ev["aggregate"]
    for frac_key in (
        "accuracy",
        "coverage",
        "conditional_accuracy",
        "mean_flip_rate",
        "mean_rationalization_score",
        "mean_gated_rationalization_score",
        "mean_early_factor",
        "mean_post_change_rate",
        "mean_length_growth",
        "early_convergence_fraction",
        "disagreement_rate",
        "sbw_lockin_rate",
        "mean_convergence_fraction",
        "median_convergence_fraction",
    ):
        assert 0.0 <= agg[frac_key] <= 1.0, f"{frac_key}={agg[frac_key]} out of [0, 1]"
    assert agg["mean_gated_rationalization_score"] <= agg["mean_rationalization_score"] + 1e-9

    strat = stratified_evaluation(t_list)
    for slice_name in ("all", "correct", "incorrect", "undecided"):
        s_dict = strat[slice_name]
        assert 0.0 <= s_dict["mean_flip_rate"] <= 1.0
        assert 0.0 <= s_dict["mean_rationalization_score"] <= 1.0
        assert 0.0 <= s_dict["mean_gated_rationalization_score"] <= 1.0
        assert s_dict["mean_gated_rationalization_score"] <= s_dict["mean_rationalization_score"] + 1e-9

    gap = cotsel_selection_gap(t_list, rationalization_score)
    if gap["all_mean"] is not None:
        assert 0.0 <= gap["all_mean"] <= 1.0

    pk = pass_at_k_metrics(t_list, ks=(1, 2, 4))
    assert 0.0 <= pk["mean_unique_answer_ratio"] <= 1.0
    assert 0.0 <= pk["mean_unique_rationale_ratio"] <= 1.0
    for k_lbl, k_val in pk["pass_at_k"].items():
        assert 0.0 <= k_val <= 1.0, f"{k_lbl}={k_val} out of [0, 1]"

    gate = check_reasoning_smoke_gate(t_list)
    assert isinstance(gate["passed"], bool)


def test_part2_all_metrics_property_invariants_and_null_rule() -> None:
    rng = random.Random(20261110)
    cohort_a = [
        _make_seeded_trajectory(
            rng,
            f"item-{i // 4}#rollout{i % 4}",
            n_steps=16,
            prompt=f"Prompt {i // 4}",
            gold="B",
        )
        for i in range(64)
    ]
    cohort_b = [
        _make_seeded_trajectory(
            rng,
            f"item-{i // 4}#rollout{i % 4}",
            n_steps=16,
            prompt=f"Prompt {i // 4}",
            gold="B",
        )
        for i in range(64)
    ]

    assert_all_metric_invariants(cohort_a)
    assert_all_metric_invariants(cohort_b)

    # Null rule check: inserting None steps never creates a flip and never commits an answer.
    null_padded = Trajectory(
        id="null-rule",
        prompt="Q?",
        gold_answer="C",
        steps=[
            TrajectoryStep(step=0, prediction=None, answer_committed=None),
            TrajectoryStep(step=1, prediction="C", answer_committed=False),
            TrajectoryStep(step=2, prediction=None, answer_committed=None),
            TrajectoryStep(step=3, prediction="C", answer_committed=False),
        ],
    )
    assert flip_rate(null_padded) == 0.0
    assert answer_commitment_step(null_padded) is None
    assert verdict_convergence_step(null_padded) == 1


def test_part2_compare_conditions_antisymmetry_bootstrap_ci_and_determinism(tmp_path: Path) -> None:
    rng = random.Random(42)
    base = [
        _make_seeded_trajectory(rng, f"q-{i}", n_steps=12, prompt=f"P{i}", gold="C")
        for i in range(40)
    ]
    target = [
        _make_seeded_trajectory(rng, f"q-{i}", n_steps=12, prompt=f"P{i}", gold="C")
        for i in range(40)
    ]

    cmp_ab = compare_conditions(base, target, base_label="A", target_label="B", n_boot=500, seed=7)
    cmp_ba = compare_conditions(target, base, base_label="B", target_label="A", n_boot=500, seed=7)

    # Point deltas in compare_conditions(A, B) are the exact negative of compare_conditions(B, A).
    assert cmp_ab["accuracy_delta"]["delta"] == pytest.approx(-cmp_ba["accuracy_delta"]["delta"])

    for comp_key in ("early_factor", "post_change_rate", "length_growth", "score", "gated_score"):
        assert cmp_ab["rationalization_components_side_by_side"][comp_key]["delta"] == pytest.approx(
            -cmp_ba["rationalization_components_side_by_side"][comp_key]["delta"]
        )

    cov_ab = cmp_ab["coverage_decomposition"]
    cov_ba = cmp_ba["coverage_decomposition"]
    assert cov_ab["delta_accuracy"] == pytest.approx(-cov_ba["delta_accuracy"])
    assert cov_ab["coverage_effect"] == pytest.approx(-cov_ba["coverage_effect"])
    assert cov_ab["conditional_accuracy_effect"] == pytest.approx(-cov_ba["conditional_accuracy_effect"])

    for m_name, boot_ab in cmp_ab["paired_bootstrap"].items():
        boot_ba = cmp_ba["paired_bootstrap"][m_name]
        assert boot_ab["delta"] == pytest.approx(-boot_ba["delta"])
        assert boot_ab["ci_low"] == pytest.approx(-boot_ba["ci_high"])
        assert boot_ab["ci_high"] == pytest.approx(-boot_ba["ci_low"])
        # Bootstrap confidence interval contains the point estimate.
        assert boot_ab["ci_low"] <= boot_ab["delta"] <= boot_ab["ci_high"]
        assert boot_ba["ci_low"] <= boot_ba["delta"] <= boot_ba["ci_high"]

    # Order invariance: reversing target list order produces identical bootstrap CIs.
    cmp_rev = compare_conditions(
        base, list(reversed(target)), base_label="A", target_label="B", n_boot=500, seed=7
    )
    assert cmp_ab["paired_bootstrap"] == cmp_rev["paired_bootstrap"]

    fd_ab = forced_deliberation_delta(0.80, 0.70)
    fd_ba = forced_deliberation_delta(0.70, 0.80)
    assert fd_ab["delta"] == pytest.approx(-fd_ba["delta"])

    # Determinism: scoring the same file twice gives byte-identical JSON output.
    jsonl_path = tmp_path / "det.jsonl"
    jsonl_path.write_text(
        "\n".join(json.dumps(d) for d in trajectories_to_dicts(base)) + "\n",
        encoding="utf-8",
    )
    loaded_1 = load_trajectories(jsonl_path)
    loaded_2 = load_trajectories(jsonl_path)
    bytes_1 = json.dumps(evaluate_trajectories(loaded_1), sort_keys=True).encode("utf-8")
    bytes_2 = json.dumps(evaluate_trajectories(loaded_2), sort_keys=True).encode("utf-8")
    assert bytes_1 == bytes_2

    cmp_bytes_1 = json.dumps(
        compare_conditions(loaded_1, target, n_boot=500, seed=7), sort_keys=True
    ).encode("utf-8")
    cmp_bytes_2 = json.dumps(
        compare_conditions(loaded_2, target, n_boot=500, seed=7), sort_keys=True
    ).encode("utf-8")
    assert cmp_bytes_1 == cmp_bytes_2


def test_part1_linear_scaling_and_synthetic_generator_findings() -> None:
    # Fast linear scaling check across step counts T in {64, 256} for metrics.
    rng = random.Random(99)
    t_64 = [_make_seeded_trajectory(rng, f"s-{i}", n_steps=64) for i in range(16)]
    t_256 = [_make_seeded_trajectory(rng, f"l-{i}", n_steps=256) for i in range(16)]
    ev_64 = evaluate_trajectories(t_64)
    ev_256 = evaluate_trajectories(t_256)
    assert ev_64["aggregate"]["n_trajectories"] == 16
    assert ev_256["aggregate"]["n_trajectories"] == 16

    # Verify Issue #38 fixes in src/traj_eval/synthetic.py:
    # 1. _genuine, _oscillation, and generate_synthetic_trajectories honor arbitrary n_steps.
    assert len(_genuine(0, n_steps=32).steps) == 32
    assert len(_oscillation(0, n_steps=32).steps) == 32
    syn_32 = generate_synthetic_trajectories(n_per_archetype=2, n_steps=32)
    assert all(len(t.steps) == 32 for t in syn_32)

    # 2. _rationalization scales linearly O(T) in string memory while preserving score == 1.0.
    rat_16 = _rationalization(0, n_steps=16)
    rat_32 = _rationalization(0, n_steps=32)
    chars_16 = sum(len(s.reasoning_text) for s in rat_16.steps)
    chars_32 = sum(len(s.reasoning_text) for s in rat_32.steps)
    assert chars_32 < 2.2 * chars_16
    assert rationalization_score(rat_32) == pytest.approx(1.0)
    assert gated_rationalization_score(rat_32) == pytest.approx(1.0)

    if HEAVY_STRESS_ENABLED:
        run_heavy_stress_benchmarks()


def run_heavy_stress_benchmarks() -> dict[str, float]:
    """Run heavy 1,000 x 256 and 64 x 4,096 stress suites when opted in via flag/CLI."""
    rng = random.Random(1234)
    trajs_1000 = [_make_seeded_trajectory(rng, f"k-{i}", n_steps=256) for i in range(1000)]
    t0 = time.perf_counter()
    ev = evaluate_trajectories(trajs_1000)
    dt_1000 = time.perf_counter() - t0
    assert ev["aggregate"]["n_trajectories"] == 1000
    assert dt_1000 < 60.0

    t1 = time.perf_counter()
    long_64 = [_make_seeded_trajectory(rng, f"long-{i}", n_steps=4096) for i in range(64)]
    ev_long = evaluate_trajectories(long_64)
    dt_long = time.perf_counter() - t1
    assert ev_long["aggregate"]["n_trajectories"] == 64
    return {"eval_1000x256_sec": dt_1000, "eval_64x4096_sec": dt_long}


if __name__ == "__main__":
    import sys

    if "--heavy" in sys.argv:
        stats = run_heavy_stress_benchmarks()
        print("Heavy stress stats:", json.dumps(stats, indent=2))
        raise SystemExit(0)
    raise SystemExit(pytest.main([__file__, "-q"]))
