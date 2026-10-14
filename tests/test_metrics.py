"""Tests for trajectory metrics."""

import pytest

from traj_eval.metrics import (
    early_convergence_fraction,
    evaluate_trajectories,
    flip_rate,
    forced_deliberation_delta,
    rationalization_score,
    step_consistency,
    trajectory_accuracy,
    verdict_convergence_step,
)
from traj_eval.synthetic import generate_synthetic_trajectories
from traj_eval.types import Trajectory, TrajectoryStep


def _traj(preds, gold="B", reasoning=None, tid="t1"):
    reasoning = reasoning or [f"reasoning {i}" for i in range(len(preds))]
    steps = [
        TrajectoryStep(step=i, prediction=p, reasoning_text=r)
        for i, (p, r) in enumerate(zip(preds, reasoning))
    ]
    return Trajectory(id=tid, prompt="q", gold_answer=gold, steps=steps)


def test_empty_trajectory():
    t = Trajectory(id="e", prompt="q", gold_answer="B", steps=[])
    assert verdict_convergence_step(t) is None
    assert flip_rate(t) == 0.0
    assert rationalization_score(t) == 0.0
    assert step_consistency(t)["longest_stable_run"] == 0


def test_single_step():
    t = _traj(["A"])
    assert verdict_convergence_step(t) == 0
    assert flip_rate(t) == 0.0
    assert step_consistency(t)["longest_stable_run"] == 1


def test_never_changes():
    t = _traj(["B", "B", "B", "B"])
    assert verdict_convergence_step(t) == 0
    assert flip_rate(t) == 0.0
    assert step_consistency(t)["consistency"] == 1.0


def test_all_flips():
    t = _traj(["A", "B", "A", "B"])
    assert flip_rate(t) == 1.0
    assert step_consistency(t)["consistency"] == 0.0
    assert step_consistency(t)["longest_stable_run"] == 1


def test_convergence_mid_trajectory():
    t = _traj(["A", "B", "C", "C", "C"])
    assert verdict_convergence_step(t) == 2
    assert flip_rate(t) == pytest.approx(2 / 4)


def test_longest_stable_run():
    t = _traj(["A", "A", "B", "B", "B", "A"])
    cons = step_consistency(t)
    assert cons["longest_stable_run"] == 3
    assert cons["longest_stable_run_fraction"] == pytest.approx(3 / 6)


def test_rationalization_high_vs_low():
    # Rationalization: verdict fixed early, reasoning keeps changing/growing.
    rat = _traj(
        ["B"] * 6,
        reasoning=["a", "a b", "a b c", "a b c d", "a b c d e", "a b c d e f"],
    )
    # Genuine: verdict settles late, reasoning freezes after.
    gen = _traj(
        ["A", "A", "B", "B", "C", "C", "C", "C"],
        gold="C",
        reasoning=["r1", "r2", "r3", "r4", "r5", "r5", "r5", "r5"],
    )
    assert rationalization_score(rat) > rationalization_score(gen)
    assert rationalization_score(rat) > 0.5
    assert rationalization_score(gen) < 0.5


def test_rationalization_is_heuristic_range():
    t = _traj(["A", "B", "B"])
    score = rationalization_score(t)
    assert 0.0 <= score <= 1.0


def test_trajectory_accuracy():
    trajs = [_traj(["A", "B"], gold="B"), _traj(["A", "A"], gold="B"), _traj(["C"], gold="C")]
    assert trajectory_accuracy(trajs) == pytest.approx(2 / 3)


def test_accuracy_ignores_empty_and_unknown_gold():
    trajs = [
        Trajectory(id="e", prompt="q", gold_answer="B", steps=[]),
        Trajectory(id="u", prompt="q", gold_answer="", steps=[TrajectoryStep(step=0, prediction="A")]),
        _traj(["B"], gold="B"),
    ]
    assert trajectory_accuracy(trajs) == 1.0
    assert trajectory_accuracy([]) == 0.0


def test_forced_deliberation_delta():
    d = forced_deliberation_delta(0.862, 0.719)
    assert d["delta"] == pytest.approx(-0.143, abs=1e-3)
    assert d["relative_drop"] == pytest.approx((0.862 - 0.719) / 0.862)
    # No division by zero.
    d0 = forced_deliberation_delta(0.0, 0.0)
    assert d0["relative_drop"] == 0.0


def test_early_convergence_fraction():
    early = _traj(["B"] * 5, tid="early")
    late = _traj(["A", "B", "C", "D", "D"], tid="late")
    assert early_convergence_fraction([early, late]) == pytest.approx(0.5)
    assert early_convergence_fraction([]) == 0.0


def test_evaluate_trajectories_aggregate():
    trajs = generate_synthetic_trajectories(n_per_archetype=2)
    out = evaluate_trajectories(trajs)
    assert out["aggregate"]["n_trajectories"] == 6
    assert len(out["per_trajectory"]) == 6
    assert 0.0 <= out["aggregate"]["accuracy"] <= 1.0
    assert "mean_convergence_step" in out["aggregate"]
    assert "median_convergence_step" in out["aggregate"]


def test_evaluate_empty_list():
    out = evaluate_trajectories([])
    assert out["aggregate"]["n_trajectories"] == 0
    assert out["aggregate"]["accuracy"] == 0.0


def test_cli_smoke_demo(capsys):
    from traj_eval.cli import main

    assert main(["demo", "--n-per-archetype", "1"]) == 0
    captured = capsys.readouterr()
    assert "traj-eval summary" in captured.out
    assert "Accuracy" in captured.out


def test_synthetic_archetypes_differ():
    trajs = generate_synthetic_trajectories(n_per_archetype=3)
    by_prefix: dict[str, list] = {}
    for t in trajs:
        prefix = t.id.split("-")[0]
        by_prefix.setdefault(prefix, []).append(t)
    assert set(by_prefix) == {"genuine", "rationalization", "oscillation"}
    # Oscillation should have the highest flip rate on average.
    from traj_eval.metrics import flip_rate as fr

    means = {k: sum(fr(t) for t in v) / len(v) for k, v in by_prefix.items()}
    assert means["oscillation"] > means["genuine"]
    assert means["oscillation"] > means["rationalization"]
