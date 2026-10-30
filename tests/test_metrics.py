"""Tests for trajectory metrics."""

import pytest

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


def test_rationalization_components_distinguish_direct_from_elaborating():
    # Direct answer: verdict settles early and reasoning text never changes after.
    direct = _traj(
        ["C"] * 4,
        gold="C",
        reasoning=["Answer: C"] * 4,
    )
    comp_dir = rationalization_components(direct)
    assert comp_dir["early_factor"] == pytest.approx(1.0)
    assert comp_dir["post_change_rate"] == 0.0
    assert comp_dir["length_growth"] == 0.0
    assert comp_dir["score"] == pytest.approx(0.4)

    # Elaborating after verdict: post_change_rate and length_growth are high.
    elab = _traj(
        ["C"] * 4,
        gold="C",
        reasoning=["C", "C because x", "C because x and y", "C because x and y and z"],
    )
    comp_elab = rationalization_components(elab)
    assert comp_elab["post_change_rate"] == pytest.approx(1.0)
    assert comp_elab["length_growth"] == pytest.approx(1.0)
    assert comp_elab["score"] == pytest.approx(1.0)


def test_commitment_clocks_and_unwritten_fraction():
    # Answer stabilizes at step 1 (T_stab=1) and commits at step 2 (T_commit=2).
    steps = [
        TrajectoryStep(
            step=0,
            prediction=None,
            reasoning_text="",
            answer_committed=False,
            committed_reasoning_fraction=0.0,
        ),
        TrajectoryStep(
            step=1,
            prediction="B",
            reasoning_text="Answer: B",
            answer_committed=False,
            committed_reasoning_fraction=0.25,
        ),
        TrajectoryStep(
            step=2,
            prediction="B",
            reasoning_text="Because of X, Answer: B",
            answer_committed=True,
            committed_reasoning_fraction=0.5,
        ),
        TrajectoryStep(
            step=3,
            prediction="B",
            reasoning_text="Because of X and Y, Answer: B",
            answer_committed=True,
            committed_reasoning_fraction=1.0,
        ),
    ]
    t = Trajectory(id="c1", prompt="q", gold_answer="B", steps=steps)
    assert verdict_convergence_step(t) == 1
    assert answer_commitment_step(t) == 2
    assert stability_commitment_lag(t) == 1
    # At T_stab (step 1), 25% of reasoning is committed -> 75% unwritten.
    assert unwritten_reasoning_at_convergence(t) == pytest.approx(0.75)

    out = evaluate_trajectories([t])
    assert out["aggregate"]["mean_answer_commitment_step"] == pytest.approx(2.0)
    assert out["aggregate"]["mean_stability_commitment_lag"] == pytest.approx(1.0)
    assert out["aggregate"]["mean_unwritten_reasoning_at_convergence"] == pytest.approx(0.75)
    # Answer commits at index 2; median reasoning (>=0.5) commits at index 2 -> order_gap = 0.0
    assert order_gap(t) == pytest.approx(0.0)
    assert three_clock_disagreement(t)["in_disagreement_slice"] is True


def test_gated_rationalization_zero_on_eos_padded_direct_answer():
    # Direct answer at step 0 with 255 EOS-padded identical steps:
    # additive v0.1 score is ~0.40, whereas gated_score is exact 0.0.
    direct = _traj(["C"] * 10, gold="C", reasoning=["Answer: C"] * 10)
    comp = rationalization_components(direct)
    assert comp["score"] == pytest.approx(0.4)
    assert comp["gated_score"] == 0.0
    assert gated_rationalization_score(direct) == 0.0

    # Full early rationalization with continuous post-verdict elaboration hits 1.0 on both.
    elab = _traj(
        ["C"] * 4,
        gold="C",
        reasoning=["C", "C because x", "C because x and y", "C because x and y and z"],
    )
    assert gated_rationalization_score(elab) == pytest.approx(1.0)


def test_rationalization_converges_at_last_step_uses_0_4_weight():
    # Converges at step 3 of 4 (conv_idx=3, n=4): early_factor = 0.25, no post steps.
    # Must use 0.4 * early_factor = 0.10 (consistent with post_change_rate=0, length_growth=0).
    t = _traj(["A", "B", "A", "C"], gold="C")
    comp = rationalization_components(t)
    assert comp["early_factor"] == pytest.approx(0.25)
    assert comp["post_change_rate"] == 0.0
    assert comp["length_growth"] == 0.0
    assert comp["score"] == pytest.approx(0.1)
    assert comp["gated_score"] == 0.0


def test_order_gap_answer_first_negative():
    # Answer commits at step 1, while median reasoning (>=0.5) commits at step 3 (4 steps total).
    steps = [
        TrajectoryStep(step=0, prediction=None, answer_committed=False, committed_reasoning_fraction=0.0),
        TrajectoryStep(step=1, prediction="B", answer_committed=True, committed_reasoning_fraction=0.2),
        TrajectoryStep(step=2, prediction="B", answer_committed=True, committed_reasoning_fraction=0.4),
        TrajectoryStep(step=3, prediction="B", answer_committed=True, committed_reasoning_fraction=1.0),
    ]
    t = Trajectory(id="af", prompt="q", gold_answer="B", steps=steps)
    # (1 - 3) / 4 = -0.5 < 0 (answer-first)
    assert order_gap(t) == pytest.approx(-0.5)


def test_sbw_lockin_separates_benign_from_wrong_early_convergence():
    # Converges at step 0 of 10 (conv_frac = 0.0 <= 0.2), wrong answer -> SBW lock-in True.
    wrong_early = _traj(["A"] * 10, gold="B", tid="sbw")
    # Converges at step 0 of 10, right answer -> Benign lock-in (SBW False).
    right_early = _traj(["B"] * 10, gold="B", tid="benign")
    # Converges at step 8 of 10 (conv_frac = 0.8 > 0.2), wrong answer -> Not early lock-in.
    wrong_late = _traj(["B"] * 8 + ["A", "A"], gold="B", tid="late")

    assert is_sbw_lockin(wrong_early) is True
    assert is_sbw_lockin(right_early) is False
    assert is_sbw_lockin(wrong_late) is False


def test_coverage_accuracy_decomposition_exact_identity():
    # Base: 4 items, 2 decided (both correct) -> coverage=0.5, cond_acc=1.0, acc=0.5.
    base = [
        _traj(["B", "B"], gold="B", tid="1"),
        _traj(["C", "C"], gold="C", tid="2"),
        _traj([None, None], gold="A", tid="3"),
        _traj([None, None], gold="D", tid="4"),
    ]
    # Target: 4 items, 4 decided (3 correct) -> coverage=1.0, cond_acc=0.75, acc=0.75.
    target = [
        _traj(["B", "B"], gold="B", tid="1"),
        _traj(["C", "C"], gold="C", tid="2"),
        _traj(["A", "A"], gold="A", tid="3"),
        _traj(["A", "A"], gold="D", tid="4"),
    ]
    decomp = coverage_accuracy_decomposition(base, target)
    assert decomp["base_coverage"] == pytest.approx(0.5)
    assert decomp["target_coverage"] == pytest.approx(1.0)
    assert decomp["base_conditional_accuracy"] == pytest.approx(1.0)
    assert decomp["target_conditional_accuracy"] == pytest.approx(0.75)
    assert decomp["delta_accuracy"] == pytest.approx(0.25)
    # Exact identity: coverage_effect + conditional_accuracy_effect == delta_accuracy.
    assert decomp["coverage_effect"] + decomp["conditional_accuracy_effect"] == pytest.approx(
        decomp["delta_accuracy"]
    )
    assert decomp["coverage_effect"] > abs(decomp["conditional_accuracy_effect"])


def test_cotsel_selection_gap_and_stratified_evaluation():
    # Correct trajectory has high gated rationalization (1.0), incorrect has 0.0.
    t_corr = _traj(
        ["B"] * 4,
        gold="B",
        reasoning=["B", "B x", "B x y", "B x y z"],
        tid="c",
    )
    t_incorr = _traj(
        ["A"] * 4,
        gold="B",
        reasoning=["Answer: A"] * 4,
        tid="i",
    )
    gap = cotsel_selection_gap([t_corr, t_incorr], gated_rationalization_score)
    assert gap["all_mean"] == pytest.approx(0.5)
    assert gap["correct_mean"] == pytest.approx(1.0)
    assert gap["incorrect_mean"] == pytest.approx(0.0)
    assert gap["selection_gap"] == pytest.approx(0.5)

    strat = stratified_evaluation([t_corr, t_incorr])
    assert strat["correct"]["n"] == 1
    assert strat["incorrect"]["n"] == 1
    assert strat["cotsel_gaps"]["gated_rationalization_score"]["selection_gap"] == pytest.approx(0.5)


def test_paired_bootstrap_delta_and_pass_at_k():
    base = [
        _traj(["A"], gold="B", tid="q1"),
        _traj(["B"], gold="B", tid="q2"),
        _traj(["A"], gold="B", tid="q3"),
        _traj(["B"], gold="B", tid="q4"),
    ]
    target = [
        _traj(["B"], gold="B", tid="q1"),
        _traj(["B"], gold="B", tid="q2"),
        _traj(["B"], gold="B", tid="q3"),
        _traj(["B"], gold="B", tid="q4"),
    ]
    boot = paired_bootstrap_delta(base, target, metric="accuracy", n_boot=200, seed=0)
    assert boot["n_paired"] == 4
    assert boot["delta"] == pytest.approx(0.5)
    assert boot["ci_low"] <= boot["delta"] <= boot["ci_high"]

    # Pass@k across 2 prompts, 4 rollouts each.
    rollouts = [
        Trajectory(id="p1#0", prompt="P1", gold_answer="B", steps=[TrajectoryStep(step=0, prediction="B", reasoning_text="r1")]),
        Trajectory(id="p1#1", prompt="P1", gold_answer="B", steps=[TrajectoryStep(step=0, prediction="A", reasoning_text="r2")]),
        Trajectory(id="p1#2", prompt="P1", gold_answer="B", steps=[TrajectoryStep(step=0, prediction="C", reasoning_text="r3")]),
        Trajectory(id="p1#3", prompt="P1", gold_answer="B", steps=[TrajectoryStep(step=0, prediction="D", reasoning_text="r4")]),
    ]
    pk = pass_at_k_metrics(rollouts, ks=(1, 2, 4))
    assert pk["n_prompts"] == 1
    assert pk["pass_at_k"]["pass@1"] == pytest.approx(0.25)
    assert pk["pass_at_k"]["pass@4"] == pytest.approx(1.0)
    assert pk["mean_unique_answer_ratio"] == pytest.approx(1.0)


def test_reasoning_smoke_gate_and_compare_conditions(tmp_path):
    # Degenerate direct-answer trace without commitment clocks or reasoning text fails gate.
    direct = _traj(["C"] * 4, gold="C", reasoning=["Answer: C"] * 4, tid="q1")
    gate_fail = check_reasoning_smoke_gate([direct])
    assert gate_fail["passed"] is False
    assert gate_fail["status"].startswith("reasoning smoke failed:")

    # Reasoning trajectory with commitment clocks populated and post-verdict text changes passes gate.
    reas = Trajectory(
        id="q1",
        prompt="Q?",
        gold_answer="C",
        steps=[
            TrajectoryStep(
                step=0,
                prediction=None,
                reasoning_text="",
                answer_committed=False,
                committed_reasoning_fraction=0.0,
            ),
            TrajectoryStep(
                step=1,
                prediction="C",
                reasoning_text="Answer: C",
                answer_committed=True,
                committed_reasoning_fraction=0.25,
            ),
            TrajectoryStep(
                step=2,
                prediction="C",
                reasoning_text="Faster rotation shortens the day. Answer: C",
                answer_committed=True,
                committed_reasoning_fraction=0.75,
            ),
            TrajectoryStep(
                step=3,
                prediction="C",
                reasoning_text="Faster rotation shortens the solar day. Answer: C",
                answer_committed=True,
                committed_reasoning_fraction=1.0,
            ),
        ],
    )
    gate_pass = check_reasoning_smoke_gate([reas])
    assert gate_pass["passed"] is True
    assert gate_pass["status"] == "reasoning smoke clean"

    comp = compare_conditions([direct], [reas], base_label="direct", target_label="reasoning")
    sbs = comp["rationalization_components_side_by_side"]
    assert sbs["post_change_rate"]["direct"] == 0.0
    assert sbs["post_change_rate"]["reasoning"] > 0.0
    assert sbs["gated_score"]["direct"] == 0.0
    assert sbs["gated_score"]["reasoning"] > 0.0

    # Test CLI compare subcommand
    import json
    from traj_eval.cli import main
    from traj_eval.io import trajectories_to_dicts

    p_dir = tmp_path / "direct.jsonl"
    p_reas = tmp_path / "reas.jsonl"
    p_out = tmp_path / "cmp.json"
    p_dir.write_text(json.dumps(trajectories_to_dicts([direct])[0]) + "\n")
    p_reas.write_text(json.dumps(trajectories_to_dicts([reas])[0]) + "\n")
    assert main(["compare", "--base", str(p_dir), "--target", str(p_reas), "--output", str(p_out)]) == 0
    assert p_out.exists()
