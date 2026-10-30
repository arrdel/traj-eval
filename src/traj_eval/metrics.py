"""Trajectory-level metrics.

Core intuition
--------------
Final-answer accuracy collapses a whole denoising trajectory into one bit.
Two models can have the same accuracy for very different reasons:

* **Genuine reasoning:** the verdict changes as reasoning develops, then
  stabilizes. The reasoning *drives* the verdict.
* **Rationalization:** the verdict is fixed in the first few steps and the
  reasoning is elaborated afterwards to justify it. The verdict drives
  the reasoning.
* **Oscillation:** the verdict flips repeatedly and never settles, which
  suggests instability even if the final answer happens to be right.

These metrics make those patterns visible. They are descriptive, not
causal proof. In particular :func:`rationalization_score` is a
**heuristic** -- a high score means "looks like rationalization by this
operational definition", not "proved to be rationalizing".

All metrics handle edge cases gracefully:
* empty trajectory -> neutral values (None / 0.0) instead of crashing
* single-step trajectory -> convergence at that step, flip rate 0.0
"""

from __future__ import annotations

import math
import random
import statistics
from typing import Any, Callable, Sequence

from traj_eval.types import Trajectory


def _convergence_index(traj: Trajectory) -> int | None:
    """Index of the first step of the longest stable suffix.

    Null-aware (Dream adapter): ``None`` means "no verdict yet" and
    is not a distinct answer. Convergence is computed over the
    non-null verdict sequence, then mapped back to the original
    index. Returns None for an empty trajectory or for a trajectory
    with no non-null verdict at all. Otherwise always returns an
    index: at minimum, the last non-null step is trivially a stable
    suffix of length 1. This is the "first step after which prediction
    stays constant to the end" definition, ignoring no-verdict steps.
    """
    if not traj.steps:
        return None
    # (original_index, prediction) for non-null steps only.
    nn = [(i, s.prediction) for i, s in enumerate(traj.steps) if s.prediction is not None]
    if not nn:
        return None
    final_val = nn[-1][1]
    k = len(nn) - 1
    while k > 0 and nn[k - 1][1] == final_val:
        k -= 1
    return nn[k][0]


def verdict_convergence_step(traj: Trajectory) -> int | None:
    """First step after which the prediction stays constant to the end.

    Returns the ``step`` value (not the list index) of the convergence
    point, or None if the trajectory is empty.

    Example: predictions [A, B, C, C, C] with step numbers 0..4
    converge at step 2.
    """
    idx = _convergence_index(traj)
    if idx is None:
        return None
    return traj.steps[idx].step


def flip_rate(traj: Trajectory) -> float:
    """Fraction of adjacent non-null verdict pairs where prediction changes.

    Null-aware: ``None`` means "no verdict yet" and is skipped.
    A ``None -> "B"`` transition is NOT a flip. Flips are counted
    only between two non-null, differing verdicts. Denominator is
    ``len(non_null) - 1``.

    Range [0, 1]. 0.0 means the verdict never changed; 1.0 means it
    changed at every adjacent non-null pair. Empty, single-step, or
    all-null trajectories return 0.0.
    """
    nn = [s.prediction for s in traj.steps if s.prediction is not None]
    if len(nn) < 2:
        return 0.0
    flips = sum(1 for a, b in zip(nn, nn[1:]) if a != b)
    return flips / (len(nn) - 1)


def step_consistency(traj: Trajectory) -> dict[str, Any]:
    """Consistency summary: ``1 - flip_rate`` plus longest stable run.

    Longest stable run is computed over the non-null verdict
    sequence (nulls are skipped, not counted as breaks). Fraction
    uses total ``len(steps)`` as denominator, so null steps still
    count toward trajectory length, per the Dream adapter spec.

    Returns a dict with:
    * ``consistency``: 1 - flip_rate, in [0, 1].
    * ``longest_stable_run``: length of the longest run of identical
      consecutive non-null predictions (0 for empty / all-null).
    * ``longest_stable_run_fraction``: longest run / n_steps.
    """
    steps = traj.steps
    if not steps:
        return {
            "consistency": 0.0,
            "longest_stable_run": 0,
            "longest_stable_run_fraction": 0.0,
        }
    consistency = 1.0 - flip_rate(traj)
    nn = [s.prediction for s in steps if s.prediction is not None]
    if not nn:
        return {
            "consistency": consistency,
            "longest_stable_run": 0,
            "longest_stable_run_fraction": 0.0,
        }
    best = 1
    cur = 1
    for a, b in zip(nn, nn[1:]):
        if a == b:
            cur += 1
            best = max(best, cur)
        else:
            cur = 1
    return {
        "consistency": consistency,
        "longest_stable_run": best,
        "longest_stable_run_fraction": best / len(steps),
    }


def trajectory_accuracy(trajectories: list[Trajectory]) -> float:
    """Fraction of trajectories whose final prediction equals gold.

    Trajectories with no steps or unknown gold (``gold_answer == ""``)
    are excluded from the denominator. Returns 0.0 if none qualify.
    """
    scored = [t for t in trajectories if t.steps and t.gold_answer != ""]
    if not scored:
        return 0.0
    correct = sum(1 for t in scored if t.is_correct)
    return correct / len(scored)


def forced_deliberation_delta(
    base_accuracy: float, forced_accuracy: float
) -> dict[str, float]:
    """Compare accuracy with vs without forced extra deliberation.

    Args:
        base_accuracy: Accuracy under normal decoding.
        forced_accuracy: Accuracy when the model is forced to deliberate
            longer (e.g. extra denoising steps, or blocked from stopping).

    Returns:
        ``{"base_accuracy", "forced_accuracy", "delta",
        "relative_drop"}`` where ``delta = forced - base`` and
        ``relative_drop = (base - forced) / base`` (0.0 if base == 0).
        A negative delta / positive relative drop means forced
        deliberation *hurt* accuracy, which is a signal that the extra
        steps are not genuine reasoning.
    """
    delta = forced_accuracy - base_accuracy
    relative_drop = (base_accuracy - forced_accuracy) / base_accuracy if base_accuracy != 0 else 0.0
    return {
        "base_accuracy": float(base_accuracy),
        "forced_accuracy": float(forced_accuracy),
        "delta": float(delta),
        "relative_drop": float(relative_drop),
    }


def rationalization_components(traj: Trajectory) -> dict[str, float]:
    """Decompose :func:`rationalization_score` into its components + gated score.

    Heuristic, not ground truth. Returns a dict with keys:
    * ``early_factor``: ``1 - conv_idx / n`` in [0, 1].
    * ``post_change_rate``: fraction of post-convergence steps whose
      ``reasoning_text`` differs from the previous step, in [0, 1].
    * ``length_growth``: relative growth in ``reasoning_text`` length from
      convergence to the final step, clipped to [0, 1].
    * ``score``: the v0.1 additive heuristic score in [0, 1]
      (``0.4 * early_factor + 0.4 * post_change_rate + 0.2 * length_growth``).
    * ``gated_score``: multiplicative-gated heuristic score in [0, 1],
      ``early_factor * ((0.4 * post_change_rate + 0.2 * length_growth) / 0.6)``.
      Guarantees ``0.0`` when no post-convergence text change or growth occurs
      (e.g. direct-answer runs that pad remaining steps with EOS tokens).

    Exposing the components and ``gated_score`` prevents a direct-answer run
    with ``post_change_rate == 0.0`` from being misread as rationalization
    solely because ``early_factor`` is close to 1.0.
    """
    steps = traj.steps
    n = len(steps)
    if n <= 1:
        return {
            "early_factor": 0.0,
            "post_change_rate": 0.0,
            "length_growth": 0.0,
            "score": 0.0,
            "gated_score": 0.0,
        }
    conv_idx = _convergence_index(traj)
    if conv_idx is None:
        return {
            "early_factor": 0.0,
            "post_change_rate": 0.0,
            "length_growth": 0.0,
            "score": 0.0,
            "gated_score": 0.0,
        }
    early_factor = 1.0 - (conv_idx / n)
    post = steps[conv_idx + 1 :]
    if not post:
        return {
            "early_factor": round(float(early_factor), 4),
            "post_change_rate": 0.0,
            "length_growth": 0.0,
            "score": round(0.4 * early_factor, 4),
            "gated_score": 0.0,
        }
    changes = 0
    prev = steps[conv_idx].reasoning_text
    for s in post:
        if s.reasoning_text != prev:
            changes += 1
        prev = s.reasoning_text
    post_change_rate = changes / len(post)
    conv_len = len(steps[conv_idx].reasoning_text)
    last_len = len(steps[-1].reasoning_text)
    if conv_len == 0:
        length_growth = 1.0 if last_len > 0 else 0.0
    else:
        length_growth = max(0.0, (last_len - conv_len) / conv_len)
        length_growth = min(1.0, length_growth)
    score = 0.4 * early_factor + 0.4 * post_change_rate + 0.2 * length_growth
    elaboration_intensity = (0.4 * post_change_rate + 0.2 * length_growth) / 0.6
    gated_score = early_factor * elaboration_intensity
    return {
        "early_factor": round(float(early_factor), 4),
        "post_change_rate": round(float(post_change_rate), 4),
        "length_growth": round(float(length_growth), 4),
        "score": round(float(max(0.0, min(1.0, score))), 4),
        "gated_score": round(float(max(0.0, min(1.0, gated_score))), 4),
    }


def rationalization_score(traj: Trajectory) -> float:
    """Heuristic 0-1 score for "verdict first, reasoning afterwards".

    Heuristic, not ground truth. A high score means:
    * the verdict converges early (small fraction of steps), AND
    * reasoning keeps changing / lengthening after convergence.

    Formula:
    ``score = 0.4 * early_factor + 0.4 * post_change_rate + 0.2 * length_growth``

    where ``early_factor = 1 - conv_idx / n``, ``post_change_rate`` is
    the fraction of post-convergence steps whose reasoning text differs
    from the previous step, and ``length_growth`` is the relative growth
    in reasoning length from convergence to the end, clipped to [0, 1].

    Empty trajectories score 0.0. Trajectories that converge at the last
    step (no post-convergence reasoning) score low.
    """
    return rationalization_components(traj)["score"]


def gated_rationalization_score(traj: Trajectory) -> float:
    """Multiplicative-gated rationalization score in [0, 1].

    Equals ``early_factor * ((0.4 * post_change_rate + 0.2 * length_growth) / 0.6)``.
    Unlike the additive v0.1 :func:`rationalization_score`, this returns ``0.0``
    whenever no post-convergence rationale edits or length growth occur (e.g.
    when a model emits a short direct answer early and pads the remaining
    denoising canvas with EOS tokens).
    """
    return rationalization_components(traj)["gated_score"]


def answer_commitment_step(traj: Trajectory) -> int | None:
    """First step after which ``answer_committed`` stays True to the end.

    Descriptive commitment clock (``T_commit`` in Kim & Choi 2026;
    irreversible answer commitment in Yeom et al. 2026). Returns None if
    the trajectory is empty, if ``answer_committed`` was not logged, or
    if the final step's ``answer_committed`` is not True.
    """
    steps = traj.steps
    if not steps or steps[-1].answer_committed is not True:
        return None
    i = len(steps) - 1
    while i > 0 and steps[i - 1].answer_committed is True:
        i -= 1
    return steps[i].step


def stability_commitment_lag(traj: Trajectory) -> int | None:
    """Difference ``T_commit - T_stab`` in denoising steps.

    Descriptive metric comparing irreversible token commitment
    (:func:`answer_commitment_step`) against retrospective parsed-verdict
    stabilization (:func:`verdict_convergence_step`). Returns None if
    either clock is undefined.

    * ``> 0``: the parsed verdict stabilized before the final answer
      token/span became irreversibly committed.
    * ``== 0``: verdict stabilization and answer commitment coincided.
    * ``< 0``: the final answer token committed earlier than the last
      parsed-verdict change (e.g. a later token altered regex parsing).
    """
    t_stab = verdict_convergence_step(traj)
    t_comm = answer_commitment_step(traj)
    if t_stab is None or t_comm is None:
        return None
    return t_comm - t_stab


def unwritten_reasoning_at_convergence(traj: Trajectory) -> float | None:
    """Fraction of rationale-side positions still uncommitted at ``T_stab``.

    Descriptive metric matching Kim & Choi's unwritten fraction
    ``U = 1 - F_R(T_stab)`` in [0, 1]. Returns None if the trajectory has
    no convergence step or if ``committed_reasoning_fraction`` was not
    logged at the convergence step.
    """
    idx = _convergence_index(traj)
    if idx is None:
        return None
    crf = traj.steps[idx].committed_reasoning_fraction
    if crf is None:
        return None
    return round(float(max(0.0, min(1.0, 1.0 - crf))), 4)


def order_gap(traj: Trajectory) -> float | None:
    """Normalized commitment order gap ``(T_commit(ans) - T_median_commit(reason)) / n``.

    Matches Yeom et al. (2026, arXiv:2608.05687): negative values (``< 0``)
    indicate answer-first commitment (the answer commits before the median
    reasoning token). Returns None if commitment clocks were not logged or
    never reached median reasoning commitment.
    """
    steps = traj.steps
    n = len(steps)
    if n == 0 or steps[-1].answer_committed is not True:
        return None
    # Index where answer_committed enters its final True suffix.
    ans_idx = n - 1
    while ans_idx > 0 and steps[ans_idx - 1].answer_committed is True:
        ans_idx -= 1
    # First index where committed_reasoning_fraction >= 0.5.
    med_reas_idx: int | None = None
    for i, s in enumerate(steps):
        if s.committed_reasoning_fraction is not None and s.committed_reasoning_fraction >= 0.5:
            med_reas_idx = i
            break
    if med_reas_idx is None:
        return None
    return round(float((ans_idx - med_reas_idx) / n), 4)


def three_clock_disagreement(
    traj: Trajectory, unwritten_threshold: float = 0.0
) -> dict[str, Any]:
    """Identify whether a trajectory falls in the Three-Clock Disagreement Slice.

    Per ``docs/paper_outline.md`` (Section 3.2, Claim C2), the three clocks on
    a trajectory are:
    1. Parsed-verdict stabilization (``verdict_convergence_step`` / ``T_stab``).
    2. Irreversible answer-span commitment (``answer_commitment_step`` / ``T_commit``).
    3. Rationale-side completion and post-stability reasoning text change.

    Returns a dict with:
    * ``verdict_commit_mismatch``: ``bool | None`` (``stability_commitment_lag != 0``
      when commitment is logged, else ``None``).
    * ``rationale_unwritten_at_stab``: ``bool | None``
      (``unwritten_reasoning_at_convergence > unwritten_threshold`` when logged,
      else ``None``).
    * ``post_verdict_text_change``: ``bool`` (``post_change_rate > 0`` or
      ``length_growth > 0``).
    * ``in_disagreement_slice``: ``bool`` — True when the trajectory has a
      valid convergence step and at least one of the three disagreement
      conditions holds.
    """
    conv_idx = _convergence_index(traj)
    if conv_idx is None or not traj.steps:
        return {
            "verdict_commit_mismatch": None,
            "rationale_unwritten_at_stab": None,
            "post_verdict_text_change": False,
            "in_disagreement_slice": False,
        }
    lag = stability_commitment_lag(traj)
    unwritten = unwritten_reasoning_at_convergence(traj)
    comp = rationalization_components(traj)

    verdict_commit_mismatch: bool | None = (lag != 0) if lag is not None else None
    rationale_unwritten: bool | None = (
        (unwritten > unwritten_threshold) if unwritten is not None else None
    )
    post_text_change = bool(comp["post_change_rate"] > 0.0 or comp["length_growth"] > 0.0)
    in_slice = bool(
        (verdict_commit_mismatch is True)
        or (rationale_unwritten is True)
        or post_text_change
    )
    return {
        "verdict_commit_mismatch": verdict_commit_mismatch,
        "rationale_unwritten_at_stab": rationale_unwritten,
        "post_verdict_text_change": post_text_change,
        "in_disagreement_slice": in_slice,
    }


def is_sbw_lockin(traj: Trajectory, threshold: float = 0.2) -> bool | None:
    """Detect Stable-But-Wrong (SBW) early lock-in (Zhao et al., LOCKR 2026).

    Returns ``None`` when correctness is unknown (empty trajectory, unknown
    gold, or undecided final prediction). Otherwise returns ``True`` iff the
    verdict converges within the first ``threshold`` fraction of steps (default
    20%) and the final prediction is incorrect (``traj.is_correct is False``).
    """
    if not traj.steps or traj.is_correct is None:
        return None
    idx = _convergence_index(traj)
    if idx is None:
        return None
    conv_frac = idx / len(traj.steps)
    return bool(conv_frac <= threshold and (traj.is_correct is False))


def coverage_accuracy_decomposition(
    base_trajectories: Sequence[Trajectory],
    target_trajectories: Sequence[Trajectory],
) -> dict[str, float]:
    """Symmetric Two-Clocks decomposition of accuracy change into coverage vs. conditional accuracy.

    Matches Kim & Choi (2026, arXiv:2610.00953): for any set of trajectories
    with known gold, overall accuracy factors as
    ``accuracy = coverage * conditional_accuracy``, where
    ``coverage = P(final_prediction is not None)`` and
    ``conditional_accuracy = P(correct | final_prediction is not None)``.

    The exact symmetric decomposition of ``delta_accuracy = acc_2 - acc_1`` is:
    * ``coverage_effect = (c_2 - c_1) * (a_1 + a_2) / 2``
    * ``conditional_accuracy_effect = (a_2 - a_1) * (c_1 + c_2) / 2``
    so ``coverage_effect + conditional_accuracy_effect == delta_accuracy``.
    """
    def _stats(trajs: Sequence[Trajectory]) -> tuple[float, float, float]:
        with_gold = [t for t in trajs if t.steps and t.gold_answer != ""]
        if not with_gold:
            return 0.0, 0.0, 0.0
        decided = [t for t in with_gold if t.final_prediction is not None]
        correct = [t for t in decided if t.is_correct is True]
        cov = len(decided) / len(with_gold)
        cond_acc = (len(correct) / len(decided)) if decided else 0.0
        acc = len(correct) / len(with_gold)
        return cov, cond_acc, acc

    c1, a1, acc1 = _stats(base_trajectories)
    c2, a2, acc2 = _stats(target_trajectories)
    cov_eff = (c2 - c1) * (a1 + a2) / 2.0
    cond_eff = (a2 - a1) * (c1 + c2) / 2.0
    return {
        "base_coverage": round(float(c1), 4),
        "target_coverage": round(float(c2), 4),
        "base_conditional_accuracy": round(float(a1), 4),
        "target_conditional_accuracy": round(float(a2), 4),
        "base_accuracy": round(float(acc1), 4),
        "target_accuracy": round(float(acc2), 4),
        "delta_accuracy": round(float(acc2 - acc1), 4),
        "coverage_effect": round(float(cov_eff), 4),
        "conditional_accuracy_effect": round(float(cond_eff), 4),
    }


def cotsel_selection_gap(
    trajectories: Sequence[Trajectory],
    metric_fn: Callable[[Trajectory], float | None] = rationalization_score,
) -> dict[str, float | None]:
    """Compute CoTsel correctness-conditioning selection gap (Pan & Pan 2026).

    Implements the exact identity ``Delta = E[A | C=1] - E[A] = Cov(A, C) / P(C=1)``
    from arXiv:2610.37070 over trajectories with known gold (``t.steps`` non-empty
    and ``t.gold_answer != ""``), where ``C = 1[t.is_correct is True]`` and
    ``A = metric_fn(t)``.
    """
    pairs: list[tuple[float, int]] = []
    for t in trajectories:
        if not t.steps or t.gold_answer == "":
            continue
        val = metric_fn(t)
        if val is None:
            continue
        c = 1 if t.is_correct is True else 0
        pairs.append((float(val), c))
    if not pairs:
        return {
            "all_mean": None,
            "correct_mean": None,
            "incorrect_mean": None,
            "selection_gap": None,
        }
    all_vals = [a for a, _ in pairs]
    corr_vals = [a for a, c in pairs if c == 1]
    incorr_vals = [a for a, c in pairs if c == 0]
    all_mean = float(statistics.mean(all_vals))
    corr_mean = float(statistics.mean(corr_vals)) if corr_vals else None
    incorr_mean = float(statistics.mean(incorr_vals)) if incorr_vals else None
    gap = (corr_mean - all_mean) if corr_mean is not None else None
    return {
        "all_mean": round(all_mean, 4),
        "correct_mean": round(corr_mean, 4) if corr_mean is not None else None,
        "incorrect_mean": round(incorr_mean, 4) if incorr_mean is not None else None,
        "selection_gap": round(gap, 4) if gap is not None else None,
    }


def stratified_evaluation(trajectories: Sequence[Trajectory]) -> dict[str, Any]:
    """Stratify trajectory metrics across ``all``, ``correct``, ``incorrect``, and ``undecided`` slices.

    Prevents CoTsel correctness-selection bias (arXiv:2610.37070) and exposes
    how convergence, flip rate, rationalization, and disagreement differ
    between correct and erroneous trajectories.
    """
    def _slice_summary(subset: list[Trajectory]) -> dict[str, Any]:
        if not subset:
            return {
                "n": 0,
                "mean_flip_rate": 0.0,
                "mean_rationalization_score": 0.0,
                "mean_gated_rationalization_score": 0.0,
                "mean_convergence_fraction": 0.0,
                "disagreement_rate": 0.0,
            }
        frs = [flip_rate(t) for t in subset]
        rats = [rationalization_score(t) for t in subset]
        grats = [gated_rationalization_score(t) for t in subset]
        cfracs = [
            idx / len(t.steps)
            for t in subset
            if t.steps and (idx := _convergence_index(t)) is not None
        ]
        disag = [
            1.0 if three_clock_disagreement(t)["in_disagreement_slice"] else 0.0
            for t in subset
        ]
        return {
            "n": len(subset),
            "mean_flip_rate": round(float(statistics.mean(frs)), 4),
            "mean_rationalization_score": round(float(statistics.mean(rats)), 4),
            "mean_gated_rationalization_score": round(float(statistics.mean(grats)), 4),
            "mean_convergence_fraction": round(float(statistics.mean(cfracs)), 4) if cfracs else 0.0,
            "disagreement_rate": round(float(statistics.mean(disag)), 4),
        }

    all_list = list(trajectories)
    correct_list = [t for t in all_list if t.is_correct is True]
    incorrect_list = [t for t in all_list if t.is_correct is False]
    undecided_list = [t for t in all_list if t.steps and t.final_prediction is None]

    return {
        "all": _slice_summary(all_list),
        "correct": _slice_summary(correct_list),
        "incorrect": _slice_summary(incorrect_list),
        "undecided": _slice_summary(undecided_list),
        "cotsel_gaps": {
            "rationalization_score": cotsel_selection_gap(all_list, rationalization_score),
            "gated_rationalization_score": cotsel_selection_gap(all_list, gated_rationalization_score),
            "flip_rate": cotsel_selection_gap(all_list, flip_rate),
        },
    }


def paired_bootstrap_delta(
    base_trajectories: Sequence[Trajectory],
    target_trajectories: Sequence[Trajectory],
    *,
    metric: str = "accuracy",
    n_boot: int = 1000,
    ci: float = 0.90,
    seed: int = 42,
) -> dict[str, Any]:
    """Paired bootstrap confidence interval for ``target - base`` over matched trajectory IDs.

    Follows Yeom et al. (2026, arXiv:2608.05687) paired bootstrap protocol over
    questions. Trajectories are paired by ``t.id``. Supported ``metric`` names:
    ``"accuracy"``, ``"flip_rate"``, ``"rationalization_score"``,
    ``"gated_rationalization_score"``, ``"convergence_fraction"``.
    """
    def _per_item(t: Trajectory) -> float | None:
        if metric == "accuracy":
            if not t.steps or t.gold_answer == "":
                return None
            return 1.0 if t.is_correct is True else 0.0
        if metric == "flip_rate":
            return flip_rate(t)
        if metric == "rationalization_score":
            return rationalization_score(t)
        if metric == "gated_rationalization_score":
            return gated_rationalization_score(t)
        if metric == "convergence_fraction":
            if not t.steps:
                return None
            idx = _convergence_index(t)
            return (idx / len(t.steps)) if idx is not None else None
        raise ValueError(f"Unsupported metric for paired_bootstrap_delta: {metric!r}")

    base_map = {t.id: _per_item(t) for t in base_trajectories}
    target_map = {t.id: _per_item(t) for t in target_trajectories}
    shared_ids = [
        tid
        for tid in base_map
        if tid in target_map and base_map[tid] is not None and target_map[tid] is not None
    ]
    n_paired = len(shared_ids)
    if n_paired == 0:
        return {
            "metric": metric,
            "n_paired": 0,
            "base_mean": 0.0,
            "target_mean": 0.0,
            "delta": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "ci_level": ci,
        }
    base_vals = [float(base_map[tid]) for tid in shared_ids]  # type: ignore[arg-type]
    target_vals = [float(target_map[tid]) for tid in shared_ids]  # type: ignore[arg-type]
    diffs = [b - a for a, b in zip(base_vals, target_vals)]
    obs_base = float(statistics.mean(base_vals))
    obs_target = float(statistics.mean(target_vals))
    obs_delta = obs_target - obs_base

    rng = random.Random(seed)
    boot_deltas: list[float] = []
    for _ in range(max(1, n_boot)):
        sample_diffs = [diffs[rng.randrange(n_paired)] for _ in range(n_paired)]
        boot_deltas.append(sum(sample_diffs) / n_paired)
    boot_deltas.sort()
    alpha = (1.0 - ci) / 2.0
    lo_idx = min(len(boot_deltas) - 1, max(0, int(math.floor(alpha * len(boot_deltas)))))
    hi_idx = min(len(boot_deltas) - 1, max(0, int(math.ceil((1.0 - alpha) * len(boot_deltas)) - 1)))
    return {
        "metric": metric,
        "n_paired": n_paired,
        "base_mean": round(obs_base, 4),
        "target_mean": round(obs_target, 4),
        "delta": round(obs_delta, 4),
        "ci_low": round(float(boot_deltas[lo_idx]), 4),
        "ci_high": round(float(boot_deltas[hi_idx]), 4),
        "ci_level": ci,
    }


def _unbiased_pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator ``1 - comb(n - c, k) / comb(n, k)`` (Chen et al., 2021)."""
    if n < k or k <= 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - (math.comb(n - c, k) / math.comb(n, k))


def pass_at_k_metrics(
    trajectories: Sequence[Trajectory],
    ks: Sequence[int] = (1, 2, 4, 8),
) -> dict[str, Any]:
    """Compute Pass@k scaling and diversity metrics across multiple rollouts per prompt.

    Diagnoses the Flexibility Trap (Ni et al. 2026, arXiv:2601.15165;
    Fork-dLLM 2026, arXiv:2609.39859) where confidence-based diffusion decoding
    collapses solution diversity. Trajectories are grouped by ``t.prompt`` (or
    by base ID prefix before ``"::"`` / ``"#rollout"`` when prompts are identical).
    """
    groups: dict[str, list[Trajectory]] = {}
    for t in trajectories:
        if not t.steps:
            continue
        base_id = t.id.split("::")[0].split("#")[0]
        key = t.prompt if t.prompt else base_id
        groups.setdefault(key, []).append(t)

    if not groups:
        return {
            "n_prompts": 0,
            "mean_rollouts_per_prompt": 0.0,
            "pass_at_k": {},
            "mean_unique_answer_ratio": 0.0,
            "mean_unique_rationale_ratio": 0.0,
        }

    pass_k_out: dict[str, float] = {}
    for k in ks:
        scores_k: list[float] = []
        for g in groups.values():
            scored = [t for t in g if t.gold_answer != ""]
            if len(scored) < k:
                continue
            c = sum(1 for t in scored if t.is_correct is True)
            scores_k.append(_unbiased_pass_at_k(len(scored), c, k))
        if scores_k:
            pass_k_out[f"pass@{k}"] = round(float(statistics.mean(scores_k)), 4)

    uniq_ans_ratios: list[float] = []
    uniq_rat_ratios: list[float] = []
    for g in groups.values():
        non_null_preds = [t.final_prediction for t in g if t.final_prediction is not None]
        if non_null_preds:
            uniq_ans_ratios.append(len(set(non_null_preds)) / len(non_null_preds))
        final_texts = [t.steps[-1].reasoning_text.strip() for t in g if t.steps]
        if final_texts:
            uniq_rat_ratios.append(len(set(final_texts)) / len(final_texts))

    return {
        "n_prompts": len(groups),
        "mean_rollouts_per_prompt": round(
            float(statistics.mean([len(g) for g in groups.values()])), 4
        ),
        "pass_at_k": pass_k_out,
        "mean_unique_answer_ratio": round(
            float(statistics.mean(uniq_ans_ratios)), 4
        ) if uniq_ans_ratios else 0.0,
        "mean_unique_rationale_ratio": round(
            float(statistics.mean(uniq_rat_ratios)), 4
        ) if uniq_rat_ratios else 0.0,
    }


def early_convergence_fraction(
    trajectories: list[Trajectory], threshold: float = 0.2
) -> float:
    """Fraction of trajectories converging within the first ``threshold`` of steps.

    Convergence fraction is ``conv_idx / n``. A trajectory converges
    "early" if that fraction is <= threshold (default 20%). Empty
    trajectories are excluded. Returns 0.0 if none qualify.
    """
    fracs: list[float] = []
    for t in trajectories:
        if not t.steps:
            continue
        idx = _convergence_index(t)
        if idx is None:
            continue
        fracs.append(idx / len(t.steps))
    if not fracs:
        return 0.0
    early = sum(1 for f in fracs if f <= threshold)
    return early / len(fracs)


def evaluate_trajectories(trajectories: list[Trajectory]) -> dict[str, Any]:
    """Evaluate a list of trajectories: per-trajectory + aggregate.

    Returns a dict with ``per_trajectory`` (list) and ``aggregate``
    (summary dict). Aggregate includes accuracy, mean/median
    convergence step and fraction, mean flip rate, mean
    rationalization score (plus its three components), early-convergence
    fraction, and optional commitment-clock summaries when logged.
    """
    per: list[dict[str, Any]] = []
    conv_steps: list[int] = []
    conv_fracs: list[float] = []
    flip_rates: list[float] = []
    rat_scores: list[float] = []
    gated_rat_scores: list[float] = []
    early_factors: list[float] = []
    post_change_rates: list[float] = []
    length_growths: list[float] = []
    comm_steps: list[int] = []
    comm_lags: list[int] = []
    unwritten_fracs: list[float] = []
    order_gaps: list[float] = []
    disagreement_flags: list[float] = []
    sbw_flags: list[float] = []

    for t in trajectories:
        conv_step = verdict_convergence_step(t)
        idx = _convergence_index(t)
        fr = flip_rate(t)
        cons = step_consistency(t)
        rat_comp = rationalization_components(t)
        rat = rat_comp["score"]
        gated_rat = rat_comp["gated_score"]
        t_comm = answer_commitment_step(t)
        lag = stability_commitment_lag(t)
        unwritten = unwritten_reasoning_at_convergence(t)
        ogap = order_gap(t)
        disag = three_clock_disagreement(t)
        sbw = is_sbw_lockin(t)
        per.append(
            {
                "id": t.id,
                "final_prediction": t.final_prediction,
                "gold_answer": t.gold_answer,
                "correct": t.is_correct,
                "convergence_step": conv_step,
                "convergence_index": idx,
                "convergence_fraction": (idx / len(t.steps)) if (t.steps and idx is not None) else None,
                "flip_rate": round(fr, 4),
                "consistency": round(cons["consistency"], 4),
                "longest_stable_run": cons["longest_stable_run"],
                "rationalization_score": rat,
                "gated_rationalization_score": gated_rat,
                "early_factor": rat_comp["early_factor"],
                "post_change_rate": rat_comp["post_change_rate"],
                "length_growth": rat_comp["length_growth"],
                "answer_commitment_step": t_comm,
                "stability_commitment_lag": lag,
                "unwritten_reasoning_at_convergence": unwritten,
                "order_gap": ogap,
                "in_disagreement_slice": disag["in_disagreement_slice"],
                "sbw_lockin": sbw,
                "n_steps": len(t.steps),
            }
        )
        if conv_step is not None and idx is not None and t.steps:
            conv_steps.append(conv_step)
            conv_fracs.append(idx / len(t.steps))
        flip_rates.append(fr)
        rat_scores.append(rat)
        gated_rat_scores.append(gated_rat)
        early_factors.append(rat_comp["early_factor"])
        post_change_rates.append(rat_comp["post_change_rate"])
        length_growths.append(rat_comp["length_growth"])
        disagreement_flags.append(1.0 if disag["in_disagreement_slice"] else 0.0)
        if sbw is not None:
            sbw_flags.append(1.0 if sbw else 0.0)
        if t_comm is not None:
            comm_steps.append(t_comm)
        if lag is not None:
            comm_lags.append(lag)
        if unwritten is not None:
            unwritten_fracs.append(unwritten)
        if ogap is not None:
            order_gaps.append(ogap)

    def _mean(xs: list[float]) -> float:
        return float(statistics.mean(xs)) if xs else 0.0

    def _median(xs: list[float]) -> float:
        return float(statistics.median(xs)) if xs else 0.0

    with_steps = [t for t in trajectories if t.steps]
    decided = [t for t in with_steps if t.final_prediction is not None]
    coverage = (len(decided) / len(with_steps)) if with_steps else 0.0
    decided_with_gold = [t for t in decided if t.gold_answer != ""]
    cond_acc = (
        sum(1 for t in decided_with_gold if t.is_correct is True) / len(decided_with_gold)
        if decided_with_gold
        else 0.0
    )

    aggregate: dict[str, Any] = {
        "n_trajectories": len(trajectories),
        "n_scored_for_accuracy": len([t for t in trajectories if t.steps and t.gold_answer != "" and t.is_correct is not None]),
        "accuracy": round(trajectory_accuracy(trajectories), 4),
        "coverage": round(float(coverage), 4),
        "conditional_accuracy": round(float(cond_acc), 4),
        "mean_flip_rate": round(_mean(flip_rates), 4),
        "mean_rationalization_score": round(_mean(rat_scores), 4),
        "mean_gated_rationalization_score": round(_mean(gated_rat_scores), 4),
        "mean_early_factor": round(_mean(early_factors), 4),
        "mean_post_change_rate": round(_mean(post_change_rates), 4),
        "mean_length_growth": round(_mean(length_growths), 4),
        "early_convergence_fraction": round(early_convergence_fraction(trajectories), 4),
        "disagreement_rate": round(_mean(disagreement_flags), 4),
        "sbw_lockin_rate": round(_mean(sbw_flags), 4) if sbw_flags else 0.0,
        "mean_convergence_step": round(_mean([float(x) for x in conv_steps]), 4) if conv_steps else 0.0,
        "median_convergence_step": round(_median([float(x) for x in conv_steps]), 4) if conv_steps else 0.0,
        "mean_convergence_fraction": round(_mean(conv_fracs), 4),
        "median_convergence_fraction": round(_median(conv_fracs), 4),
        "mean_answer_commitment_step": round(_mean([float(x) for x in comm_steps]), 4) if comm_steps else None,
        "mean_stability_commitment_lag": round(_mean([float(x) for x in comm_lags]), 4) if comm_lags else None,
        "mean_unwritten_reasoning_at_convergence": round(_mean(unwritten_fracs), 4) if unwritten_fracs else None,
        "mean_order_gap": round(_mean(order_gaps), 4) if order_gaps else None,
    }
    return {"per_trajectory": per, "aggregate": aggregate}


def check_reasoning_smoke_gate(trajectories: Sequence[Trajectory]) -> dict[str, Any]:
    """Validate the reasoning-condition smoke gate from ``docs/CURSOR_FOLLOWUP_2026-10-29.md`` §4.

    Gate requirements:
    1. Commitment fields (``T_commit``, ``stability_commitment_lag``, and
       ``unwritten_reasoning_at_convergence``) must be non-null on decided
       trajectories.
    2. At least some trajectories must contain real post-verdict or
       pre-verdict reasoning text (``post_change_rate > 0`` on some items, or
       non-trivial rationale text beyond a bare ``"Answer: <letter>"`` stub).
    """
    trajs = list(trajectories)
    if not trajs:
        return {
            "passed": False,
            "status": "reasoning smoke failed: empty trajectory list",
            "n_trajectories": 0,
            "n_decided": 0,
            "n_with_commitment_clocks": 0,
            "n_with_post_change": 0,
            "n_with_pre_verdict_rationale": 0,
            "commitment_clocks_populated": False,
            "has_reasoning_activity": False,
        }

    decided = [t for t in trajs if t.steps and t.final_prediction is not None]
    n_with_comm = sum(
        1
        for t in decided
        if (
            answer_commitment_step(t) is not None
            and stability_commitment_lag(t) is not None
            and unwritten_reasoning_at_convergence(t) is not None
        )
    )
    n_with_post_change = sum(
        1 for t in decided if rationalization_components(t)["post_change_rate"] > 0.0
    )
    n_with_pre_rationale = 0
    for t in decided:
        idx = _convergence_index(t)
        if idx is not None and t.steps:
            text_at_conv = t.steps[idx].reasoning_text.strip()
            # More than a bare "Answer: X" stub (<= 15 chars).
            if len(text_at_conv) > 15:
                n_with_pre_rationale += 1

    comm_ok = len(decided) > 0 and n_with_comm == len(decided)
    reasoning_ok = (n_with_post_change > 0) or (n_with_pre_rationale > 0)
    reasons: list[str] = []
    if not comm_ok:
        reasons.append(
            f"commitment clocks populated on {n_with_comm}/{len(decided)} decided trajectories"
        )
    if not reasoning_ok:
        reasons.append(
            "traces collapsed to bare answer stub without post-verdict or pre-verdict rationale"
        )
    passed = bool(comm_ok and reasoning_ok)
    status = (
        "reasoning smoke clean"
        if passed
        else f"reasoning smoke failed: {'; '.join(reasons)}"
    )
    return {
        "passed": passed,
        "status": status,
        "n_trajectories": len(trajs),
        "n_decided": len(decided),
        "n_with_commitment_clocks": n_with_comm,
        "n_with_post_change": n_with_post_change,
        "n_with_pre_verdict_rationale": n_with_pre_rationale,
        "commitment_clocks_populated": comm_ok,
        "has_reasoning_activity": reasoning_ok,
    }


def compare_conditions(
    base_trajectories: Sequence[Trajectory],
    target_trajectories: Sequence[Trajectory],
    *,
    base_label: str = "base",
    target_label: str = "target",
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compare two trajectory conditions side by side (e.g. direct vs. reasoning, or base vs. forced).

    Reports:
    * Full ``evaluate_trajectories`` aggregate for each condition.
    * Side-by-side rationalization components (``early_factor``,
      ``post_change_rate``, ``length_growth``, ``score``, ``gated_score``) so
      the scalar is never reported alone (``docs/CURSOR_FOLLOWUP_2026-10-29.md`` §4).
    * :func:`forced_deliberation_delta` and :func:`coverage_accuracy_decomposition`.
    * :func:`paired_bootstrap_delta` across accuracy, flip rate,
      rationalization score, gated rationalization score, and convergence
      fraction.
    """
    base_list = list(base_trajectories)
    target_list = list(target_trajectories)
    base_eval = evaluate_trajectories(base_list)
    target_eval = evaluate_trajectories(target_list)
    b_agg = base_eval["aggregate"]
    t_agg = target_eval["aggregate"]

    side_by_side_components = {
        "early_factor": {
            base_label: b_agg["mean_early_factor"],
            target_label: t_agg["mean_early_factor"],
            "delta": round(t_agg["mean_early_factor"] - b_agg["mean_early_factor"], 4),
        },
        "post_change_rate": {
            base_label: b_agg["mean_post_change_rate"],
            target_label: t_agg["mean_post_change_rate"],
            "delta": round(t_agg["mean_post_change_rate"] - b_agg["mean_post_change_rate"], 4),
        },
        "length_growth": {
            base_label: b_agg["mean_length_growth"],
            target_label: t_agg["mean_length_growth"],
            "delta": round(t_agg["mean_length_growth"] - b_agg["mean_length_growth"], 4),
        },
        "score": {
            base_label: b_agg["mean_rationalization_score"],
            target_label: t_agg["mean_rationalization_score"],
            "delta": round(
                t_agg["mean_rationalization_score"] - b_agg["mean_rationalization_score"], 4
            ),
        },
        "gated_score": {
            base_label: b_agg["mean_gated_rationalization_score"],
            target_label: t_agg["mean_gated_rationalization_score"],
            "delta": round(
                t_agg["mean_gated_rationalization_score"]
                - b_agg["mean_gated_rationalization_score"],
                4,
            ),
        },
    }

    bootstrap = {
        m: paired_bootstrap_delta(
            base_list, target_list, metric=m, n_boot=n_boot, seed=seed
        )
        for m in (
            "accuracy",
            "flip_rate",
            "rationalization_score",
            "gated_rationalization_score",
            "convergence_fraction",
        )
    }

    return {
        "base_label": base_label,
        "target_label": target_label,
        "base_aggregate": b_agg,
        "target_aggregate": t_agg,
        "rationalization_components_side_by_side": side_by_side_components,
        "accuracy_delta": forced_deliberation_delta(
            b_agg["accuracy"], t_agg["accuracy"]
        ),
        "coverage_decomposition": coverage_accuracy_decomposition(
            base_list, target_list
        ),
        "paired_bootstrap": bootstrap,
    }
