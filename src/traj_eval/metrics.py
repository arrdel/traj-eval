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

import statistics
from typing import Any

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
    steps = traj.steps
    n = len(steps)
    if n == 0:
        return 0.0
    if n == 1:
        return 0.0
    conv_idx = _convergence_index(traj)
    if conv_idx is None:
        # No verdict at all (all null): nothing to rationalize.
        return 0.0
    early_factor = 1.0 - (conv_idx / n)
    post = steps[conv_idx + 1 :]
    if not post:
        # Converged at last step: no elaboration after verdict.
        return round(0.3 * early_factor, 4)
    # Post-convergence reasoning change rate.
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
    return round(float(max(0.0, min(1.0, score))), 4)


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
    rationalization score, and early-convergence fraction.
    """
    per: list[dict[str, Any]] = []
    conv_steps: list[int] = []
    conv_fracs: list[float] = []
    flip_rates: list[float] = []
    rat_scores: list[float] = []

    for t in trajectories:
        conv_step = verdict_convergence_step(t)
        idx = _convergence_index(t)
        fr = flip_rate(t)
        cons = step_consistency(t)
        rat = rationalization_score(t)
        per.append(
            {
                "id": t.id,
                "final_prediction": t.final_prediction,
                "gold_answer": t.gold_answer,
                "correct": t.is_correct,
                "convergence_step": conv_step,
                "convergence_index": idx,
                "convergence_fraction": (idx / len(t.steps)) if t.steps else None,
                "flip_rate": round(fr, 4),
                "consistency": round(cons["consistency"], 4),
                "longest_stable_run": cons["longest_stable_run"],
                "rationalization_score": rat,
                "n_steps": len(t.steps),
            }
        )
        if conv_step is not None and idx is not None and t.steps:
            conv_steps.append(conv_step)
            conv_fracs.append(idx / len(t.steps))
        flip_rates.append(fr)
        rat_scores.append(rat)

    def _mean(xs: list[float]) -> float:
        return float(statistics.mean(xs)) if xs else 0.0

    def _median(xs: list[float]) -> float:
        return float(statistics.median(xs)) if xs else 0.0

    aggregate = {
        "n_trajectories": len(trajectories),
        "n_scored_for_accuracy": len([t for t in trajectories if t.steps and t.gold_answer != ""]),
        "accuracy": round(trajectory_accuracy(trajectories), 4),
        "mean_flip_rate": round(_mean(flip_rates), 4),
        "mean_rationalization_score": round(_mean(rat_scores), 4),
        "early_convergence_fraction": round(early_convergence_fraction(trajectories), 4),
        "mean_convergence_step": round(_mean([float(x) for x in conv_steps]), 4) if conv_steps else 0.0,
        "median_convergence_step": round(_median([float(x) for x in conv_steps]), 4) if conv_steps else 0.0,
        "mean_convergence_fraction": round(_mean(conv_fracs), 4),
        "median_convergence_fraction": round(_median(conv_fracs), 4),
    }
    return {"per_trajectory": per, "aggregate": aggregate}
