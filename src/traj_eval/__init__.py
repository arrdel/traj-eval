"""Trajectory-level evaluation for diffusion language models.

Measures reasoning *in* the denoising trajectory, not just final-answer accuracy.
"""

from __future__ import annotations

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
from traj_eval.types import Trajectory, TrajectoryStep

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "Trajectory",
    "TrajectoryStep",
    "verdict_convergence_step",
    "answer_commitment_step",
    "stability_commitment_lag",
    "unwritten_reasoning_at_convergence",
    "order_gap",
    "three_clock_disagreement",
    "is_sbw_lockin",
    "flip_rate",
    "step_consistency",
    "forced_deliberation_delta",
    "rationalization_components",
    "rationalization_score",
    "gated_rationalization_score",
    "coverage_accuracy_decomposition",
    "cotsel_selection_gap",
    "stratified_evaluation",
    "paired_bootstrap_delta",
    "pass_at_k_metrics",
    "check_reasoning_smoke_gate",
    "compare_conditions",
    "trajectory_accuracy",
    "early_convergence_fraction",
    "evaluate_trajectories",
]
