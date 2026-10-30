"""Trajectory-level evaluation for diffusion language models.

Measures reasoning *in* the denoising trajectory, not just final-answer accuracy.
"""

from __future__ import annotations

from traj_eval.metrics import (
    answer_commitment_step,
    early_convergence_fraction,
    evaluate_trajectories,
    flip_rate,
    forced_deliberation_delta,
    rationalization_components,
    rationalization_score,
    stability_commitment_lag,
    step_consistency,
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
    "flip_rate",
    "step_consistency",
    "forced_deliberation_delta",
    "rationalization_components",
    "rationalization_score",
    "trajectory_accuracy",
    "early_convergence_fraction",
    "evaluate_trajectories",
]
