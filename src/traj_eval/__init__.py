"""Trajectory-level evaluation for diffusion language models.

Measures reasoning *in* the denoising trajectory, not just final-answer accuracy.
"""

from __future__ import annotations

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
from traj_eval.types import Trajectory, TrajectoryStep

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "Trajectory",
    "TrajectoryStep",
    "verdict_convergence_step",
    "flip_rate",
    "step_consistency",
    "forced_deliberation_delta",
    "rationalization_score",
    "trajectory_accuracy",
    "early_convergence_fraction",
    "evaluate_trajectories",
]
