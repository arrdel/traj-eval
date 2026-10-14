"""Core data types for trajectories."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TrajectoryStep:
    """One step in a denoising / reasoning trajectory.

    Attributes:
        step: Monotonic step index (0-based). Used to order steps.
        prediction: The model's current verdict / answer at this step.
            For multiple-choice, e.g. "A", "B", "C", "D". For free-form,
            a normalized answer string.
        reasoning_text: The model's current reasoning / rationale text at
            this step, if available. May be empty when a model only exposes
            a verdict.
        confidence: Optional confidence in [0, 1] for this step's prediction.
        logits: Optional raw logits for this step (kept for future adapters).
    """

    step: int
    prediction: str
    reasoning_text: str = ""
    confidence: float | None = None
    logits: list[float] | None = None


@dataclass
class Trajectory:
    """A full denoising trajectory for one prompt.

    Attributes:
        id: Unique identifier for this trajectory.
        prompt: The input prompt / question.
        gold_answer: Ground-truth answer (normalized, same format as
            ``prediction``). Empty string means gold is unknown; accuracy
            metrics will skip such trajectories for correctness but still
            compute trajectory-shape metrics.
        steps: Ordered list of steps. May be empty (edge case, handled
            gracefully by all metrics).
    """

    id: str
    prompt: str
    gold_answer: str
    steps: list[TrajectoryStep] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Keep steps sorted by step index so metrics are order-stable.
        self.steps = sorted(self.steps, key=lambda s: s.step)

    @property
    def final_prediction(self) -> str | None:
        """Prediction at the last step, or None if there are no steps."""
        if not self.steps:
            return None
        return self.steps[-1].prediction

    @property
    def is_correct(self) -> bool | None:
        """True/False if gold is known and steps exist, else None."""
        if not self.steps or self.gold_answer == "":
            return None
        return self.final_prediction == self.gold_answer
