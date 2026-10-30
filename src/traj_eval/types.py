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
            a normalized answer string. ``None`` means no verdict yet
            (e.g. Dream adapter: generation region still all masks, no
            decodable letter). Metrics treat ``None`` as "no verdict",
            not as a distinct answer: flips are counted only between
            two non-null verdicts, and convergence is computed over
            the non-null suffix.
        reasoning_text: The model's current reasoning / rationale text at
            this step, if available. May be empty when a model only exposes
            a verdict.
        confidence: Optional confidence in [0, 1] for this step's prediction.
        logits: Optional raw logits for this step (kept for future adapters).
        answer_committed: Optional flag indicating whether the final answer
            token/span has irreversibly committed by this step (left the mask
            state and never changed again).
        committed_reasoning_fraction: Optional fraction in [0, 1] of
            rationale/prefix positions before the answer that have
            irreversibly committed by this step.
    """

    step: int
    prediction: str | None
    reasoning_text: str = ""
    confidence: float | None = None
    logits: list[float] | None = None
    answer_committed: bool | None = None
    committed_reasoning_fraction: float | None = None


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
        """Prediction at the last step, or None if there are no steps.

        Note: this may also be None if the last step itself has no
        verdict (e.g. Dream adapter all-mask final step). Callers
        should treat None as "no verdict".
        """
        if not self.steps:
            return None
        return self.steps[-1].prediction

    @property
    def is_correct(self) -> bool | None:
        """True/False if gold is known and steps exist, else None.

        Returns None when gold is unknown, when there are no steps,
        or when the final step has no verdict (final_prediction is None).
        """
        if not self.steps or self.gold_answer == "":
            return None
        final = self.final_prediction
        if final is None:
            return None
        return final == self.gold_answer
