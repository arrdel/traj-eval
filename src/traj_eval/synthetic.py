"""Synthetic trajectories for demos, tests, and quickstart.

Three archetypes:

* **genuine** -- verdict changes gradually as reasoning develops, then
  stabilizes. Reasoning also stabilizes after convergence.
* **rationalization** -- verdict is fixed at step 0-1 and reasoning is
  elaborated afterwards to justify it.
* **oscillation** -- verdict flips repeatedly and never settles.

These are hand-crafted, deterministic, and deliberately simple. They are
not a substitute for real model trajectories, but they make the metrics
interpretable and give the demo / tests something to run without any
model weights.
"""

from __future__ import annotations

import random

from traj_eval.types import Trajectory, TrajectoryStep


def _genuine(idx: int, n_steps: int = 8) -> Trajectory:
    if n_steps <= 8:
        # Verdict path: A -> A -> B -> B -> C -> C -> C -> C (converges ~62%).
        preds = ["A", "A", "B", "B", "C", "C", "C", "C"][:n_steps]
        reasoning = [
            "Initial guess, little evidence.",
            "Initial guess, considering alternatives.",
            "Reconsidering: evidence points to B.",
            "Evidence for B is getting stronger.",
            "Actually C fits the evidence best.",
            "C fits the evidence best.",
            "C fits the evidence best.",
            "C fits the evidence best.",
        ][:n_steps]
        steps = [
            TrajectoryStep(step=i, prediction=p, reasoning_text=r)
            for i, (p, r) in enumerate(zip(preds, reasoning))
        ]
    else:
        q1 = max(1, n_steps // 4)
        q2 = max(q1 + 1, n_steps // 2)
        steps = []
        for i in range(n_steps):
            if i < q1:
                pred = "A"
                text = f"Step {i}: Initial guess, considering alternatives."
            elif i < q2:
                pred = "B"
                text = f"Step {i}: Reconsidering, evidence points to B."
            elif i == q2:
                pred = "C"
                text = "Actually C fits the evidence best."
            else:
                pred = "C"
                text = "C fits the evidence best."
            steps.append(TrajectoryStep(step=i, prediction=pred, reasoning_text=text))
    return Trajectory(
        id=f"genuine-{idx}",
        prompt=f"Synthetic genuine-reasoning question {idx}",
        gold_answer="C",
        steps=steps,
    )


def _rationalization(idx: int, n_steps: int = 6) -> Trajectory:
    # Verdict fixed from step 0, reasoning keeps growing.
    base = "The answer is B."
    steps = []
    if n_steps <= 6:
        for i in range(n_steps):
            extra = " " + " ".join(f"Supporting detail {j+1}." for j in range(i))
            steps.append(
                TrajectoryStep(
                    step=i,
                    prediction="B",
                    reasoning_text=base + extra,
                )
            )
    else:
        # O(T) space and time for arbitrary n_steps: step 0 has base text, and each
        # subsequent step updates its rationale text with bounded O(1) length >= 2 * len(base).
        for i in range(n_steps):
            if i == 0:
                text = base + " "
            else:
                text = f"{base} Supporting detail {i} of {n_steps}: elaborating post-hoc justification."
            steps.append(
                TrajectoryStep(
                    step=i,
                    prediction="B",
                    reasoning_text=text,
                )
            )
    return Trajectory(
        id=f"rationalization-{idx}",
        prompt=f"Synthetic rationalization question {idx}",
        gold_answer="B",
        steps=steps,
    )


def _oscillation(idx: int, n_steps: int = 8) -> Trajectory:
    steps = [
        TrajectoryStep(
            step=i,
            prediction=("A" if i % 2 == 0 else "B"),
            reasoning_text=f"Step {i}: leaning {'A' if i % 2 == 0 else 'B'}, uncertain.",
        )
        for i in range(n_steps)
    ]
    return Trajectory(
        id=f"oscillation-{idx}",
        prompt=f"Synthetic oscillation question {idx}",
        gold_answer="B",
        steps=steps,
    )


def generate_synthetic_trajectories(
    n_per_archetype: int = 5,
    seed: int = 42,
    n_steps: int | None = None,
) -> list[Trajectory]:
    """Generate deterministic synthetic trajectories.

    Args:
        n_per_archetype: How many of each archetype to generate.
        seed: Kept for API stability; generation is currently
            deterministic and does not depend on randomness beyond
            shuffling the final order with this seed.
        n_steps: Optional step count per trajectory. When ``None``, uses
            the v0.1 defaults (8 steps for genuine/oscillation, 6 for
            rationalization).
    """
    out: list[Trajectory] = []
    for i in range(n_per_archetype):
        if n_steps is None:
            out.append(_genuine(i))
            out.append(_rationalization(i))
            out.append(_oscillation(i))
        else:
            out.append(_genuine(i, n_steps=n_steps))
            out.append(_rationalization(i, n_steps=n_steps))
            out.append(_oscillation(i, n_steps=n_steps))
    rng = random.Random(seed)
    rng.shuffle(out)
    return out
