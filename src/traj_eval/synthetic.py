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
    # Verdict path: A -> A -> B -> B -> C -> C -> C -> C (converges ~62%).
    preds = ["A", "A", "B", "B", "C", "C", "C", "C"][:n_steps]
    # Reasoning develops, then freezes after convergence.
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
    for i in range(n_steps):
        extra = " " + " ".join(f"Supporting detail {j+1}." for j in range(i))
        steps.append(
            TrajectoryStep(
                step=i,
                prediction="B",
                reasoning_text=base + extra,
            )
        )
    return Trajectory(
        id=f"rationalization-{idx}",
        prompt=f"Synthetic rationalization question {idx}",
        gold_answer="B",
        steps=steps,
    )


def _oscillation(idx: int, n_steps: int = 8) -> Trajectory:
    preds = ["A", "B", "A", "B", "A", "B", "A", "B"][:n_steps]
    steps = [
        TrajectoryStep(
            step=i,
            prediction=p,
            reasoning_text=f"Step {i}: leaning {p}, uncertain.",
        )
        for i, p in enumerate(preds)
    ]
    return Trajectory(
        id=f"oscillation-{idx}",
        prompt=f"Synthetic oscillation question {idx}",
        gold_answer="B",
        steps=steps,
    )


def generate_synthetic_trajectories(
    n_per_archetype: int = 5, seed: int = 42
) -> list[Trajectory]:
    """Generate deterministic synthetic trajectories.

    Args:
        n_per_archetype: How many of each archetype to generate.
        seed: Kept for API stability; generation is currently
            deterministic and does not depend on randomness beyond
            shuffling the final order with this seed.
    """
    out: list[Trajectory] = []
    for i in range(n_per_archetype):
        out.append(_genuine(i))
        out.append(_rationalization(i))
        out.append(_oscillation(i))
    rng = random.Random(seed)
    rng.shuffle(out)
    return out
