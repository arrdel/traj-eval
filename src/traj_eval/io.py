"""Loading and saving trajectories and results.

File formats
------------
JSONL (preferred, one trajectory per line)::

    {"id": "q1", "prompt": "...", "gold_answer": "B",
     "steps": [{"step": 0, "prediction": "A", "reasoning_text": "..."}, ...]}

JSON (a list of the same objects, or {"trajectories": [...]}) is also
accepted for convenience.

All loaders raise ``ValueError`` with a clear message on malformed input
instead of silently dropping data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from traj_eval.types import Trajectory, TrajectoryStep


def _parse_step(raw: Any, traj_id: str, idx: int) -> TrajectoryStep:
    if not isinstance(raw, dict):
        raise ValueError(f"Trajectory {traj_id!r}: step {idx} is not an object: {raw!r}")
    if "prediction" not in raw:
        raise ValueError(f"Trajectory {traj_id!r}: step {idx} missing required 'prediction'")
    if "step" not in raw:
        raise ValueError(f"Trajectory {traj_id!r}: step {idx} missing required 'step'")
    try:
        step_no = int(raw["step"])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Trajectory {traj_id!r}: step {idx} has non-integer 'step': {raw['step']!r}"
        ) from exc
    return TrajectoryStep(
        step=step_no,
        prediction=str(raw["prediction"]),
        reasoning_text=str(raw.get("reasoning_text", "") or ""),
        confidence=raw.get("confidence"),
        logits=raw.get("logits"),
    )


def _parse_trajectory(raw: Any, idx: int) -> Trajectory:
    if not isinstance(raw, dict):
        raise ValueError(f"Entry {idx} is not an object: {raw!r}")
    for key in ("id", "prompt", "gold_answer", "steps"):
        if key not in raw:
            raise ValueError(f"Entry {idx} missing required field {key!r}: {raw!r}")
    if not isinstance(raw["steps"], list):
        raise ValueError(f"Trajectory {raw['id']!r}: 'steps' must be a list")
    steps = [_parse_step(s, str(raw["id"]), i) for i, s in enumerate(raw["steps"])]
    return Trajectory(
        id=str(raw["id"]),
        prompt=str(raw["prompt"]),
        gold_answer=str(raw["gold_answer"]),
        steps=steps,
    )


def load_trajectories(path: str | Path) -> list[Trajectory]:
    """Load trajectories from a JSONL or JSON file.

    Args:
        path: Path to ``.jsonl`` / ``.json`` file. JSONL is detected by
            extension ``.jsonl`` or by content starting with ``{`` on the
            first non-empty line when the file is not valid JSON.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is empty or malformed.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {p}")
    text = p.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Input file is empty: {p}")

    # Try JSON first if it looks like a JSON list / object.
    if p.suffix.lower() == ".json" or text.startswith("["):
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {p}: {exc}") from exc
        if isinstance(data, dict) and "trajectories" in data:
            data = data["trajectories"]
        if not isinstance(data, list):
            raise ValueError(f"JSON file must contain a list or {{'trajectories': [...]}}: {p}")
        return [_parse_trajectory(item, i) for i, item in enumerate(data)]

    # Otherwise treat as JSONL.
    trajectories: list[Trajectory] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{p}:{lineno}: invalid JSON: {exc}") from exc
        trajectories.append(_parse_trajectory(raw, lineno - 1))
    if not trajectories:
        raise ValueError(f"No trajectories found in {p}")
    return trajectories


def trajectories_to_dicts(trajectories: list[Trajectory]) -> list[dict[str, Any]]:
    """Serialize trajectories back to plain dicts (round-trip safe)."""
    out: list[dict[str, Any]] = []
    for t in trajectories:
        out.append(
            {
                "id": t.id,
                "prompt": t.prompt,
                "gold_answer": t.gold_answer,
                "steps": [
                    {
                        "step": s.step,
                        "prediction": s.prediction,
                        "reasoning_text": s.reasoning_text,
                        "confidence": s.confidence,
                        "logits": s.logits,
                    }
                    for s in t.steps
                ],
            }
        )
    return out


def save_results(results: dict[str, Any], path: str | Path) -> Path:
    """Save an evaluation result dict as pretty JSON. Returns the path."""
    p = Path(path)
    if p.parent != Path(".") and str(p.parent) != "":
        p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return p
