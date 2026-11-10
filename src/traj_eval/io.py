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
import math
from pathlib import Path
from typing import Any

from traj_eval.types import Trajectory, TrajectoryStep


def _reject_nonfinite_constant(val: str) -> Any:
    raise ValueError(f"Non-finite JSON constant {val!r} is not allowed")


def _parse_step(raw: Any, traj_id: str, idx: int) -> TrajectoryStep:
    if not isinstance(raw, dict):
        raise ValueError(f"Trajectory {traj_id!r}: step {idx} is not an object: {raw!r}")
    if "step" not in raw:
        raise ValueError(f"Trajectory {traj_id!r}: step {idx} missing required 'step'")
    if isinstance(raw["step"], bool):
        raise ValueError(
            f"Trajectory {traj_id!r}: step {idx} has boolean 'step': {raw['step']!r}"
        )
    try:
        step_no = int(raw["step"])
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"Trajectory {traj_id!r}: step {idx} has non-integer 'step': {raw['step']!r}"
        ) from exc
    if step_no < 0:
        raise ValueError(
            f"Trajectory {traj_id!r}: step {idx} has negative 'step': {step_no}"
        )

    # prediction may be None (no verdict yet, Dream adapter) or a string.
    # Missing prediction is treated as None for adapter compatibility.
    raw_pred = raw.get("prediction", None)
    if raw_pred is not None:
        if not isinstance(raw_pred, str):
            raise ValueError(
                f"Trajectory {traj_id!r}: step {idx} 'prediction' must be a string or null, "
                f"got {type(raw_pred).__name__}"
            )

    raw_ans_comm = raw.get("answer_committed", None)
    if raw_ans_comm is not None and not isinstance(raw_ans_comm, bool):
        raise ValueError(
            f"Trajectory {traj_id!r}: step {idx} 'answer_committed' must be a boolean or null, "
            f"got {raw_ans_comm!r}"
        )
    ans_comm: bool | None = raw_ans_comm

    raw_crf = raw.get("committed_reasoning_fraction", None)
    crf: float | None = None
    if raw_crf is not None:
        if isinstance(raw_crf, bool):
            raise ValueError(
                f"Trajectory {traj_id!r}: step {idx} 'committed_reasoning_fraction' must be a number, "
                f"got {raw_crf!r}"
            )
        try:
            crf = float(raw_crf)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"Trajectory {traj_id!r}: step {idx} has invalid 'committed_reasoning_fraction': {raw_crf!r}"
            ) from exc
        if not math.isfinite(crf) or not (0.0 <= crf <= 1.0):
            raise ValueError(
                f"Trajectory {traj_id!r}: step {idx} 'committed_reasoning_fraction' must lie in [0, 1], "
                f"got {crf!r}"
            )

    raw_conf = raw.get("confidence", None)
    conf: float | None = None
    if raw_conf is not None:
        if isinstance(raw_conf, bool):
            raise ValueError(
                f"Trajectory {traj_id!r}: step {idx} 'confidence' must be a number, got {raw_conf!r}"
            )
        try:
            conf = float(raw_conf)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"Trajectory {traj_id!r}: step {idx} has invalid 'confidence': {raw_conf!r}"
            ) from exc
        if not math.isfinite(conf):
            raise ValueError(
                f"Trajectory {traj_id!r}: step {idx} has non-finite 'confidence': {conf!r}"
            )

    return TrajectoryStep(
        step=step_no,
        prediction=raw_pred,
        reasoning_text=str(raw.get("reasoning_text", "") or ""),
        confidence=conf,
        logits=raw.get("logits"),
        answer_committed=ans_comm,
        committed_reasoning_fraction=crf,
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
    if len({s.step for s in steps}) != len(steps):
        raise ValueError(f"Trajectory {raw['id']!r}: duplicate step numbers are not allowed")
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
        ValueError: If the path is a directory or the file is empty or malformed.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {p}")
    if p.is_dir():
        raise ValueError(f"Input path is a directory, expected a file: {p}")
    text = p.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Input file is empty: {p}")

    # Try JSON first if it looks like a JSON list / object.
    if p.suffix.lower() == ".json" or text.startswith("["):
        try:
            data = json.loads(text, parse_constant=_reject_nonfinite_constant)
        except (json.JSONDecodeError, ValueError) as exc:
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
            raw = json.loads(line, parse_constant=_reject_nonfinite_constant)
        except (json.JSONDecodeError, ValueError) as exc:
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
                        "answer_committed": s.answer_committed,
                        "committed_reasoning_fraction": s.committed_reasoning_fraction,
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
