"""Model adapters: turn real model generations into standard Trajectories.

Adapters are intentionally light at import time. They do NOT import
torch or transformers at module import, so tests and ``traj_eval demo``
work without GPU deps. Heavy imports happen inside the ``run_*``
functions that actually load a model.

Current adapter:
- Dream (Dream-org/Dream-v0-Instruct-7B) for ARC-C, see ``dream.py``.
"""

from __future__ import annotations

from traj_eval.adapters.dream import (
    DREAM_MODEL_ID,
    DEFAULT_SMOKE_CONFIG,
    DreamSmokeConfig,
    compute_step_commitments,
    decode_generation_region,
    extract_gsm8k_verdict,
    extract_verdict,
    history_to_trajectory,
    normalize_numeric_answer,
    strip_gsm8k_exemplars,
)

__all__ = [
    "DREAM_MODEL_ID",
    "DEFAULT_SMOKE_CONFIG",
    "DreamSmokeConfig",
    "compute_step_commitments",
    "decode_generation_region",
    "extract_gsm8k_verdict",
    "extract_verdict",
    "history_to_trajectory",
    "normalize_numeric_answer",
    "strip_gsm8k_exemplars",
]
