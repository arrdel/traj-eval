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
from traj_eval.adapters.llada import (
    LLADA_15_ID,
    LLADA_EOS_TOKEN_ID,
    LLADA_EOT_TOKEN_ID,
    LLADA_INSTRUCT_ID,
    LLADA_MASK_TOKEN_ID,
    LLaDASmokeConfig,
    compute_transfer_schedule,
    llada_diffusion_generate,
    llada_history_to_trajectory,
    run_llada_smoke,
)

__all__ = [
    "DREAM_MODEL_ID",
    "DEFAULT_SMOKE_CONFIG",
    "DreamSmokeConfig",
    "LLADA_INSTRUCT_ID",
    "LLADA_15_ID",
    "LLADA_MASK_TOKEN_ID",
    "LLADA_EOS_TOKEN_ID",
    "LLADA_EOT_TOKEN_ID",
    "LLaDASmokeConfig",
    "compute_step_commitments",
    "compute_transfer_schedule",
    "decode_generation_region",
    "extract_gsm8k_verdict",
    "extract_verdict",
    "history_to_trajectory",
    "llada_diffusion_generate",
    "llada_history_to_trajectory",
    "normalize_numeric_answer",
    "run_llada_smoke",
    "strip_gsm8k_exemplars",
]
