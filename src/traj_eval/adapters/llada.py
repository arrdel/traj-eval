# Copyright (C) 2026 Lenovo
# All Rights Reserved.
# Lenovo Confidential Restricted.

"""LLaDA adapter for traj-eval (LLaDA-8B-Instruct and LLaDA-1.5).

Supported checkpoints:
- ``GSAI-ML/LLaDA-8B-Instruct``
- ``GSAI-ML/LLaDA-1.5``

Why LLaDA needs its own adapter loop
------------------------------------
Unlike Dream (which ships ``model.diffusion_generate(output_history=True)``
in ``generation_utils.py``), LLaDA's ``LLaDAModelLM`` only exposes the
forward pass ``model(x, attention_mask=...).logits``, and its tokenizer
sets ``tokenizer.mask_token_id = None`` while using ``mask_id = 126336``
(EOS = ``126081``, EoT = ``126348``).

This module implements LLaDA's released blockwise low-confidence
remasking sampler (matching ``ML-GSAI/LLaDA/generate.py`` and Yeom et al.
2026's EOS/EoT confidence suppression) and records ``x.clone()`` after
every denoising step so :func:`history_to_trajectory` produces the exact
same ``Trajectory`` schema as the Dream adapter.

No torch or transformers imports occur at module import time. Tests use
lightweight stub models and tokenizers without weights or GPU.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from traj_eval.adapters.dream import (
    TokenizerLike,
    _as_int_list,
    extract_gsm8k_verdict,
    extract_verdict,
    history_to_trajectory,
)
from traj_eval.types import Trajectory

LLADA_INSTRUCT_ID = "GSAI-ML/LLaDA-8B-Instruct"
LLADA_15_ID = "GSAI-ML/LLaDA-1.5"
LLADA_MASK_TOKEN_ID = 126336
LLADA_EOS_TOKEN_ID = 126081
LLADA_EOT_TOKEN_ID = 126348


def compute_transfer_schedule(block_length: int, steps_per_block: int) -> list[int]:
    """Compute per-step token transfer counts for one block of length ``block_length``.

    Mirrors ``get_num_transfer_tokens`` in ``ML-GSAI/LLaDA/generate.py``:
    distributes ``block_length`` tokens evenly across ``steps_per_block``
    steps, placing the remainder on the earliest steps so the sum equals
    ``block_length``.
    """
    if steps_per_block <= 0:
        raise ValueError(f"steps_per_block must be > 0, got {steps_per_block}")
    if block_length < 0:
        raise ValueError(f"block_length must be >= 0, got {block_length}")
    base = block_length // steps_per_block
    rem = block_length % steps_per_block
    return [base + (1 if i < rem else 0) for i in range(steps_per_block)]


@dataclass
class LLaDASmokeConfig:
    """Sampler + run config for LLaDA-8B-Instruct and LLaDA-1.5."""

    model_id: str = LLADA_INSTRUCT_ID
    task: str = "arcc"
    gen_length: int = 256
    steps: int = 256
    block_length: int = 256
    temperature: float = 0.0
    cfg_scale: float = 0.0
    remasking: str = "low_confidence"
    mask_id: int = LLADA_MASK_TOKEN_ID
    logits_eos_inf: bool = False
    confidence_eos_eot_inf: bool = True
    # When set (e.g. w=16, Yeom et al. 2026), restricts eligible commitments at
    # step t to positions i < f(t) + frontier_width where f(t) is the left-most
    # masked position in the generation canvas.
    frontier_width: int | None = None
    n_items: int = 5
    out_jsonl: str = "results/llada_arcc_smoke5.jsonl"
    out_ids: str = "results/llada_arcc_smoke5_ids.json"
    out_log: str = "results/llada_arcc_smoke5_run_log.json"
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "task": self.task,
            "gen_length": self.gen_length,
            "steps": self.steps,
            "block_length": self.block_length,
            "temperature": self.temperature,
            "cfg_scale": self.cfg_scale,
            "remasking": self.remasking,
            "mask_id": self.mask_id,
            "logits_eos_inf": self.logits_eos_inf,
            "confidence_eos_eot_inf": self.confidence_eos_eot_inf,
            "frontier_width": self.frontier_width,
            "n_items": self.n_items,
            "out_jsonl": self.out_jsonl,
            "out_ids": self.out_ids,
            "out_log": self.out_log,
            "extra": self.extra,
        }


def llada_diffusion_generate(
    model: Any,
    input_ids: Any,
    *,
    attention_mask: Any | None = None,
    steps: int = 256,
    gen_length: int = 256,
    block_length: int = 256,
    temperature: float = 0.0,
    cfg_scale: float = 0.0,
    remasking: str = "low_confidence",
    mask_id: int = LLADA_MASK_TOKEN_ID,
    logits_eos_inf: bool = False,
    confidence_eos_eot_inf: bool = True,
    frontier_width: int | None = None,
) -> tuple[Any, list[Any]]:
    """Run LLaDA blockwise masked diffusion decoding and record full step history.

    If ``model`` implements a stub ``diffusion_generate`` method (used in
    lightweight unit tests without torch), delegates to it. Otherwise runs
    the torch-backed LLaDA denoising loop and returns ``(sequences, history)``
    with ``len(history) == steps``.
    """
    stub_fn = getattr(model, "diffusion_generate", None)
    if callable(stub_fn):
        out = stub_fn(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=gen_length,
            steps=steps,
            mask_id=mask_id,
            frontier_width=frontier_width,
        )
        return getattr(out, "sequences", None), list(getattr(out, "history", []))

    import torch  # noqa: PLC0415
    import torch.nn.functional as F  # noqa: PLC0415

    if gen_length % block_length != 0:
        raise ValueError(
            f"gen_length ({gen_length}) must be divisible by block_length ({block_length})"
        )
    num_blocks = gen_length // block_length
    if steps % num_blocks != 0:
        raise ValueError(f"steps ({steps}) must be divisible by num_blocks ({num_blocks})")
    steps_per_block = steps // num_blocks
    schedule = compute_transfer_schedule(block_length, steps_per_block)

    prompt_len = int(input_ids.shape[1])
    device = input_ids.device
    x = torch.full(
        (input_ids.shape[0], prompt_len + gen_length),
        mask_id,
        dtype=torch.long,
        device=device,
    )
    x[:, :prompt_len] = input_ids.clone()

    if attention_mask is not None:
        full_attn = torch.cat(
            [
                attention_mask,
                torch.ones((input_ids.shape[0], gen_length), dtype=attention_mask.dtype, device=device),
            ],
            dim=-1,
        )
    else:
        full_attn = None

    history: list[Any] = []
    with torch.no_grad():
        for b_idx in range(num_blocks):
            block_start = prompt_len + b_idx * block_length
            block_end = prompt_len + (b_idx + 1) * block_length
            for s_idx in range(steps_per_block):
                mask_index = x == mask_id
                logits = model(x, attention_mask=full_attn).logits
                if logits_eos_inf:
                    logits[:, :, LLADA_EOS_TOKEN_ID] = -torch.inf
                if temperature > 0.0:
                    logits_f64 = logits.to(torch.float64)
                    noise = torch.rand_like(logits_f64, dtype=torch.float64)
                    gumbel = (-torch.log(noise.clamp_min(1e-12))) ** temperature
                    logits_noised = logits_f64.exp() / gumbel
                else:
                    logits_noised = logits.clone()
                if confidence_eos_eot_inf:
                    logits_noised[:, :, LLADA_EOS_TOKEN_ID] = -torch.inf
                    logits_noised[:, :, LLADA_EOT_TOKEN_ID] = -torch.inf
                x0 = torch.argmax(logits_noised, dim=-1)
                if remasking == "low_confidence":
                    p = F.softmax(logits.to(torch.float64), dim=-1)
                    x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
                    if confidence_eos_eot_inf:
                        eos_or_eot = (x0 == LLADA_EOS_TOKEN_ID) | (x0 == LLADA_EOT_TOKEN_ID)
                        x0_p = torch.where(eos_or_eot, torch.full_like(x0_p, -torch.inf), x0_p)
                elif remasking == "random":
                    x0_p = torch.rand(x0.shape, device=device, dtype=torch.float64)
                else:
                    raise NotImplementedError(f"Unsupported remasking strategy: {remasking}")

                # Restrict unmasking to the active block.
                x0_p[:, :block_start] = -torch.inf
                x0_p[:, block_end:] = -torch.inf
                if frontier_width is not None and frontier_width > 0:
                    for row in range(x.shape[0]):
                        masked_cols = torch.nonzero(mask_index[row, block_start:block_end], as_tuple=False)
                        if masked_cols.numel() > 0:
                            f_t = block_start + int(masked_cols[0].item())
                            cutoff = min(block_end, f_t + frontier_width)
                            x0_p[row, cutoff:block_end] = -torch.inf
                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, torch.full_like(x0_p, -torch.inf))

                k_transfer = schedule[s_idx]
                if k_transfer > 0:
                    for row in range(confidence.shape[0]):
                        _, select_idx = torch.topk(confidence[row], k=k_transfer)
                        x[row, select_idx] = x0[row, select_idx]
                history.append(x.clone())

    return x, history


def llada_history_to_trajectory(
    *,
    traj_id: str,
    prompt: str,
    gold_answer: str,
    history: Sequence[Any],
    tokenizer: TokenizerLike,
    prompt_len: int,
    mask_token_id: int = LLADA_MASK_TOKEN_ID,
    task: str = "arcc",
) -> Trajectory:
    """Convert LLaDA step history into a standard ``Trajectory``."""
    verdict_fn = extract_gsm8k_verdict if task.lower() == "gsm8k" else extract_verdict
    return history_to_trajectory(
        traj_id=traj_id,
        prompt=prompt,
        gold_answer=gold_answer,
        history=history,
        tokenizer=tokenizer,
        prompt_len=prompt_len,
        mask_token_id=mask_token_id,
        verdict_fn=verdict_fn,
    )


def run_llada_smoke(
    items: Sequence[dict[str, Any]],
    config: LLaDASmokeConfig | None = None,
    *,
    model: Any | None = None,
    tokenizer: Any | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """Run LLaDA-8B-Instruct or LLaDA-1.5 smoke evaluation and write JSONL + log."""
    import os
    import subprocess

    cfg = config or LLaDASmokeConfig()
    if model is None or tokenizer is None:
        import torch  # noqa: PLC0415
        from transformers import AutoModel, AutoTokenizer  # noqa: PLC0415

        tok = AutoTokenizer.from_pretrained(cfg.model_id, trust_remote_code=True)
        mdl = AutoModel.from_pretrained(
            cfg.model_id, trust_remote_code=True, torch_dtype=torch.bfloat16
        )
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            mdl = mdl.to(dev).eval()
        except Exception:
            pass
        model, tokenizer = mdl, tok

    try:
        dev = device or next(model.parameters()).device  # type: ignore[union-attr]
    except Exception:
        dev = device or "cuda"

    out_jsonl = Path(cfg.out_jsonl)
    out_ids = Path(cfg.out_ids)
    out_log = Path(cfg.out_log)
    for p in (out_jsonl, out_ids, out_log):
        if p.parent != Path("."):
            p.parent.mkdir(parents=True, exist_ok=True)

    ids = [str(it["id"]) for it in items]
    out_ids.write_text(json.dumps(ids, indent=2) + "\n", encoding="utf-8")

    verdict_fn = extract_gsm8k_verdict if cfg.task.lower() == "gsm8k" else extract_verdict
    trajs: list[Trajectory] = []
    sanity: list[dict[str, Any]] = []

    for it in items:
        prompt = str(it["prompt"])
        gold = str(it["gold_answer"])
        tid = str(it["id"])
        messages = [{"role": "user", "content": prompt}]
        try:
            chat_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            chat_text = prompt
        inputs = tokenizer(chat_text, return_tensors="pt")
        input_ids = inputs["input_ids"]
        attn = inputs.get("attention_mask")
        try:
            input_ids = input_ids.to(dev)
            if attn is not None:
                attn = attn.to(dev)
        except Exception:
            pass
        prompt_len = int(input_ids.shape[1]) if hasattr(input_ids, "shape") else len(input_ids[0])

        sequences, history = llada_diffusion_generate(
            model,
            input_ids,
            attention_mask=attn,
            steps=cfg.steps,
            gen_length=cfg.gen_length,
            block_length=cfg.block_length,
            temperature=cfg.temperature,
            cfg_scale=cfg.cfg_scale,
            remasking=cfg.remasking,
            mask_id=cfg.mask_id,
            logits_eos_inf=cfg.logits_eos_inf,
            confidence_eos_eot_inf=cfg.confidence_eos_eot_inf,
            frontier_width=cfg.frontier_width,
        )
        assert len(history) == cfg.steps, (
            f"{tid}: len(history)={len(history)} != steps={cfg.steps}"
        )
        traj = llada_history_to_trajectory(
            traj_id=tid,
            prompt=prompt,
            gold_answer=gold,
            history=history,
            tokenizer=tokenizer,
            prompt_len=prompt_len,
            mask_token_id=cfg.mask_id,
            task=cfg.task,
        )
        if sequences is not None:
            seq_ids = _as_int_list(sequences)
            seq_text = tokenizer.decode(seq_ids[prompt_len:], skip_special_tokens=True)
            final_verdict = traj.final_prediction
            seq_verdict = verdict_fn(seq_text)
            sanity.append(
                {
                    "id": tid,
                    "final_verdict": final_verdict,
                    "sequences_verdict": seq_verdict,
                    "match": final_verdict == seq_verdict,
                }
            )
            assert final_verdict == seq_verdict, (
                f"{tid}: final verdict {final_verdict!r} != sequences verdict {seq_verdict!r}"
            )
        sanity.append({"id": tid, "last_step_non_null": traj.final_prediction is not None})
        trajs.append(traj)

    from traj_eval.io import trajectories_to_dicts

    out_jsonl.write_text(
        "\n".join(json.dumps(d, ensure_ascii=False) for d in trajectories_to_dicts(trajs)) + "\n",
        encoding="utf-8",
    )

    git_rev = None
    try:
        git_rev = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, timeout=5
        ).strip()
    except Exception:
        git_rev = None

    log: dict[str, Any] = {
        "config": cfg.to_dict(),
        "git_revision": git_rev,
        "env": {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "device": str(dev),
        },
        "n_trajectories": len(trajs),
        "ids": ids,
        "sanity": sanity,
    }
    log.update(cfg.extra)
    out_log.write_text(json.dumps(log, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return log
