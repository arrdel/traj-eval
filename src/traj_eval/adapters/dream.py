"""Dream adapter for traj-eval.

Spec: ``docs/adapter_dream_scoping.md`` (Dream-first, ARC-C-first,
regex + null-for-undecided verdict rule).

Model: ``Dream-org/Dream-v0-Instruct-7B`` (the Instruct variant).
Do NOT use ``Dream-org/DreamOn-v0-7B`` (infilling) for this adapter;
its page is a code-pattern reference only.

How it works
------------
Dream's ``diffusion_generate`` with ``output_history=True`` and
``return_dict_in_generate=True`` returns ``output.history``: a list
with one cloned full-length tensor per denoising step, each of length
``input_len + max_new_tokens``, padded with ``mask_token_id``.

The adapter, for each history entry:
1. Slices off the prompt (``input_ids.shape[1]`` tokens).
2. Strips mask tokens from the generation region.
3. Decodes the remainder to text (``reasoning_text``).
4. Extracts the final standalone option letter A-D via regex
   (``extract_verdict``). No decodable letter -> ``prediction=None``
   (no verdict), never a guess.

Null rule (DECIDED, 2026-10-19)
------------------------------
``None`` means "no verdict yet", NOT a distinct answer. Metrics are
null-aware (see ``traj_eval.metrics``):
- A ``None -> "B"`` transition is NOT a flip. Flips are counted
  only between two non-null, differing verdicts.
- Convergence is computed over the non-null verdict sequence,
  mapped back to the original step index.
- Null steps still count toward trajectory length (``n_steps``,
  convergence fraction denominator).

Do NOT change this rule in the adapter without updating the metrics
docstring, the tests, and the run log.

What this file does NOT do
-------------------------
- No token-commitment logging (Yeom et al. style ``answer_committed``).
  History gives us what we need to add it later; first run is
  parsed-verdict only.
- No LLaDA / MMaDA adapters.
- No forced deliberation.
- No GSM8K numeric extraction. Do NOT reuse the A-D regex for GSM8K.

Env: run from the ``cdr-gpu`` conda env (torch 2.5.1+cu121,
transformers 4.46.2), NOT base (torch 2.11 + transformers 5.5, CUDA
unavailable) and NOT the repo ``.venv`` (no transformers). Pin a
single GPU with >= ~25 GB free via ``CUDA_VISIBLE_DEVICES``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence

from traj_eval.types import Trajectory, TrajectoryStep

DREAM_MODEL_ID = "Dream-org/Dream-v0-Instruct-7B"

# Answer-prefix patterns take precedence over a bare standalone letter.
# We return the LAST match: the most recent explicit answer statement.
_ANSWER_PREFIX_RES = [
    re.compile(r"answer\s+is\s*[:\s]*\(?\s*([A-D])\s*\)?", re.IGNORECASE),
    re.compile(r"final\s+answer\s*[:\s]*\(?\s*([A-D])\s*\)?", re.IGNORECASE),
    re.compile(r"option\s*[:\s]*\(?\s*([A-D])\s*\)?", re.IGNORECASE),
    re.compile(r"choice\s+is\s*[:\s]*\(?\s*([A-D])\s*\)?", re.IGNORECASE),
]
_STANDALONE_LETTER_RE = re.compile(r"\b([A-D])\b")


def extract_verdict(text: str) -> str | None:
    """Extract the final ARC-C option letter A-D from decoded text.

    Returns ``"A"``..``"D"`` or ``None`` when no letter is decodable.
    Never guesses.

    Precedence: answer-prefix patterns (``answer is (B)``, ``final
    answer: C``, ``option B``) win over a bare standalone letter, and
    the last such match wins. Otherwise the last standalone
    ``\\b([A-D])\\b`` match wins. ``None`` for empty / all-mask text.

    Limitation: a bare ``"A"`` as an English article can match the
    standalone pattern. In ARC-C smoke this is acceptable because the
    eval prompt asks for a letter and answer-prefix matches take
    precedence. If this becomes a problem, tighten to require the
    letter on its own line or after "Answer:" in the prompt design,
    do not silently guess.
    """
    if not text or not text.strip():
        return None
    for pat in _ANSWER_PREFIX_RES:
        matches = pat.findall(text)
        if matches:
            # findall returns the captured group; last wins.
            return str(matches[-1]).upper()
    matches = _STANDALONE_LETTER_RE.findall(text)
    if matches:
        return str(matches[-1]).upper()
    return None


class TokenizerLike(Protocol):
    """Minimal tokenizer surface the adapter needs (duck-typed).

    Real Hugging Face tokenizers satisfy this. Tests use a fake.
    """

    mask_token_id: int

    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = ...) -> str: ...


def _as_int_list(tokens: Any) -> list[int]:
    """Coerce a history entry to a plain list of ints.

    Accepts: list[int], tuple, torch tensor (via .tolist()), numpy
    array (via .tolist()). Keeps the adapter testable without torch.
    """
    if isinstance(tokens, list):
        return [int(x) for x in tokens]
    if isinstance(tokens, tuple):
        return [int(x) for x in tokens]
    tolist = getattr(tokens, "tolist", None)
    if callable(tolist):
        raw = tolist()
        # tolist() may return nested lists for batched tensors.
        if raw and isinstance(raw[0], (list, tuple)):
            raw = raw[0]
        return [int(x) for x in raw]
    # Fallback: iterate.
    return [int(x) for x in tokens]  # type: ignore[arg-type]


def decode_generation_region(
    history_entry: Any,
    prompt_len: int,
    tokenizer: TokenizerLike,
    mask_token_id: int | None = None,
) -> str:
    """Decode one history step's generation region, masks stripped.

    Args:
        history_entry: full sequence for this step (prompt + generation),
            as a list of token ids, a torch tensor, or a numpy array.
        prompt_len: number of prompt tokens to slice off (``input_ids.shape[1]``).
        tokenizer: object with ``decode(ids, skip_special_tokens=True)``.
        mask_token_id: mask id to strip. Defaults to
            ``tokenizer.mask_token_id`` when None.

    Returns decoded text with mask tokens removed. May be ``""`` when
    the region is still all masks (early steps) - that is the intended
    no-verdict case, not an error.
    """
    mask_id = mask_token_id if mask_token_id is not None else tokenizer.mask_token_id
    ids = _as_int_list(history_entry)
    gen_ids = ids[prompt_len:] if prompt_len < len(ids) else []
    # Strip masks before decoding so regex never sees a mask placeholder.
    gen_ids = [t for t in gen_ids if t != mask_id]
    if not gen_ids:
        return ""
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


def history_to_trajectory(
    *,
    traj_id: str,
    prompt: str,
    gold_answer: str,
    history: Sequence[Any],
    tokenizer: TokenizerLike,
    prompt_len: int,
    mask_token_id: int | None = None,
) -> Trajectory:
    """Convert Dream ``output.history`` to a standard ``Trajectory``.

    One step per history entry: ``step=i``, ``prediction`` is the
    extracted A-D letter or ``None``, ``reasoning_text`` is the
    decoded mask-stripped generation region.

    Sanity: caller should assert ``len(history) == steps`` and that
    the final parsed verdict matches the verdict parsed from
    ``output.sequences`` (see ``run_dream_arcc_smoke``).
    """
    steps: list[TrajectoryStep] = []
    for i, entry in enumerate(history):
        text = decode_generation_region(entry, prompt_len, tokenizer, mask_token_id)
        verdict = extract_verdict(text)
        steps.append(TrajectoryStep(step=i, prediction=verdict, reasoning_text=text))
    return Trajectory(id=traj_id, prompt=prompt, gold_answer=gold_answer, steps=steps)


@dataclass
class DreamSmokeConfig:
    """Sampler + run config for the Dream ARC-C smoke (5 items).

    Mirrors ``docs/adapter_dream_scoping.md`` smoke protocol. Do NOT
    change temperature / top_p / steps mid-run; log exactly what ran.
    """

    model_id: str = DREAM_MODEL_ID
    max_new_tokens: int = 256
    steps: int = 256
    temperature: float = 0.2
    top_p: float = 0.95
    alg: str = "entropy"
    alg_temp: float = 0.0
    # Output locations (relative to repo root unless absolute).
    n_items: int = 5
    out_jsonl: str = "results/dream_arcc_smoke5.jsonl"
    out_ids: str = "results/dream_arcc_smoke5_ids.json"
    out_log: str = "results/dream_arcc_smoke5_run_log.json"
    # Extra free-form metadata recorded in the run log.
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "max_new_tokens": self.max_new_tokens,
            "steps": self.steps,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "alg": self.alg,
            "alg_temp": self.alg_temp,
            "n_items": self.n_items,
            "out_jsonl": self.out_jsonl,
            "out_ids": self.out_ids,
            "out_log": self.out_log,
            "extra": self.extra,
        }


DEFAULT_SMOKE_CONFIG = DreamSmokeConfig()


def run_dream_arcc_smoke(
    items: Sequence[dict[str, Any]],
    config: DreamSmokeConfig | None = None,
    *,
    model: Any | None = None,
    tokenizer: Any | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """Run the Dream ARC-C 5-item smoke and write JSONL + ids + run log.

    Args:
        items: list of ``{"id": str, "prompt": str, "gold_answer": str}``
            dicts. Caller must write ``items`` ids to ``out_ids`` BEFORE
            running (see smoke protocol) - this function also writes
            them for convenience if the file does not exist.
        config: sampler / output config. Defaults to ``DEFAULT_SMOKE_CONFIG``.
        model / tokenizer: if None, loaded lazily from ``config.model_id``
            via transformers (requires ``cdr-gpu`` env + GPU). Tests
            pass fakes here; do NOT pass real weights in tests.
        device: e.g. ``"cuda:0"``. When None, uses model device if
            available, else ``"cuda"`` if torch.cuda is available.

    Returns the run-log dict that was written to ``out_log``.

    Side effects: writes ``out_jsonl`` (one trajectory per line),
    ``out_ids`` (ids that were run), ``out_log`` (config + git
    revision if available + sanity checks). Raises ``AssertionError``
    if sanity checks fail: ``len(history) == steps`` or final parsed
    verdict != verdict parsed from ``sequences``.
    """
    import os
    import subprocess

    cfg = config or DEFAULT_SMOKE_CONFIG
    if model is None or tokenizer is None:
        # Lazy heavy imports: keep module import light and follow Dream's
        # AutoModel registration (DreamConfig registers AutoModel, not
        # AutoModelForCausalLM).
        import torch  # noqa: PLC0415 - lazy, cdr-gpu env only
        from transformers import AutoModel, AutoTokenizer  # noqa: PLC0415

        tok = AutoTokenizer.from_pretrained(cfg.model_id, trust_remote_code=True)
        mdl = AutoModel.from_pretrained(
            cfg.model_id, trust_remote_code=True, torch_dtype=torch.bfloat16
        )
        # Dream generation lives on the model class; keep device handling simple.
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            mdl = mdl.to(dev).eval()
        except Exception:
            pass
        model, tokenizer = mdl, tok

    # Resolve device for input tensors.
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

    # Write ids BEFORE running, per protocol.
    ids = [str(it["id"]) for it in items]
    out_ids.write_text(json.dumps(ids, indent=2) + "\n", encoding="utf-8")

    trajs: list[Trajectory] = []
    sanity: list[dict[str, Any]] = []
    for it in items:
        prompt = str(it["prompt"])
        gold = str(it["gold_answer"])
        tid = str(it["id"])
        # Chat template, per spec.
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
        prompt_len = int(input_ids.shape[1])
        mask_id = int(getattr(tokenizer, "mask_token_id", 0) or 0)

        output = model.diffusion_generate(
            input_ids,
            attention_mask=attn,
            max_new_tokens=cfg.max_new_tokens,
            steps=cfg.steps,
            output_history=True,
            return_dict_in_generate=True,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            alg=cfg.alg,
            alg_temp=cfg.alg_temp,
        )
        history = list(getattr(output, "history", []))
        sequences = getattr(output, "sequences", None)
        # Sanity 1: len(history) == steps.
        assert len(history) == cfg.steps, (
            f"{tid}: len(history)={len(history)} != steps={cfg.steps}"
        )
        traj = history_to_trajectory(
            traj_id=tid,
            prompt=prompt,
            gold_answer=gold,
            history=history,
            tokenizer=tokenizer,
            prompt_len=prompt_len,
            mask_token_id=mask_id,
        )
        # Sanity 2: final parsed verdict == verdict from sequences.
        if sequences is not None:
            try:
                seq_ids = _as_int_list(sequences)
                seq_text = tokenizer.decode(seq_ids[prompt_len:], skip_special_tokens=True)
                # sequences may be batched; _as_int_list flattens first row.
                final_verdict = traj.final_prediction
                seq_verdict = extract_verdict(seq_text)
                sanity.append(
                    {
                        "id": tid,
                        "final_verdict": final_verdict,
                        "sequences_verdict": seq_verdict,
                        "match": final_verdict == seq_verdict,
                    }
                )
                assert final_verdict == seq_verdict, (
                    f"{tid}: final verdict {final_verdict!r} != "
                    f"sequences verdict {seq_verdict!r} (text={seq_text[:200]!r})"
                )
            except AssertionError:
                raise
            except Exception as exc:
                sanity.append({"id": tid, "sequences_check": f"skipped: {exc}"})
        # Sanity 3 (soft): at least last step non-null on answered items.
        # We log, not assert, because a model may legitimately leave an
        # item undecided; the paper must report that, not crash on it.
        last_non_null = traj.final_prediction is not None
        sanity.append({"id": tid, "last_step_non_null": last_non_null})
        trajs.append(traj)

    # Write JSONL: prediction None -> null in JSON (via io helper).
    from traj_eval.io import trajectories_to_dicts

    out_jsonl.write_text(
        "\n".join(json.dumps(d, ensure_ascii=False) for d in trajectories_to_dicts(trajs)) + "\n",
        encoding="utf-8",
    )

    # Git revision, best effort.
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
        "null_rule": (
            "None = no verdict yet, not a distinct answer. "
            "Flip counted only between two non-null differing verdicts. "
            "Convergence over non-null sequence, mapped to original index."
        ),
    }
    # Record GPU name if torch available.
    try:
        import torch as _torch

        if _torch.cuda.is_available():
            log["env"]["gpu_name"] = _torch.cuda.get_device_name(0)
    except Exception:
        pass
    log.update(cfg.extra)
    out_log.write_text(json.dumps(log, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return log
