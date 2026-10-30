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

What this file provides
-----------------------
- ARC-C verdict extraction (:func:`extract_verdict`) and GSM8K numeric
  extraction (:func:`extract_gsm8k_verdict`, :func:`strip_gsm8k_exemplars`,
  :func:`normalize_numeric_answer`). Never reuse the A-D regex for GSM8K.
- Per-step token-commitment tracking (:func:`compute_step_commitments`)
  populating ``TrajectoryStep.answer_committed`` and
  ``TrajectoryStep.committed_reasoning_fraction`` inside
  :func:`history_to_trajectory`.
- Optional forced-deliberation EOS suppression (``force_deliberation_steps``)
  and frontier gating (``frontier_width``) via Dream's
  ``generation_logits_hook_func``.
- LLaDA checkpoints live in ``src/traj_eval/adapters/llada.py``; MMaDA is
  not in this module.

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
from typing import Any, Callable, Protocol, Sequence

from traj_eval.types import Trajectory, TrajectoryStep

DREAM_MODEL_ID = "Dream-org/Dream-v0-Instruct-7B"

# Answer-prefix patterns take precedence over a bare standalone letter.
# Require same-line whitespace/colon ([ \t:]*) and a word boundary (\b([A-D])\b)
# around the option letter so headings like "Provide the final answer\nAnswer: C"
# or "Choose the most likely option\nBased on..." never match the first letter
# of the next word ("Answer" -> "A", "Based" -> "B").
# Across all prefix patterns, we return the match that appears LAST in text order.
_ANSWER_PREFIX_RES = [
    re.compile(r"\\boxed\{\s*\(?\s*\b([A-D])\b\s*\)?\s*\}", re.IGNORECASE),
    re.compile(r"final\s+answer\s*(?:is\s*)?[ \t:]*\(?\s*\b([A-D])\b\s*\)?", re.IGNORECASE),
    re.compile(r"(?:correct\s+)?answer\s+is[ \t:]*\(?\s*\b([A-D])\b\s*\)?", re.IGNORECASE),
    re.compile(r"\banswer[ \t]*:[ \t]*\(?\s*\b([A-D])\b\s*\)?", re.IGNORECASE),
    re.compile(r"\boption[ \t:]*\(?\s*\b([A-D])\b\s*\)?", re.IGNORECASE),
    re.compile(r"\bchoice\s+is[ \t:]*\(?\s*\b([A-D])\b\s*\)?", re.IGNORECASE),
]
# Clause-initial English indefinite article "A <lowercase_word>" (excluding
# linking verbs like "A is/was/fits/matches") so partial CoT sentences such
# as "A planet rotates faster..." do not emit a false "A" verdict.
_ARTICLE_A_RE = re.compile(
    r"(^|[.!?]\s+|\n\s*)A\s+(?!(?:is|was|fits|matches|means|equals)\b)([a-z]{2,}\b)"
)
_STANDALONE_LETTER_RE = re.compile(r"\b([A-D])\b")


def extract_verdict(text: str) -> str | None:
    """Extract the final ARC-C option letter A-D from decoded text.

    Returns ``"A"``..``"D"`` or ``None`` when no letter is decodable.
    Never guesses.

    Precedence: answer-prefix patterns (``\\boxed{B}``, ``final answer: C``,
    ``answer is (B)``, ``Answer: A``, ``option B``) win over a bare
    standalone letter, and the last such match in text order wins.
    Otherwise the last standalone ``\\b([A-D])\\b`` match wins after
    stripping clause-initial English indefinite articles (``"A planet..."``).
    ``None`` for empty / all-mask text.
    """
    if not text or not text.strip():
        return None
    best_pos: tuple[int, int] | None = None
    best_letter: str | None = None
    for pat in _ANSWER_PREFIX_RES:
        for m in pat.finditer(text):
            pos = (m.end(), m.start())
            if best_pos is None or pos >= best_pos:
                best_pos = pos
                best_letter = str(m.group(1)).upper()
    if best_letter is not None:
        return best_letter
    scrubbed = _ARTICLE_A_RE.sub(r"\1\2", text)
    matches = _STANDALONE_LETTER_RE.findall(scrubbed)
    if matches:
        return str(matches[-1]).upper()
    return None


# GSM8K numeric extraction patterns.
# Intermediate arithmetic numbers in a chain-of-thought are NOT verdicts;
# only explicit answer markers ("#### <num>", "\\boxed{<num>}",
# "final answer: <num>", "the answer is <num>") emit a verdict.
_GSM8K_HASH_RE = re.compile(r"####\s*\$?\s*([+-]?\d[\d,]*(?:\.\d+)?)")
_GSM8K_BOXED_RE = re.compile(r"\\boxed\{\s*\$?\s*([+-]?\d[\d,]*(?:\.\d+)?)\s*\}")
_GSM8K_PREFIX_RES = [
    re.compile(
        r"(?:final\s+answer|the\s+answer\s+is|answer\s+is|answer\s*:)\s*(?:is\s*)?[:\s]*\$?\s*([+-]?\d[\d,]*(?:\.\d+)?)",
        re.IGNORECASE,
    ),
]
_EXEMPLAR_TURN_STOP_RE = re.compile(r"\n\s*(?:Question|Q)\s*:", re.IGNORECASE)


def normalize_numeric_answer(raw: str) -> str | None:
    """Normalize a numeric answer string (strip commas/$/+, canonicalize ints).

    Examples:
        ``"1,200"`` -> ``"1200"``
        ``"$18.00"`` -> ``"18"``
        ``"-3.50"`` -> ``"-3.5"``
        ``"+42"`` -> ``"42"``
    """
    if not raw:
        return None
    cleaned = raw.strip().replace(",", "").replace("$", "").strip()
    if cleaned.startswith("+"):
        cleaned = cleaned[1:]
    if not cleaned:
        return None
    if "." in cleaned:
        sign = "-" if cleaned.startswith("-") else ""
        body = cleaned[1:] if sign else cleaned
        int_part, frac_part = body.split(".", 1)
        frac_stripped = frac_part.rstrip("0")
        if not frac_stripped:
            cleaned = f"{sign}{int_part or '0'}"
        else:
            cleaned = f"{sign}{int_part or '0'}.{frac_stripped}"
    if cleaned in ("-0", "+0"):
        cleaned = "0"
    return cleaned


def strip_gsm8k_exemplars(text: str) -> str:
    """Strip echoed few-shot exemplars and trailing hallucinated Question turns.

    In 8-shot GSM8K prompting, either:
    1. A full prompt+generation string contains multiple ``Question: ... #### N``
       exemplar blocks before the final target question, OR
    2. A prompt-sliced generation region finishes answering the target question
       and then starts generating a new ``\\nQuestion: ...`` exemplar.

    This helper isolates the target response block so verdict extraction never
    reads an exemplar's answer.
    """
    if not text:
        return ""
    out = text
    # Case 1: full prompt with exemplars starting with "Question: ..."
    if re.match(r"^\s*(?:Question|Q)\s*:", out, flags=re.IGNORECASE) and re.search(
        r"Question\s*:.*####", out, flags=re.IGNORECASE | re.DOTALL
    ):
        parts = re.split(r"(?:^|\n)\s*(?:Question|Q)\s*:", out, flags=re.IGNORECASE)
        if len(parts) > 1:
            last_q = parts[-1]
            ans_split = re.split(
                r"\n\s*(?:Answer|A)\s*:", last_q, maxsplit=1, flags=re.IGNORECASE
            )
            if len(ans_split) == 2:
                out = ans_split[1]
            else:
                out = last_q
    # Case 2: prompt-sliced generation that continues into a new "\nQuestion:" turn.
    stop_match = _EXEMPLAR_TURN_STOP_RE.search(out)
    if stop_match:
        out = out[: stop_match.start()]
    return out.strip()


def extract_gsm8k_verdict(text: str) -> str | None:
    """Extract the normalized numeric verdict from a GSM8K generation step.

    Returns a normalized number string (e.g. ``"18"``, ``"-3.5"``) or ``None``
    when no explicit answer marker is decodable yet. Never guesses from bare
    intermediate operands in unfinished reasoning sentences.

    Precedence (after :func:`strip_gsm8k_exemplars`):
    1. ``#### <number>`` (last match)
    2. ``\\boxed{<number>}`` (last match)
    3. Explicit answer phrases (``Final answer: <number>``, ``The answer is <number>``)
    """
    cleaned = strip_gsm8k_exemplars(text)
    if not cleaned:
        return None
    for pat in (_GSM8K_HASH_RE, _GSM8K_BOXED_RE, *_GSM8K_PREFIX_RES):
        matches = pat.findall(cleaned)
        if matches:
            return normalize_numeric_answer(str(matches[-1]))
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


def compute_step_commitments(
    history: Sequence[Any],
    prompt_len: int,
    tokenizer: TokenizerLike,
    mask_token_id: int | None = None,
    final_verdict: str | None = None,
) -> list[tuple[bool, float]]:
    """Compute per-step ``(answer_committed, committed_reasoning_fraction)``.

    A generation canvas position ``j`` is irreversibly committed at step
    ``tau[j]`` once its token id leaves ``mask_token_id`` and stays equal
    to its final-step value for all remaining steps in ``history``.

    Content positions ``C`` are canvas positions whose final token is
    non-mask and decodes to a non-empty string with
    ``skip_special_tokens=True`` (excluding trailing EOS/padding tokens).
    When ``final_verdict`` is non-null, the answer position ``a`` is the
    last position in ``C`` whose decoded piece contains ``final_verdict``
    (or ``C[-1]`` if the verdict spans subwords), and the rationale-side
    region is ``R = [j for j in C if j < a]``.
    """
    n_steps = len(history)
    if n_steps == 0:
        return []
    mask_id = mask_token_id if mask_token_id is not None else tokenizer.mask_token_id
    raw_seqs: list[list[int]] = []
    for entry in history:
        ids = _as_int_list(entry)
        raw_seqs.append(ids[prompt_len:] if prompt_len < len(ids) else [])
    canvas_len = max((len(s) for s in raw_seqs), default=0)
    if canvas_len == 0:
        return [(False, 0.0) for _ in range(n_steps)]

    # Pad any shorter stub rows with mask_id so indexing is uniform.
    gen_seqs = [s + [mask_id] * (canvas_len - len(s)) for s in raw_seqs]
    final_row = gen_seqs[-1]

    # First irreversible commitment step tau[j] for each canvas slot j.
    inf_step = n_steps + 1
    tau: list[int] = [inf_step] * canvas_len
    for j in range(canvas_len):
        final_tok = final_row[j]
        if final_tok == mask_id:
            continue
        t = n_steps - 1
        while t > 0 and gen_seqs[t - 1][j] == final_tok:
            t -= 1
        tau[j] = t

    # Content positions: non-mask final tokens that decode to non-empty text.
    content_pos: list[int] = []
    pieces: dict[int, str] = {}
    for j, tok_id in enumerate(final_row):
        if tok_id == mask_id:
            continue
        piece = tokenizer.decode([tok_id], skip_special_tokens=True)
        if piece:
            content_pos.append(j)
            pieces[j] = piece

    if not content_pos or final_verdict is None:
        return [(False, 0.0) for _ in range(n_steps)]

    target = final_verdict.strip().upper()
    ans_pos: int | None = None
    # Pass 1: exact match after stripping whitespace and punctuation, scanning from end.
    if target:
        for j in reversed(content_pos):
            cleaned_piece = re.sub(r"[\s()\[\]{}:.,;\"'`$]+", "", pieces[j]).upper()
            if cleaned_piece == target:
                ans_pos = j
                break
    # Pass 2: fallback substring match if no exact token match exists.
    if ans_pos is None and target:
        for j in reversed(content_pos):
            if target in pieces[j].upper():
                ans_pos = j
                break
    # Pass 3: fallback to last content token.
    if ans_pos is None:
        ans_pos = content_pos[-1]

    rationale_pos = [j for j in content_pos if j < ans_pos]
    out: list[tuple[bool, float]] = []
    for t in range(n_steps):
        ans_comm = tau[ans_pos] <= t
        if rationale_pos:
            crf = sum(1 for j in rationale_pos if tau[j] <= t) / len(rationale_pos)
        else:
            crf = 1.0 if ans_comm else 0.0
        out.append((bool(ans_comm), round(float(crf), 4)))
    return out


def history_to_trajectory(
    *,
    traj_id: str,
    prompt: str,
    gold_answer: str,
    history: Sequence[Any],
    tokenizer: TokenizerLike,
    prompt_len: int,
    mask_token_id: int | None = None,
    verdict_fn: Callable[[str], str | None] = extract_verdict,
) -> Trajectory:
    """Convert Dream ``output.history`` to a standard ``Trajectory``.

    One step per history entry: ``step=i``, ``prediction`` is the
    extracted verdict (A-D letter by default, or custom ``verdict_fn``)
    or ``None``, ``reasoning_text`` is the decoded mask-stripped
    generation region, and ``answer_committed`` /
    ``committed_reasoning_fraction`` track token commitment across the
    canvas.

    Sanity: caller should assert ``len(history) == steps`` and that
    the final parsed verdict matches the verdict parsed from
    ``output.sequences`` (see ``run_dream_arcc_smoke``).
    """
    decoded_steps: list[tuple[str, str | None]] = []
    for entry in history:
        text = decode_generation_region(entry, prompt_len, tokenizer, mask_token_id)
        verdict = verdict_fn(text)
        decoded_steps.append((text, verdict))
    final_verdict = decoded_steps[-1][1] if decoded_steps else None
    commitments = compute_step_commitments(
        history=history,
        prompt_len=prompt_len,
        tokenizer=tokenizer,
        mask_token_id=mask_token_id,
        final_verdict=final_verdict,
    )
    steps: list[TrajectoryStep] = []
    for i, ((text, verdict), (ans_comm, crf)) in enumerate(zip(decoded_steps, commitments)):
        steps.append(
            TrajectoryStep(
                step=i,
                prediction=verdict,
                reasoning_text=text,
                answer_committed=ans_comm,
                committed_reasoning_fraction=crf,
            )
        )
    return Trajectory(id=traj_id, prompt=prompt, gold_answer=gold_answer, steps=steps)


@dataclass
class DreamSmokeConfig:
    """Sampler + run config for the Dream ARC-C smoke (5 items).

    Mirrors ``docs/adapter_dream_scoping.md`` smoke protocol. Do NOT
    change temperature / top_p / steps mid-run; log exactly what ran.
    """

    model_id: str = DREAM_MODEL_ID
    task: str = "arcc"
    max_new_tokens: int = 256
    steps: int = 256
    temperature: float = 0.2
    top_p: float = 0.95
    alg: str = "entropy"
    alg_temp: float = 0.0
    # When > 0, suppresses EOS token logits for the first K denoising steps
    # via Dream's generation_logits_hook_func (forced deliberation).
    force_deliberation_steps: int = 0
    # When set (e.g. w=16, Yeom et al. 2026), restricts eligible commitments at
    # step t to positions i < f(t) + frontier_width where f(t) is the left-most
    # masked position in the generation canvas.
    frontier_width: int | None = None
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
            "task": self.task,
            "max_new_tokens": self.max_new_tokens,
            "steps": self.steps,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "alg": self.alg,
            "alg_temp": self.alg_temp,
            "force_deliberation_steps": self.force_deliberation_steps,
            "frontier_width": self.frontier_width,
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

    verdict_fn = extract_gsm8k_verdict if cfg.task.lower() == "gsm8k" else extract_verdict

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

        gen_kwargs: dict[str, Any] = {}
        if cfg.force_deliberation_steps > 0 or (
            cfg.frontier_width is not None and cfg.frontier_width > 0
        ):
            eos_id = getattr(tokenizer, "eos_token_id", None)
            pad_id = getattr(tokenizer, "pad_token_id", None)
            suppress_ids = [i for i in (eos_id, pad_id) if isinstance(i, int)]
            f_width = cfg.frontier_width

            def _intervention_logits_hook(step_idx: Any, x: Any, logits: Any) -> Any:
                if (
                    cfg.force_deliberation_steps > 0
                    and step_idx is not None
                    and int(step_idx) < cfg.force_deliberation_steps
                ):
                    for sid in suppress_ids:
                        logits[..., sid] = float("-inf")
                if f_width is not None and f_width > 0 and hasattr(x, "shape"):
                    # Frontier gating (Yeom et al. 2026): flatten logits outside
                    # [f(t), f(t) + w) to 0.0 (uniform distribution -> minimum
                    # confidence under entropy/maskgit_plus/topk_margin) so only
                    # positions within w of the left-most mask f(t) are selected.
                    for r in range(x.shape[0]):
                        row_tokens = _as_int_list(x[r])
                        masked_cols = [
                            c for c in range(prompt_len, len(row_tokens)) if row_tokens[c] == mask_id
                        ]
                        if masked_cols:
                            f_t = masked_cols[0]
                            cutoff = f_t + f_width
                            if cutoff < len(row_tokens):
                                logits[r, cutoff:, :] = 0.0
                return logits

            gen_kwargs["generation_logits_hook_func"] = _intervention_logits_hook

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
            **gen_kwargs,
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
            verdict_fn=verdict_fn,
        )
        # Sanity 2: final parsed verdict == verdict from sequences.
        if sequences is not None:
            try:
                seq_ids = _as_int_list(sequences)
                seq_text = tokenizer.decode(seq_ids[prompt_len:], skip_special_tokens=True)
                # sequences may be batched; _as_int_list flattens first row.
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
