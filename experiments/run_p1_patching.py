"""Run P1 hidden-state patching (k vs k+10) on LLaDA-8B-Instruct ARC-C 64-item reasoning.

Usage:
  1) Capture baseline hidden states at steps k in {5, 15, 25, 35, 45} (supports sharding):
     CUDA_VISIBLE_DEVICES=0 python experiments/run_p1_patching.py --mode capture --shard 0/4

  2) Run partner-swap patching at a given step k (supports sharding or single k per GPU):
     CUDA_VISIBLE_DEVICES=0 python experiments/run_p1_patching.py --mode patch --k 5

  3) Run matched-norm Gaussian noise control at a given step k:
     CUDA_VISIBLE_DEVICES=0 python experiments/run_p1_patching.py --mode noise --k 5

  4) Aggregate all patching JSONLs into results/p1_llada_patching_summary.json:
     python experiments/run_p1_patching.py --mode summarize
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from traj_eval.adapters.llada import (  # noqa: E402
    LLADA_INSTRUCT_ID,
    LLADA_MASK_TOKEN_ID,
    HiddenPatchSpec,
    llada_diffusion_generate,
    llada_history_to_trajectory,
)
from traj_eval.io import load_trajectories, trajectories_to_dicts  # noqa: E402
from traj_eval.metrics import answer_commitment_step, verdict_convergence_step  # noqa: E402

DEFAULT_STEPS = (5, 15, 25, 35, 45)
DEFAULT_LAYER = 16


def _load_items(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "items" in data:
        data = data["items"]
    return list(data)


def _prepare_inputs(tokenizer: Any, prompt: str, device: Any) -> tuple[Any, Any, int]:
    messages = [{"role": "user", "content": prompt}]
    try:
        chat_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        chat_text = prompt
    inputs = tokenizer(chat_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attn = inputs.get("attention_mask")
    if attn is not None:
        attn = attn.to(device)
    prompt_len = int(input_ids.shape[1])
    return input_ids, attn, prompt_len


def _load_llada(model_id: str, device: str) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    mdl = AutoModel.from_pretrained(
        model_id, trust_remote_code=True, torch_dtype=torch.bfloat16
    )
    mdl = mdl.to(device).eval()
    return mdl, tok


def _parse_shard(shard_str: str | None, total: int) -> tuple[int, int]:
    if not shard_str:
        return 0, total
    idx_s, n_s = shard_str.split("/")
    idx, n = int(idx_s), int(n_s)
    chunk = (total + n - 1) // n
    start = idx * chunk
    end = min(total, (idx + 1) * chunk)
    return start, end


def run_capture(args: argparse.Namespace) -> None:
    import torch

    items = _load_items(Path(args.items))
    start, end = _parse_shard(args.shard, len(items))
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    steps = tuple(int(s) for s in args.capture_steps.split(","))

    model, tok = _load_llada(args.model_id, args.device)
    for idx in range(start, end):
        it = items[idx]
        tid = str(it["id"])
        out_pt = cache_dir / f"{tid}.pt"
        if out_pt.exists() and not args.overwrite:
            continue
        input_ids, attn, prompt_len = _prepare_inputs(tok, str(it["prompt"]), args.device)
        captured: dict[int, Any] = {}
        spec = HiddenPatchSpec(layer_idx=args.layer, capture_steps=steps, gen_only=True)
        _, history = llada_diffusion_generate(
            model,
            input_ids,
            attention_mask=attn,
            steps=256,
            gen_length=256,
            block_length=256,
            temperature=0.0,
            remasking="low_confidence",
            confidence_eos_eot_inf=True,
            patch_spec=spec,
            captured_hidden=captured,
        )
        traj = llada_history_to_trajectory(
            traj_id=tid,
            prompt=str(it["prompt"]),
            gold_answer=str(it["gold_answer"]),
            history=history,
            tokenizer=tok,
            prompt_len=prompt_len,
            mask_token_id=LLADA_MASK_TOKEN_ID,
            task="arcc",
        )
        torch.save(
            {
                "id": tid,
                "final_prediction": traj.final_prediction,
                "gold_answer": str(it["gold_answer"]),
                "commit_step": answer_commitment_step(traj),
                "conv_step": verdict_convergence_step(traj),
                "hidden": captured,
            },
            out_pt,
        )
        print(f"[capture {idx+1}/{len(items)}] {tid} pred={traj.final_prediction} commit={answer_commitment_step(traj)}")


def run_patch(args: argparse.Namespace, noise_mode: bool = False) -> None:
    import torch

    items = _load_items(Path(args.items))
    n = len(items)
    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    k = int(args.k)
    tag = f"noise_k{k}" if noise_mode else f"patch_k{k}"
    out_jsonl = out_dir / f"p1_llada_{tag}.jsonl"

    model, tok = _load_llada(args.model_id, args.device)
    trajs = []
    gen = torch.Generator(device="cpu").manual_seed(42 + k)

    for idx, it in enumerate(items):
        tid = str(it["id"])
        partner_tid = str(items[(idx + 1) % n]["id"])
        base_data = torch.load(cache_dir / f"{tid}.pt", map_location="cpu", weights_only=True)
        partner_data = torch.load(cache_dir / f"{partner_tid}.pt", map_location="cpu", weights_only=True)

        base_h = base_data["hidden"][k]
        partner_h = partner_data["hidden"][k]
        if noise_mode:
            delta = partner_h.float() - base_h.float()
            r = torch.randn(delta.shape, generator=gen, dtype=torch.float32)
            r = r / r.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            r = r * delta.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            patch_h = (base_h.float() + r).to(base_h.dtype)
        else:
            patch_h = partner_h

        input_ids, attn, prompt_len = _prepare_inputs(tok, str(it["prompt"]), args.device)
        spec = HiddenPatchSpec(
            layer_idx=args.layer,
            patch_step=k,
            patch_hidden=patch_h,
            gen_only=True,
        )
        _, history = llada_diffusion_generate(
            model,
            input_ids,
            attention_mask=attn,
            steps=256,
            gen_length=256,
            block_length=256,
            temperature=0.0,
            remasking="low_confidence",
            confidence_eos_eot_inf=True,
            patch_spec=spec,
        )
        traj = llada_history_to_trajectory(
            traj_id=tid,
            prompt=str(it["prompt"]),
            gold_answer=str(it["gold_answer"]),
            history=history,
            tokenizer=tok,
            prompt_len=prompt_len,
            mask_token_id=LLADA_MASK_TOKEN_ID,
            task="arcc",
        )
        trajs.append(traj)
        print(
            f"[{tag} {idx+1}/{n}] {tid}: base={base_data['final_prediction']} -> patched={traj.final_prediction}"
        )

    out_jsonl.write_text(
        "\n".join(json.dumps(d, ensure_ascii=False) for d in trajectories_to_dicts(trajs)) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {out_jsonl}")


def summarize(args: argparse.Namespace) -> None:
    import math

    base_trajs = load_trajectories(args.baseline_jsonl)
    base_by_id = {t.id: t for t in base_trajs}
    out_dir = Path(args.out_dir)

    summary: dict[str, Any] = {
        "model_id": args.model_id,
        "layer_idx": args.layer,
        "n_items": len(base_trajs),
        "baseline_accuracy": sum(
            1 for t in base_trajs if t.final_prediction == t.gold_answer
        ) / len(base_trajs),
        "conditions": [],
    }

    for p in sorted(out_dir.glob("p1_llada_*.jsonl")):
        name = p.stem.replace("p1_llada_", "")
        trajs = load_trajectories(p)
        verdict_flips = []
        commit_deltas = []
        correct = []
        for t in trajs:
            b = base_by_id[t.id]
            verdict_flips.append(1.0 if t.final_prediction != b.final_prediction else 0.0)
            correct.append(1.0 if t.final_prediction == t.gold_answer else 0.0)
            c_b = answer_commitment_step(b)
            c_t = answer_commitment_step(t)
            if c_b is not None and c_t is not None:
                commit_deltas.append(float(c_t - c_b))
        n = len(trajs)
        flip_mean = sum(verdict_flips) / n
        flip_se = math.sqrt(flip_mean * (1.0 - flip_mean) / n)
        cd_mean = sum(commit_deltas) / len(commit_deltas) if commit_deltas else 0.0
        cd_var = (
            sum((x - cd_mean) ** 2 for x in commit_deltas) / max(1, len(commit_deltas) - 1)
            if len(commit_deltas) > 1
            else 0.0
        )
        cd_se = math.sqrt(cd_var / len(commit_deltas)) if commit_deltas else 0.0
        summary["conditions"].append(
            {
                "name": name,
                "jsonl": str(p),
                "n": n,
                "accuracy": sum(correct) / n,
                "verdict_flip_rate": flip_mean,
                "verdict_flip_se": flip_se,
                "mean_commit_step_delta": cd_mean,
                "se_commit_step_delta": cd_se,
            }
        )

    out_json = out_dir / "p1_llada_patching_summary.json"
    out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("capture", "patch", "noise", "summarize"), required=True)
    parser.add_argument("--items", default="results/arcc_smoke64_reasoning_input.json")
    parser.add_argument("--baseline-jsonl", default="results/llada_instruct_arcc_smoke64_reasoning.jsonl")
    parser.add_argument("--cache-dir", default="results/p1_hidden_cache")
    parser.add_argument("--out-dir", default="results/p1_patching")
    parser.add_argument("--model-id", default=LLADA_INSTRUCT_ID)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    parser.add_argument("--capture-steps", default=",".join(str(s) for s in DEFAULT_STEPS))
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--shard", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.mode == "capture":
        run_capture(args)
    elif args.mode == "patch":
        run_patch(args, noise_mode=False)
    elif args.mode == "noise":
        run_patch(args, noise_mode=True)
    else:
        summarize(args)


if __name__ == "__main__":
    main()
