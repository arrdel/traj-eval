"""Run the Dream ARC-C smoke (5 items).

Usage (real GPU, cdr-gpu env):
    conda activate cdr-gpu
    CUDA_VISIBLE_DEVICES=<card with >=25GB free> python experiments/run_dream_arcc_smoke.py \\
        --items results/dream_arcc_smoke5_input.json \\
        --out-jsonl results/dream_arcc_smoke5.jsonl

The --items file is a JSON list of {"id","prompt","gold_answer"}.
Create it from ARC-C first (5 ids), write ids to
results/dream_arcc_smoke5_ids.json BEFORE running (the adapter also
writes them, but the protocol requires ids fixed before the run).

Fake test without GPU (Cursor agent / CI):
    python experiments/run_dream_arcc_smoke.py --fake --n-items 2

This exercises history_to_trajectory + JSONL writing with a stub
model that mimics diffusion_generate, no weights, no torch needed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running from repo root without install: add src to path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from traj_eval.adapters.dream import (  # noqa: E402
    DreamSmokeConfig,
    history_to_trajectory,
)


class _FakeTokenizer:
    mask_token_id = 999

    def __init__(self):
        self._vocab = {1: "Q", 20: "Answer is (", 21: "B", 22: ")", 999: ""}

    def decode(self, ids, skip_special_tokens=True):
        return "".join(self._vocab.get(int(i), "") for i in ids)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return messages[0]["content"]

    def __call__(self, text, return_tensors=None):
        # Minimal fake: input_ids [[1]] (prompt_len=1).
        class _Out(dict):
            pass

        out = _Out()
        out["input_ids"] = [[1]]
        out["attention_mask"] = [[1]]
        return out


class _FakeOutput:
    def __init__(self, history, sequences):
        self.history = history
        self.sequences = sequences


class _FakeModel:
    def diffusion_generate(self, *args, **kwargs):
        steps = int(kwargs.get("steps", 3))
        # Step0-1: all masks -> null. Step2+: "Answer is (B)" -> B.
        history = []
        for i in range(steps):
            if i < 2:
                history.append([1, 999, 999])
            else:
                history.append([1, 20, 21, 22])
        return _FakeOutput(history, [1, 20, 21, 22])


def _run_fake(n_items: int, out_jsonl: str, out_ids: str, out_log: str) -> int:
    tok = _FakeTokenizer()
    model = _FakeModel()
    items = [
        {"id": f"fake-{i}", "prompt": "Q?", "gold_answer": "B"} for i in range(n_items)
    ]
    trajs = []
    for it in items:
        out = model.diffusion_generate(steps=3)
        traj = history_to_trajectory(
            traj_id=it["id"],
            prompt=it["prompt"],
            gold_answer=it["gold_answer"],
            history=out.history,
            tokenizer=tok,
            prompt_len=1,
            mask_token_id=999,
        )
        trajs.append(traj)
    from traj_eval.io import trajectories_to_dicts

    for p, content in [
        (Path(out_jsonl), "\n".join(json.dumps(d) for d in trajectories_to_dicts(trajs)) + "\n"),
        (Path(out_ids), json.dumps([it["id"] for it in items], indent=2) + "\n"),
        (Path(out_log), json.dumps({"mode": "fake", "n": len(trajs)}, indent=2) + "\n"),
    ]:
        if p.parent != Path("."):
            p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    print(f"Fake smoke wrote {len(trajs)} trajectories to {out_jsonl}")
    print(f"Final verdicts: {[t.final_prediction for t in trajs]} (expect all 'B')")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Dream ARC-C smoke runner")
    ap.add_argument("--items", default="results/dream_arcc_smoke5_input.json",
                    help="JSON list of {id,prompt,gold_answer}")
    ap.add_argument("--out-jsonl", default="results/dream_arcc_smoke5.jsonl")
    ap.add_argument("--out-ids", default="results/dream_arcc_smoke5_ids.json")
    ap.add_argument("--out-log", default="results/dream_arcc_smoke5_run_log.json")
    ap.add_argument("--fake", action="store_true", help="Run with stub model, no GPU/weights")
    ap.add_argument("--n-items", type=int, default=2, help="Fake mode only")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--block-length", type=int, default=256, help="LLaDA block length")
    ap.add_argument(
        "--model-family",
        choices=["dream", "llada-instruct", "llada-1.5"],
        default="dream",
        help="Checkpoint family to evaluate",
    )
    ap.add_argument("--task", choices=["arcc", "gsm8k"], default="arcc")
    ap.add_argument("--force-deliberation-steps", type=int, default=0)
    args = ap.parse_args(argv)

    if args.fake:
        return _run_fake(args.n_items, args.out_jsonl, args.out_ids, args.out_log)

    items_path = Path(args.items)
    if not items_path.exists():
        print(
            f"ERROR: items file not found: {items_path}\n"
            "Create it first as a JSON list of {id, prompt, gold_answer} "
            "for 5 ARC-C items, with ids fixed before the run.",
            file=sys.stderr,
        )
        return 2
    items = json.loads(items_path.read_text(encoding="utf-8"))
    if len(items) != 5:
        print(f"WARNING: expected 5 items for smoke, got {len(items)}", file=sys.stderr)

    if args.model_family in ("llada-instruct", "llada-1.5"):
        from traj_eval.adapters.llada import (
            LLADA_15_ID,
            LLADA_INSTRUCT_ID,
            LLaDASmokeConfig,
            run_llada_smoke,
        )

        llada_id = LLADA_INSTRUCT_ID if args.model_family == "llada-instruct" else LLADA_15_ID
        lcfg = LLaDASmokeConfig(
            model_id=llada_id,
            task=args.task,
            gen_length=args.max_new_tokens,
            steps=args.steps,
            block_length=args.block_length,
            temperature=args.temperature,
            n_items=len(items),
            out_jsonl=args.out_jsonl,
            out_ids=args.out_ids,
            out_log=args.out_log,
        )
        log = run_llada_smoke(items, lcfg)
    else:
        from traj_eval.adapters.dream import run_dream_arcc_smoke

        cfg = DreamSmokeConfig(
            task=args.task,
            temperature=args.temperature,
            top_p=args.top_p,
            steps=args.steps,
            max_new_tokens=args.max_new_tokens,
            force_deliberation_steps=args.force_deliberation_steps,
            n_items=len(items),
            out_jsonl=args.out_jsonl,
            out_ids=args.out_ids,
            out_log=args.out_log,
        )
        log = run_dream_arcc_smoke(items, cfg)
    print(f"Wrote {log['n_trajectories']} trajectories to {args.out_jsonl}")
    print(f"Run log: {args.out_log}")
    print("Next: python -m traj_eval run --input", args.out_jsonl)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
