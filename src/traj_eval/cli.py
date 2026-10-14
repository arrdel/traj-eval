"""Command-line interface for traj-eval.

Usage:
    python -m traj_eval run --input examples/synthetic_demo.jsonl --output results.json
    python -m traj_eval demo
    python -m traj_eval example --output examples/synthetic_demo.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from traj_eval.io import load_trajectories, save_results, trajectories_to_dicts
from traj_eval.metrics import evaluate_trajectories
from traj_eval.synthetic import generate_synthetic_trajectories


def _print_summary(results: dict) -> None:
    agg = results["aggregate"]
    print("\n=== traj-eval summary ===")
    print(f"Trajectories:            {agg['n_trajectories']}")
    print(f"Accuracy (final):        {agg['accuracy']:.4f}")
    print(f"Mean flip rate:          {agg['mean_flip_rate']:.4f}")
    print(f"Mean rationalization:    {agg['mean_rationalization_score']:.4f}")
    print(f"Early convergence frac:  {agg['early_convergence_fraction']:.4f}")
    print(f"Mean convergence step:   {agg['mean_convergence_step']:.2f}")
    print(f"Median convergence step: {agg['median_convergence_step']:.2f}")
    print(f"Mean convergence frac:   {agg['mean_convergence_fraction']:.4f}")
    print("\nPer-trajectory:")
    header = f"{'id':<20} {'final':<8} {'gold':<6} {'corr':<5} {'conv':<5} {'flip':<6} {'rat':<6} {'steps':<5}"
    print(header)
    print("-" * len(header))
    for row in results["per_trajectory"]:
        corr = "" if row["correct"] is None else ("Y" if row["correct"] else "N")
        conv = "" if row["convergence_step"] is None else str(row["convergence_step"])
        print(
            f"{row['id']:<20} {str(row['final_prediction']):<8} {row['gold_answer']:<6} "
            f"{corr:<5} {conv:<5} {row['flip_rate']:<6.3f} {row['rationalization_score']:<6.3f} {row['n_steps']:<5}"
        )
    print()


def _cmd_run(args: argparse.Namespace) -> int:
    try:
        trajs = load_trajectories(args.input)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    results = evaluate_trajectories(trajs)
    _print_summary(results)
    if args.output:
        save_results(results, args.output)
        print(f"Saved JSON results to {args.output}")
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    trajs = generate_synthetic_trajectories(n_per_archetype=args.n_per_archetype)
    results = evaluate_trajectories(trajs)
    print(f"Demo on {len(trajs)} synthetic trajectories "
          f"({args.n_per_archetype} per archetype: genuine / rationalization / oscillation)")
    _print_summary(results)
    if args.output:
        save_results(results, args.output)
        print(f"Saved JSON results to {args.output}")
    return 0


def _cmd_example(args: argparse.Namespace) -> int:
    trajs = generate_synthetic_trajectories(n_per_archetype=args.n_per_archetype)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "\n".join(json.dumps(d) for d in trajectories_to_dicts(trajs)) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(trajs)} synthetic trajectories to {out_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="traj-eval",
        description="Trajectory-level evaluation for diffusion LMs.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Evaluate trajectories from a JSONL/JSON file.")
    p_run.add_argument("--input", required=True, help="Input .jsonl or .json file")
    p_run.add_argument("--output", default=None, help="Output JSON file (optional)")
    p_run.set_defaults(func=_cmd_run)

    p_demo = sub.add_parser("demo", help="Run evaluation on synthetic trajectories.")
    p_demo.add_argument("--n-per-archetype", type=int, default=5)
    p_demo.add_argument("--output", default=None, help="Output JSON file (optional)")
    p_demo.set_defaults(func=_cmd_demo)

    p_ex = sub.add_parser("example", help="Generate synthetic trajectories to a JSONL file.")
    p_ex.add_argument("--output", default="examples/synthetic_demo.jsonl")
    p_ex.add_argument("--n-per-archetype", type=int, default=5)
    p_ex.set_defaults(func=_cmd_example)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
