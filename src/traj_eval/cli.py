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
from traj_eval.metrics import (
    check_reasoning_smoke_gate,
    compare_conditions,
    evaluate_trajectories,
)
from traj_eval.synthetic import generate_synthetic_trajectories


def _print_summary(results: dict) -> None:
    agg = results["aggregate"]
    print("\n=== traj-eval summary ===")
    print(f"Trajectories:            {agg['n_trajectories']}")
    print(f"Accuracy (final):        {agg['accuracy']:.4f}")
    print(f"Mean flip rate:          {agg['mean_flip_rate']:.4f}")
    print(f"Mean rationalization:    {agg['mean_rationalization_score']:.4f}")
    if "mean_early_factor" in agg:
        print(
            f"  [components] early={agg['mean_early_factor']:.4f} "
            f"post_change={agg['mean_post_change_rate']:.4f} "
            f"len_growth={agg['mean_length_growth']:.4f} "
            f"gated={agg.get('mean_gated_rationalization_score', 0.0):.4f}"
        )
    print(f"Early convergence frac:  {agg['early_convergence_fraction']:.4f}")
    print(f"Mean convergence step:   {agg['mean_convergence_step']:.2f}")
    print(f"Median convergence step: {agg['median_convergence_step']:.2f}")
    print(f"Mean convergence frac:   {agg['mean_convergence_fraction']:.4f}")
    if agg.get("mean_answer_commitment_step") is not None:
        print(f"Mean T_commit step:      {agg['mean_answer_commitment_step']:.2f}")
        print(f"Mean lag (T_comm-T_stab):{agg['mean_stability_commitment_lag']:.2f}")
        print(f"Mean unwritten @ T_stab: {agg['mean_unwritten_reasoning_at_convergence']:.4f}")
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
    if getattr(args, "check_reasoning_gate", False):
        gate = check_reasoning_smoke_gate(trajs)
        results["reasoning_smoke_gate"] = gate
        print(f"Reasoning gate:          {gate['status']}")
    if args.output:
        save_results(results, args.output)
        print(f"Saved JSON results to {args.output}")
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    try:
        base_trajs = load_trajectories(args.base)
        target_trajs = load_trajectories(args.target)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    comp = compare_conditions(
        base_trajs,
        target_trajs,
        base_label=args.base_label,
        target_label=args.target_label,
    )
    b_lbl = comp["base_label"]
    t_lbl = comp["target_label"]
    b_agg = comp["base_aggregate"]
    t_agg = comp["target_aggregate"]
    print(f"\n=== traj-eval comparison ({b_lbl} vs {t_lbl}) ===")
    print(f"{'Metric':<28} {b_lbl:>12} {t_lbl:>12} {'Delta':>12}")
    print("-" * 66)
    print(
        f"{'Accuracy':<28} {b_agg['accuracy']:>12.4f} {t_agg['accuracy']:>12.4f} "
        f"{comp['accuracy_delta']['delta']:>+12.4f}"
    )
    for k, row in comp["rationalization_components_side_by_side"].items():
        print(
            f"{'rat.' + k:<28} {row[b_lbl]:>12.4f} {row[t_lbl]:>12.4f} {row['delta']:>+12.4f}"
        )
    print()
    if args.output:
        save_results(comp, args.output)
        print(f"Saved comparison JSON to {args.output}")
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
    p_run.add_argument(
        "--check-reasoning-gate",
        action="store_true",
        help="Validate CURSOR_FOLLOWUP_2026-10-29 §4 reasoning smoke gate",
    )
    p_run.set_defaults(func=_cmd_run)

    p_cmp = sub.add_parser(
        "compare",
        help="Compare two trajectory JSONL/JSON files side by side (e.g. direct vs reasoning).",
    )
    p_cmp.add_argument("--base", required=True, help="Base condition .jsonl or .json file")
    p_cmp.add_argument("--target", required=True, help="Target condition .jsonl or .json file")
    p_cmp.add_argument("--base-label", default="direct", help="Label for base condition")
    p_cmp.add_argument("--target-label", default="reasoning", help="Label for target condition")
    p_cmp.add_argument("--output", default=None, help="Output comparison JSON file (optional)")
    p_cmp.set_defaults(func=_cmd_compare)

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
