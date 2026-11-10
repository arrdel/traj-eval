"""Stage and publish the `traj-eval` Hugging Face Static Space, Dataset, and Collection.

Supports:
1. `--dry-run`: Stages both the Hugging Face Static Space folder (`space/`) and
   the Hugging Face Dataset folder (`dataset/`, including per-trajectory
   metrics splits in `results/metrics/`) locally under `build/hf_staging/`
   with zero network calls so they can be inspected or tested offline.
2. `--push`: Creates/updates the Hugging Face Space (`<namespace>/<space-name>`),
   Dataset (`<namespace>/<dataset-name>`), and Collection via `huggingface_hub`
   (`HfApi`).

Usage:
  # Stage locally without network calls
  python scripts/publish_hf.py --dry-run

  # Authenticate and push Space, Dataset, and Collection to Hugging Face Hub
  HF_TOKEN=hf_xxx python scripts/publish_hf.py --push
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
HF_DIR = REPO_ROOT / "hf"
SITE_DIR = REPO_ROOT / "site"
RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_HF_NAMESPACE = "arrdel"

# Only the 9 verified 64-item benchmark suites and their paired comparisons/inputs
# are published in the Hugging Face Dataset bundle. 5-item pipeline smoke runs are
# intentionally excluded per Issue #29.
DATASET_RESULT_GLOBS: tuple[str, ...] = (
    "*smoke64*.jsonl",
    "*smoke64*_results.json",
    "*smoke64*_run_log.json",
    "*smoke64*_vs_*.json",
    "*smoke64*_input.json",
    "*smoke64*_ids.json",
    "baselines.md",
)

SUITES_64: tuple[tuple[str, str], ...] = (
    ("dream_arcc_direct_64", "dream_arcc_smoke64_direct"),
    ("dream_arcc_reasoning_64", "dream_arcc_smoke64_reasoning"),
    ("llada_instruct_arcc_direct_64", "llada_instruct_arcc_smoke64_direct"),
    ("llada_instruct_arcc_reasoning_64", "llada_instruct_arcc_smoke64_reasoning"),
    ("llada_15_arcc_direct_64", "llada_15_arcc_smoke64_direct"),
    ("llada_15_arcc_reasoning_64", "llada_15_arcc_smoke64_reasoning"),
    ("dream_gsm8k_64", "dream_gsm8k_smoke64"),
    ("llada_instruct_gsm8k_64", "llada_instruct_gsm8k_smoke64"),
    ("llada_15_gsm8k_64", "llada_15_gsm8k_smoke64"),
)

SPACE_SITE_ASSETS: tuple[str, ...] = (
    "index.html",
    "styles.css",
    "app.js",
)


def resolve_hf_token(repo_root: Path = REPO_ROOT) -> str | None:
    """Resolve `HF_TOKEN` from environment variables or a local `.env` file."""
    for env_key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        val = os.environ.get(env_key)
        if val and val.strip():
            return val.strip()

    dotenv_path = repo_root / ".env"
    if dotenv_path.exists():
        text = dotenv_path.read_text(encoding="utf-8")
        matches = [m for m in re.findall(r"hf_[A-Za-z0-9]{16,}", text)]
        if matches:
            return matches[-1]
    return None


def stage_hf_space_dir(
    staging_dir: Path,
    site_dir: Path = SITE_DIR,
    hf_dir: Path = HF_DIR,
) -> Path:
    """Populate a clean directory ready to upload as a Hugging Face Static Space."""
    space_readme = hf_dir / "space_README.md"
    if not space_readme.exists():
        raise FileNotFoundError(f"Missing Space README template: {space_readme}")

    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(space_readme, staging_dir / "README.md")
    for asset in SPACE_SITE_ASSETS:
        src = site_dir / asset
        if not src.exists():
            raise FileNotFoundError(f"Missing required site asset for HF Space: {src}")
        shutil.copy2(src, staging_dir / asset)

    return staging_dir


def stage_hf_dataset_dir(
    staging_dir: Path,
    results_dir: Path = RESULTS_DIR,
    hf_dir: Path = HF_DIR,
) -> Path:
    """Populate a clean directory ready to upload as a Hugging Face Dataset repo."""
    dataset_readme = hf_dir / "dataset_README.md"
    if not dataset_readme.exists():
        raise FileNotFoundError(f"Missing Dataset README template: {dataset_readme}")

    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(dataset_readme, staging_dir / "README.md")
    out_results = staging_dir / "results"
    out_results.mkdir(parents=True, exist_ok=True)

    for pattern in DATASET_RESULT_GLOBS:
        for src in sorted(results_dir.glob(pattern)):
            if "smoke5" in src.name:
                continue
            shutil.copy2(src, out_results / src.name)

    metrics_dir = out_results / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    for split_name, file_stem in SUITES_64:
        res_json_path = results_dir / f"{file_stem}_results.json"
        if not res_json_path.exists():
            continue
        res_data = json.loads(res_json_path.read_text(encoding="utf-8"))
        per_traj = res_data.get("per_trajectory", [])
        out_jsonl = metrics_dir / f"{file_stem}_metrics.jsonl"
        is_dream = file_stem.startswith("dream_")
        lines = []
        for row in per_traj:
            clean_row = {
                "suite": split_name,
                "id": str(row["id"]),
                "final_prediction": row.get("final_prediction"),
                "gold_answer": str(row.get("gold_answer", "")),
                "correct": row.get("correct"),
                "convergence_step": row.get("convergence_step"),
                "convergence_index": row.get("convergence_index"),
                "convergence_fraction": (
                    float(row["convergence_fraction"])
                    if row.get("convergence_fraction") is not None
                    else None
                ),
                "flip_rate": float(row.get("flip_rate", 0.0)),
                "consistency": float(row.get("consistency", 1.0)),
                "longest_stable_run": int(row.get("longest_stable_run", 0)),
                "answer_commitment_step": row.get("answer_commitment_step"),
                "stability_commitment_lag": row.get("stability_commitment_lag"),
                "unwritten_reasoning_at_convergence": (
                    float(row["unwritten_reasoning_at_convergence"])
                    if row.get("unwritten_reasoning_at_convergence") is not None
                    else None
                ),
                "order_gap": (
                    float(row["order_gap"])
                    if row.get("order_gap") is not None
                    else None
                ),
                "early_factor": (
                    float(row["early_factor"])
                    if (is_dream and row.get("early_factor") is not None)
                    else None
                ),
                "post_change_rate": (
                    float(row["post_change_rate"])
                    if (is_dream and row.get("post_change_rate") is not None)
                    else None
                ),
                "length_growth": (
                    float(row["length_growth"])
                    if (is_dream and row.get("length_growth") is not None)
                    else None
                ),
                "rationalization_score": (
                    float(row["rationalization_score"])
                    if (is_dream and row.get("rationalization_score") is not None)
                    else None
                ),
                "gated_rationalization_score": (
                    float(row["gated_rationalization_score"])
                    if (is_dream and row.get("gated_rationalization_score") is not None)
                    else None
                ),
                "n_steps": int(row.get("n_steps", 256)),
            }
            lines.append(json.dumps(clean_row, ensure_ascii=False))
        out_jsonl.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return staging_dir


def _find_hf_python() -> str | None:
    for candidate in (
        "/home/achinda1/miniconda3/bin/python3",
        shutil.which("python3"),
    ):
        if candidate and Path(candidate).exists():
            return str(candidate)
    return None


def push_staged_repo(
    local_dir: Path,
    repo_id: str,
    repo_type: str,
    token: str | None = None,
) -> None:
    """Create (if needed) and upload `local_dir` to Hugging Face `repo_id` via `HfApi`."""
    try:
        from huggingface_hub import HfApi  # type: ignore[import-not-found]

        api = HfApi(token=token)
        api.create_repo(
            repo_id=repo_id,
            repo_type=repo_type,
            exist_ok=True,
            space_sdk="static" if repo_type == "space" else None,
        )
        api.upload_folder(
            repo_id=repo_id,
            repo_type=repo_type,
            folder_path=str(local_dir),
            commit_message=f"Sync traj-eval {repo_type} artifacts",
        )
        return
    except ImportError:
        pass

    hf_py = _find_hf_python()
    if hf_py is not None:
        env = dict(os.environ)
        if token:
            env["HF_TOKEN"] = token
        inline_code = (
            "import os, sys; "
            "from huggingface_hub import HfApi; "
            "repo_id, repo_type, folder_path = sys.argv[1], sys.argv[2], sys.argv[3]; "
            "tok = os.environ.get('HF_TOKEN') or None; "
            "api = HfApi(token=tok); "
            "api.create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True, "
            "space_sdk='static' if repo_type == 'space' else None); "
            "api.upload_folder(repo_id=repo_id, repo_type=repo_type, folder_path=folder_path, "
            "commit_message=f'Sync traj-eval {repo_type} artifacts')"
        )
        subprocess.run(
            [hf_py, "-c", inline_code, repo_id, repo_type, str(local_dir)],
            check=True,
            env=env,
        )
        return

    raise RuntimeError(
        "Could not find `huggingface_hub` to upload to Hugging Face Hub. Install `huggingface_hub` to use `--push`."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stage and publish traj-eval Hugging Face Static Space and Dataset."
    )
    parser.add_argument(
        "--namespace",
        default=DEFAULT_HF_NAMESPACE,
        help=f"Hugging Face username or organization (default: {DEFAULT_HF_NAMESPACE})",
    )
    parser.add_argument(
        "--space-name",
        default="traj-eval",
        help="Hugging Face Static Space repo name (default: traj-eval)",
    )
    parser.add_argument(
        "--dataset-name",
        default="traj-eval-benchmarks",
        help="Hugging Face Dataset repo name (default: traj-eval-benchmarks)",
    )
    parser.add_argument(
        "--staging-root",
        default=str(REPO_ROOT / "build" / "hf_staging"),
        help="Directory to stage HF Space and Dataset bundles",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only stage Space and Dataset folders locally without pushing",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Create repos if needed and upload staged folders to Hugging Face Hub",
    )
    args = parser.parse_args(argv)

    staging_root = Path(args.staging_root)
    sp_dir = stage_hf_space_dir(staging_root / "space")
    ds_dir = stage_hf_dataset_dir(staging_root / "dataset")

    space_repo = f"{args.namespace}/{args.space_name}"
    dataset_repo = f"{args.namespace}/{args.dataset_name}"

    print(f"Staged HF Space   ({space_repo}) -> {sp_dir}")
    print(f"Staged HF Dataset ({dataset_repo}) -> {ds_dir}")

    if args.push and not args.dry_run:
        token = resolve_hf_token()
        push_staged_repo(sp_dir, space_repo, repo_type="space", token=token)
        push_staged_repo(ds_dir, dataset_repo, repo_type="dataset", token=token)
        print(f"Published Space:   https://huggingface.co/spaces/{space_repo}")
        print(f"Published Dataset: https://huggingface.co/datasets/{dataset_repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
