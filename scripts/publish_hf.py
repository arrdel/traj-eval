"""Stage and publish `traj-eval` Dataset and Static Space to Hugging Face Hub.

Supports:
1. `--dry-run`: Stages both the Hugging Face Dataset folder and Hugging Face
   Static Space folder locally under `build/hf_staging/` so they can be
   inspected or pushed manually.
2. Live push via `huggingface_hub` (`HfApi`) or the `hf` CLI (`--push`).

Usage examples:
  # Stage locally and inspect without network calls
  python scripts/publish_hf.py --namespace arrdel --dry-run

  # Authenticate and push both Dataset and Static Space to Hugging Face
  HF_TOKEN=hf_xxx python scripts/publish_hf.py --namespace arrdel --push
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.build_project_site import (
    HF_DIR,
    REPO_ROOT,
    RESULTS_DIR,
    SITE_DIR,
    write_all_artifacts,
)


def stage_hf_dataset_dir(
    staging_dir: Path,
    results_dir: Path = RESULTS_DIR,
    hf_dir: Path = HF_DIR,
) -> Path:
    """Populate a clean directory ready to upload as a Hugging Face Dataset repo."""
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    dataset_readme = hf_dir / "dataset_README.md"
    if not dataset_readme.exists():
        write_all_artifacts(results_dir=results_dir, hf_dir=hf_dir)
    shutil.copy2(dataset_readme, staging_dir / "README.md")

    out_results = staging_dir / "results"
    out_results.mkdir(parents=True, exist_ok=True)

    for pattern in ("*.jsonl", "*_results.json", "*_run_log.json", "*_vs_*.json", "*_input.json", "baselines.md"):
        for src in sorted(results_dir.glob(pattern)):
            shutil.copy2(src, out_results / src.name)

    return staging_dir


def stage_hf_space_dir(
    staging_dir: Path,
    site_dir: Path = SITE_DIR,
    hf_dir: Path = HF_DIR,
    results_dir: Path = RESULTS_DIR,
) -> Path:
    """Populate a clean directory ready to upload as a Hugging Face Static Space."""
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    space_readme = hf_dir / "space_README.md"
    summary_json = site_dir / "data" / "summary.json"
    if not space_readme.exists() or not summary_json.exists():
        write_all_artifacts(
            results_dir=results_dir,
            site_data_dir=site_dir / "data",
            hf_dir=hf_dir,
        )

    shutil.copy2(space_readme, staging_dir / "README.md")
    for asset in ("index.html", "styles.css", "app.js"):
        shutil.copy2(site_dir / asset, staging_dir / asset)

    shutil.copytree(site_dir / "data", staging_dir / "data", dirs_exist_ok=True)
    return staging_dir


def _find_hf_cli() -> str:
    for candidate in (
        shutil.which("hf"),
        "/home/achinda1/miniconda3/bin/hf",
    ):
        if candidate and Path(candidate).exists():
            return str(candidate)
    raise RuntimeError(
        "Could not find `hf` CLI. Install `huggingface_hub` or ensure `/home/achinda1/miniconda3/bin/hf` is on PATH."
    )


def push_staged_repo(
    local_dir: Path,
    repo_id: str,
    repo_type: str,
    token: str | None = None,
) -> None:
    """Create (if needed) and upload `local_dir` to Hugging Face `repo_id`."""
    hf_bin = _find_hf_cli()
    env = dict(os.environ)
    if token:
        env["HF_TOKEN"] = token

    create_cmd = [
        hf_bin,
        "repo",
        "create",
        repo_id,
        "--type",
        repo_type,
        "--exist-ok",
    ]
    if repo_type == "space":
        create_cmd.extend(["--space-sdk", "static"])

    subprocess.run(create_cmd, check=True, env=env)

    upload_cmd = [
        hf_bin,
        "upload",
        repo_id,
        str(local_dir),
        ".",
        "--repo-type",
        repo_type,
        "--commit-message",
        f"Sync traj-eval {repo_type} artifacts",
    ]
    subprocess.run(upload_cmd, check=True, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage and publish traj-eval HF Dataset and Space.")
    parser.add_argument(
        "--namespace",
        default="arrdel",
        help="Hugging Face username or organization (default: arrdel)",
    )
    parser.add_argument(
        "--dataset-name",
        default="traj-eval-benchmarks",
        help="Hugging Face Dataset repo name (default: traj-eval-benchmarks)",
    )
    parser.add_argument(
        "--space-name",
        default="traj-eval",
        help="Hugging Face Space repo name (default: traj-eval)",
    )
    parser.add_argument(
        "--staging-root",
        default=str(REPO_ROOT / "build" / "hf_staging"),
        help="Directory to stage HF Dataset and Space bundles",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only build and stage Dataset and Space folders locally without pushing",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Create repos if needed and upload staged folders to Hugging Face Hub",
    )
    args = parser.parse_args()

    write_all_artifacts()

    staging_root = Path(args.staging_root)
    ds_dir = stage_hf_dataset_dir(staging_root / "dataset")
    sp_dir = stage_hf_space_dir(staging_root / "space")

    dataset_repo = f"{args.namespace}/{args.dataset_name}"
    space_repo = f"{args.namespace}/{args.space_name}"

    print(f"Staged HF Dataset ({dataset_repo}) -> {ds_dir}")
    print(f"Staged HF Space   ({space_repo}) -> {sp_dir}")

    if args.push and not args.dry_run:
        token = os.environ.get("HF_TOKEN")
        push_staged_repo(ds_dir, dataset_repo, repo_type="dataset", token=token)
        push_staged_repo(sp_dir, space_repo, repo_type="space", token=token)
        print(f"Published Dataset: https://huggingface.co/datasets/{dataset_repo}")
        print(f"Published Space:   https://huggingface.co/spaces/{space_repo}")


if __name__ == "__main__":
    main()
