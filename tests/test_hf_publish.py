"""Unit tests for the Hugging Face Space and Dataset cards and staging script."""

from __future__ import annotations

import json
import re
from pathlib import Path

from scripts.publish_hf import (
    HF_DIR,
    REPO_ROOT,
    SPACE_SITE_ASSETS,
    SUITES_64,
    main as publish_hf_main,
    stage_hf_dataset_dir,
    stage_hf_space_dir,
)


def test_hf_readmes_exist_have_valid_frontmatter_and_no_em_dashes_or_vendor_headers() -> None:
    space_md = (HF_DIR / "space_README.md").read_text(encoding="utf-8")
    dataset_md = (HF_DIR / "dataset_README.md").read_text(encoding="utf-8")

    assert space_md.startswith("---\n")
    assert "sdk: static" in space_md
    assert "app_file: index.html" in space_md
    assert "license: mit" in space_md
    assert "datasets:" in space_md
    assert "arrdel/traj-eval-benchmarks" in space_md
    assert "models:" in space_md
    assert "Dream-org/Dream-v0-Instruct-7B" in space_md

    assert dataset_md.startswith("---\n")
    assert "license: mit" in dataset_md
    assert "task_categories:" in dataset_md
    assert "configs:" in dataset_md
    assert "config_name: per_trajectory_metrics" in dataset_md
    assert "config_name: trajectories" in dataset_md

    for name, content in (("space_README.md", space_md), ("dataset_README.md", dataset_md)):
        assert "\u2014" not in content, f"Em dash found in {name}"
        assert "&mdash;" not in content, f"&mdash; found in {name}"
        assert "&#8212;" not in content, f"&#8212; found in {name}"
        assert "Lenovo" not in content, f"Vendor header found in {name}"
        assert "Confidential" not in content, f"Confidential header found in {name}"
        for forbidden in ("smoke5", "0.9818", "0.8732", "SBW", "CoTsel", "Pass@k", "StepRS", "ForkGRPO", "DTCA"):
            assert forbidden not in content, f"Forbidden term {forbidden!r} found in {name}"


def test_hf_dataset_readme_table_matches_master_scored_jsons() -> None:
    dataset_md = (HF_DIR / "dataset_README.md").read_text(encoding="utf-8")
    row_pattern = re.compile(
        r"^\|\s*`([^`]+)`\s*\|\s*`([^`]+)`\s*\|\s*`([^`]+)`\s*\|\s*(\d+)\s*\|\s*`([0-9.]+)`\s*\|\s*`([0-9.]+)`\s*\|\s*`([0-9.]+)`\s*\|\s*`([0-9.]+)`\s*\|[^|]+\|\s*`(results/[^`]+_results\.json)`\s*\|$",
        flags=re.MULTILINE,
    )
    matches = row_pattern.findall(dataset_md)
    assert len(matches) == 9, f"Expected 9 64-item rows in hf/dataset_README.md, got {len(matches)}"

    for _model, _task, _cond, n_str, acc_str, cov_str, cacc_str, conv_str, src_rel in matches:
        assert int(n_str) == 64
        src_path = REPO_ROOT / src_rel
        assert src_path.exists(), f"Missing source file {src_path}"
        agg = json.loads(src_path.read_text(encoding="utf-8"))["aggregate"]
        assert agg["n_trajectories"] == 64
        assert round(float(agg["accuracy"]), 4) == round(float(acc_str), 4)
        assert round(float(agg["coverage"]), 4) == round(float(cov_str), 4)
        assert round(float(agg["conditional_accuracy"]), 4) == round(float(cacc_str), 4)
        assert round(float(agg["mean_convergence_step"]), 1) == round(float(conv_str), 1)


def test_stage_hf_space_and_dataset_dry_run(tmp_path: Path) -> None:
    sp_dir = stage_hf_space_dir(tmp_path / "space")
    assert (sp_dir / "README.md").exists()
    for asset in SPACE_SITE_ASSETS:
        assert (sp_dir / asset).exists(), f"Missing {asset} in staged HF Space"

    ds_dir = stage_hf_dataset_dir(tmp_path / "dataset")
    assert (ds_dir / "README.md").exists()
    staged_results = list((ds_dir / "results").glob("*"))
    assert len(staged_results) > 0
    assert (ds_dir / "results" / "baselines.md").exists()
    for f in staged_results:
        if f.is_file():
            assert "smoke5" not in f.name, f"5-item smoke file leaked into dataset bundle: {f.name}"

    metrics_files = list((ds_dir / "results" / "metrics").glob("*_metrics.jsonl"))
    assert len(metrics_files) == len(SUITES_64)

    rc = publish_hf_main(["--staging-root", str(tmp_path / "cli_staging"), "--dry-run"])
    assert rc == 0
    assert (tmp_path / "cli_staging" / "space" / "index.html").exists()
    assert (tmp_path / "cli_staging" / "dataset" / "README.md").exists()
