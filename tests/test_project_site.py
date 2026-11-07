"""Unit tests for static project site builder and Hugging Face staging."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.build_project_site import (
    REPO_ROOT,
    build_bundle,
    compress_steps_lossless,
    pack_mask_runs,
    render_hf_dataset_card,
    render_hf_space_card,
    unpack_mask_runs,
    write_all_artifacts,
)
from scripts.publish_hf import stage_hf_dataset_dir, stage_hf_space_dir


def test_pack_and_unpack_mask_runs_roundtrip() -> None:
    raw = "Step 1: Let us think.<|mdm_mask|><|mdm_mask|><|mdm_mask|> Answer: <|mask|>"
    packed = pack_mask_runs(raw)
    assert packed == "Step 1: Let us think.<<M:3>> Answer: <<M:1>>"
    unpacked = unpack_mask_runs(packed, mask_token="<|mask|>")
    assert unpacked == "Step 1: Let us think.<|mask|><|mask|><|mask|> Answer: <|mask|>"


def test_compress_steps_lossless_keeps_state_transitions_and_final_step() -> None:
    steps = [
        {
            "step": 1,
            "prediction": None,
            "reasoning_text": "<|mask|><|mask|>",
            "confidence": 0.1,
            "answer_committed": False,
            "committed_reasoning_fraction": 0.0,
        },
        {
            "step": 2,
            "prediction": None,
            "reasoning_text": "<|mask|><|mask|>",
            "confidence": 0.1,
            "answer_committed": False,
            "committed_reasoning_fraction": 0.0,
        },
        {
            "step": 3,
            "prediction": "C",
            "reasoning_text": "Answer: C",
            "confidence": 0.95,
            "answer_committed": True,
            "committed_reasoning_fraction": 1.0,
        },
        {
            "step": 4,
            "prediction": "C",
            "reasoning_text": "Answer: C",
            "confidence": 0.99,
            "answer_committed": True,
            "committed_reasoning_fraction": 1.0,
        },
    ]
    keyframes = compress_steps_lossless(steps)
    assert [k["step"] for k in keyframes] == [1, 3, 4]
    assert keyframes[0]["reasoning_text"] == "<<M:2>>"
    assert keyframes[1]["prediction"] == "C"
    assert keyframes[1]["answer_committed"] is True


def test_build_bundle_and_hf_cards_include_all_64_item_suites() -> None:
    bundle = build_bundle()
    suite_ids = {s["suite_id"] for s in bundle["suites"]}
    expected_64 = {
        "dream_arcc_smoke64_direct",
        "dream_arcc_smoke64_reasoning",
        "llada_instruct_arcc_smoke64_direct",
        "llada_instruct_arcc_smoke64_reasoning",
        "llada_15_arcc_smoke64_direct",
        "llada_15_arcc_smoke64_reasoning",
        "dream_gsm8k_smoke64",
        "llada_instruct_gsm8k_smoke64",
        "llada_15_gsm8k_smoke64",
    }
    assert expected_64.issubset(suite_ids)

    ds_card = render_hf_dataset_card(bundle)
    assert "pretty_name: \"traj-eval: Masked Diffusion Reasoning Trajectory Benchmarks\"" in ds_card
    assert "llada_15_gsm8k_smoke64" in ds_card
    assert "dream_arcc_smoke64_reasoning" in ds_card

    sp_card = render_hf_space_card()
    assert "sdk: static" in sp_card
    assert "traj-eval Trajectory Explorer" in sp_card


def test_site_static_assets_and_staging_dirs(tmp_path: Path) -> None:
    write_all_artifacts()
    site_dir = REPO_ROOT / "site"
    assert (site_dir / "index.html").exists()
    assert (site_dir / "styles.css").exists()
    assert (site_dir / "app.js").exists()
    assert (site_dir / "data" / "summary.json").exists()

    summary = json.loads((site_dir / "data" / "summary.json").read_text(encoding="utf-8"))
    assert len(summary["suites"]) == 19
    assert len(summary["comparisons"]) == 4

    ds_stage = stage_hf_dataset_dir(tmp_path / "dataset")
    assert (ds_stage / "README.md").exists()
    assert (ds_stage / "results" / "llada_15_gsm8k_smoke64_results.json").exists()

    sp_stage = stage_hf_space_dir(tmp_path / "space")
    assert (sp_stage / "README.md").exists()
    assert (sp_stage / "index.html").exists()
    assert (sp_stage / "data" / "summary.json").exists()
