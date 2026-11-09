"""Unit tests for the static project page (`site/`) per Issue #29."""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SITE_DIR = REPO_ROOT / "site"
RESULTS_DIR = REPO_ROOT / "results"


def _extract_embedded_page_data() -> dict:
    html = (SITE_DIR / "index.html").read_text(encoding="utf-8")
    match = re.search(
        r'<script\s+id="page-data"\s+type="application/json">\s*(\{.*?\})\s*</script>',
        html,
        flags=re.DOTALL,
    )
    assert match is not None, "Missing <script id='page-data' type='application/json'> in site/index.html"
    return json.loads(match.group(1))


def test_site_files_exist_and_have_no_em_dashes_or_vendor_headers() -> None:
    for filename in ("index.html", "styles.css", "app.js"):
        path = SITE_DIR / filename
        assert path.exists(), f"Missing {path}"
        content = path.read_text(encoding="utf-8")
        assert "\u2014" not in content, f"Em dash found in {filename}"
        assert "&mdash;" not in content, f"&mdash; entity found in {filename}"
        assert "&#8212;" not in content, f"&#8212; entity found in {filename}"
        assert "Lenovo" not in content, f"Vendor header found in {filename}"
        assert "Confidential" not in content, f"Confidential header found in {filename}"


def test_site_respects_issue_29_hard_exclusions_and_required_sections() -> None:
    combined = "\n".join(
        (SITE_DIR / name).read_text(encoding="utf-8")
        for name in ("index.html", "styles.css", "app.js")
    )
    forbidden_terms = (
        "SBW",
        "sbw_lockin",
        "CoTsel",
        "cotsel",
        "Pass@k",
        "pass_at_k",
        "StepRS",
        "step_rs",
        "ForkGRPO",
        "fork_only",
        "DTCA",
        "dtca",
        "GRPO",
        "smoke5",
        "0.9818",
        "0.8732",
    )
    for term in forbidden_terms:
        assert term not in combined, f"Forbidden term {term!r} found in site/"

    html = (SITE_DIR / "index.html").read_text(encoding="utf-8")
    assert "Measure reasoning in the denoising trajectory, not just final-answer accuracy." in html
    assert "not a paired comparison on the same item" in html
    assert "Descriptive, not causal proof" in html
    assert "Five-item runs are pipeline checks only" in html
    assert "MMaDA-8B-MixCoT" in html
    assert "Paper status: in preparation" in html


def test_site_embedded_results_match_master_scored_jsons() -> None:
    data = _extract_embedded_page_data()
    rows = data["results_64"]
    assert len(rows) >= 7
    for row in rows:
        assert row["n"] == 64, f"Non-64-item suite found in results_64: {row}"
        src_path = REPO_ROOT / row["source_file"]
        assert src_path.exists(), f"Missing source file {src_path}"
        scored = json.loads(src_path.read_text(encoding="utf-8"))["aggregate"]
        assert scored["n_trajectories"] == 64
        assert round(float(scored["accuracy"]), 4) == round(float(row["accuracy"]), 4)
        assert round(float(scored["coverage"]), 4) == round(float(row["coverage"]), 4)
        assert round(float(scored["conditional_accuracy"]), 4) == round(
            float(row["conditional_accuracy"]), 4
        )
        assert round(float(scored["mean_convergence_step"]), 1) == round(
            float(row["mean_convergence_step"]), 1
        )

    rat = data["dream_rationalization_side_by_side"]
    comp_src = json.loads((REPO_ROOT / rat["source_file"]).read_text(encoding="utf-8"))
    sbs = comp_src["rationalization_components_side_by_side"]
    key_map = {
        "early_factor": "early_factor",
        "post_change_rate": "post_change_rate",
        "length_growth": "length_growth",
        "gated": "gated_score",
    }
    for item in rat["components"]:
        src_entry = sbs[key_map[item["key"]]]
        assert round(float(item["direct"]), 4) == round(float(src_entry["direct_64"]), 4)
        assert round(float(item["reasoning"]), 4) == round(float(src_entry["reasoning_64"]), 4)


def test_site_embedded_trajectories_match_sampled_steps_in_jsonl() -> None:
    data = _extract_embedded_page_data()
    tcomp = data["trajectory_comparison"]
    sampled_steps = tcomp["sampled_steps"]
    assert sampled_steps == [0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255]

    for key, expected_id in (("reasoning", "Mercury_LBS10610"), ("direct", "Mercury_7175875")):
        entry = tcomp[key]
        assert entry["id"] == expected_id
        jsonl_path = REPO_ROOT / entry["source_file"]
        matched_row = None
        with jsonl_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                if row["id"] == expected_id:
                    matched_row = row
                    break
        assert matched_row is not None, f"Could not find {expected_id} in {jsonl_path}"
        steps_by_num = {int(s["step"]): s["prediction"] for s in matched_row["steps"]}
        actual_verdicts = [steps_by_num[st] for st in sampled_steps]
        assert entry["verdicts"] == actual_verdicts
