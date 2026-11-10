"""Adversarial red-team corpus and 10,000-trajectory fuzz tests (Issue #34 Part 3).

Covers:
- All verdicts `None`; verdict never settles; flips every step; flips only at last step.
- Single-step trajectories; empty trajectory list; empty file; header/whitespace-only file.
- Malformed JSON lines, truncated lines, missing fields, wrong types.
- Duplicate trajectory IDs; out-of-order step numbers; duplicate step numbers; negative step numbers.
- Verdicts outside A-D; missing/empty `gold_answer`; `gold_answer` that never appears.
- Non-ASCII, emoji, and >1 MB single-step `reasoning_text`.
- `NaN` and `Infinity` in numeric fields; huge step numbers in metadata.
- CLI (`run`, `compare`, `demo`) on nonexistent paths, malformed files, and directories.
- Adapter input validation with stubs/fakes only (zero weights).
- 10,000-trajectory seeded fuzz generator scored end-to-end with zero crashes and all invariants holding.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import pytest

from tests.test_stress import assert_all_metric_invariants
from traj_eval.adapters.dream import (
    DreamSmokeConfig,
    extract_gsm8k_verdict,
    extract_verdict,
    history_to_trajectory,
    run_dream_arcc_smoke,
)
from traj_eval.adapters.llada import (
     compute_transfer_schedule,
    llada_diffusion_generate,
    run_llada_smoke,
)
from traj_eval.cli import main as cli_main
from traj_eval.io import load_trajectories, trajectories_to_dicts
from traj_eval.metrics import (
    answer_commitment_step,
    compare_conditions,
    evaluate_trajectories,
    flip_rate,
    order_gap,
    paired_bootstrap_delta,
    rationalization_components,
    step_consistency,
    unwritten_reasoning_at_convergence,
    verdict_convergence_step,
)
from traj_eval.types import Trajectory, TrajectoryStep


def test_adversarial_verdict_patterns() -> None:
    # 1. All verdicts None.
    all_none = Trajectory(
        id="all-none",
        prompt="Q?",
        gold_answer="A",
        steps=[TrajectoryStep(step=i, prediction=None, reasoning_text="") for i in range(8)],
    )
    assert verdict_convergence_step(all_none) is None
    assert flip_rate(all_none) == 0.0
    assert all_none.final_prediction is None
    assert all_none.is_correct is None

    # 2. Verdict flips at every step and never settles until termination.
    flip_every = Trajectory(
        id="flip-every",
        prompt="Q?",
        gold_answer="B",
        steps=[
            TrajectoryStep(step=i, prediction=("A" if i % 2 == 0 else "B"), reasoning_text=f"s{i}")
            for i in range(8)
        ],
    )
    assert flip_rate(flip_every) == 1.0
    assert verdict_convergence_step(flip_every) == 7
    assert step_consistency(flip_every)["longest_stable_run"] == 1

    # 3. Verdict flips only at the very last step.
    flip_last = Trajectory(
        id="flip-last",
        prompt="Q?",
        gold_answer="D",
        steps=[
            TrajectoryStep(step=i, prediction=("A" if i < 7 else "D"), reasoning_text=f"s{i}")
            for i in range(8)
        ],
    )
    assert verdict_convergence_step(flip_last) == 7
    assert flip_rate(flip_last) == pytest.approx(1.0 / 7.0)
    assert rationalization_components(flip_last)["gated_score"] == 0.0

    # 4. Single-step trajectory and empty trajectory list.
    single = Trajectory(
        id="single",
        prompt="Q?",
        gold_answer="A",
        steps=[TrajectoryStep(step=0, prediction="A", reasoning_text="Answer: A", answer_committed=True)],
    )
    assert verdict_convergence_step(single) == 0
    assert answer_commitment_step(single) == 0
    assert flip_rate(single) == 0.0

    assert_all_metric_invariants([all_none, flip_every, flip_last, single])
    empty_eval = evaluate_trajectories([])
    assert empty_eval["aggregate"]["n_trajectories"] == 0


def test_adversarial_files_and_malformed_json(tmp_path: Path) -> None:
    # Empty file and whitespace/header-only file.
    empty_p = tmp_path / "empty.jsonl"
    empty_p.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_trajectories(empty_p)

    header_only = tmp_path / "header_only.jsonl"
    header_only.write_text("# comment header\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        load_trajectories(header_only)

    # Truncated JSON line, missing fields, non-list steps, non-integer step.
    truncated = tmp_path / "truncated.jsonl"
    truncated.write_text('{"id": "q1", "prompt": "p", "gold_answer": "A", "steps": [{"step": 0', encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        load_trajectories(truncated)

    bad_steps_type = tmp_path / "bad_steps.jsonl"
    bad_steps_type.write_text('{"id": "q1", "prompt": "p", "gold_answer": "A", "steps": "not-a-list"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="must be a list"):
        load_trajectories(bad_steps_type)

    bad_step_num = tmp_path / "bad_step_num.jsonl"
    bad_step_num.write_text(
        '{"id": "q1", "prompt": "p", "gold_answer": "A", "steps": [{"step": "abc", "prediction": "A"}]}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="non-integer 'step'"):
        load_trajectories(bad_step_num)


def test_adversarial_step_ordering_verdicts_outside_ad_and_1mb_unicode_text() -> None:
    # Out-of-order steps are sorted monotonically by Trajectory.__post_init__.
    out_of_order = Trajectory(
        id="ooo",
        prompt="Q?",
        gold_answer="Z_OUTSIDE_AD",
        steps=[
            TrajectoryStep(step=2, prediction="Z_OUTSIDE_AD", reasoning_text="final"),
            TrajectoryStep(step=0, prediction=None, reasoning_text="start"),
            TrajectoryStep(step=1, prediction="42", reasoning_text="mid"),
        ],
    )
    assert [s.step for s in out_of_order.steps] == [0, 1, 2]
    assert out_of_order.final_prediction == "Z_OUTSIDE_AD"
    assert out_of_order.is_correct is True
    assert verdict_convergence_step(out_of_order) == 2

    # Missing/empty gold_answer and gold_answer that never appears.
    no_gold = Trajectory(
        id="no-gold",
        prompt="Q?",
        gold_answer="",
        steps=[TrajectoryStep(step=0, prediction="A", reasoning_text="r")],
    )
    never_gold = Trajectory(
        id="never-gold",
        prompt="Q?",
        gold_answer="NEVER_APPEARS",
        steps=[TrajectoryStep(step=0, prediction="A", reasoning_text="r")],
    )
    assert no_gold.is_correct is None
    assert never_gold.is_correct is False

    # Non-ASCII, emoji, and >1 MB single-step reasoning_text.
    one_mb_text = "Δ reasoning 🚀 café Начало " * 45000  # > 1.1 MB
    assert len(one_mb_text.encode("utf-8")) > 1_000_000
    huge_text_traj = Trajectory(
        id="huge-1mb",
        prompt="Unicode prompt ∑ ∫ Ω 🧠",
        gold_answer="B",
        steps=[
            TrajectoryStep(step=0, prediction="B", reasoning_text="Answer: B"),
            TrajectoryStep(step=1, prediction="B", reasoning_text=one_mb_text),
        ],
    )
    ev = evaluate_trajectories([out_of_order, no_gold, never_gold, huge_text_traj])
    assert ev["aggregate"]["n_trajectories"] == 4
    assert ev["per_trajectory"][3]["length_growth"] == 1.0


def test_documented_red_team_findings_linked_to_issues(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Pin the red-team findings filed in Issues #37, #39, and #40."""
    # Issue #37 finding 1: directory input to load_trajectories / CLI raises IsADirectoryError.
    with pytest.raises(IsADirectoryError):
        load_trajectories(tmp_path)

    # Issue #37 finding 2: Infinity in JSON step raises OverflowError.
    inf_step_file = tmp_path / "inf_step.jsonl"
    inf_step_file.write_text(
        '{"id":"q1","prompt":"p","gold_answer":"A","steps":[{"step":Infinity,"prediction":"A"}]}\n',
        encoding="utf-8",
    )
    with pytest.raises(OverflowError):
        load_trajectories(inf_step_file)

    # Issue #37 finding 3: NaN in committed_reasoning_fraction and string "false" in answer_committed.
    nan_crf_file = tmp_path / "nan_crf.jsonl"
    nan_crf_file.write_text(
        '{"id":"q1","prompt":"p","gold_answer":"A","steps":[{"step":0,"prediction":42,"answer_committed":"false","committed_reasoning_fraction":NaN}]}\n',
        encoding="utf-8",
    )
    loaded_nan = load_trajectories(nan_crf_file)
    assert loaded_nan[0].steps[0].prediction == "42"
    assert loaded_nan[0].steps[0].answer_committed is True  # bool("false") == True (Issue #37)
    assert math.isnan(loaded_nan[0].steps[0].committed_reasoning_fraction)
    assert unwritten_reasoning_at_convergence(loaded_nan[0]) == 1.0
    assert order_gap(loaded_nan[0]) is None

    # Issue #37 finding 5: negative step numbers and huge step metadata return raw .step outside [0, n_steps).
    neg_traj = Trajectory(
        id="neg",
        prompt="p",
        gold_answer="A",
        steps=[
            TrajectoryStep(step=-5, prediction="A", answer_committed=True),
            TrajectoryStep(step=10**9, prediction="A", answer_committed=True),
        ],
    )
    assert verdict_convergence_step(neg_traj) == -5
    assert answer_commitment_step(neg_traj) == -5

    # Issue #39 finding: duplicate trajectory IDs overwrite earlier entries in paired_bootstrap_delta.
    dup_base = [
        Trajectory(id="dup-1", prompt="p", gold_answer="A", steps=[TrajectoryStep(step=0, prediction="A")]),
        Trajectory(id="dup-1", prompt="p", gold_answer="A", steps=[TrajectoryStep(step=0, prediction="B")]),
    ]
    dup_target = [
        Trajectory(id="dup-1", prompt="p", gold_answer="A", steps=[TrajectoryStep(step=0, prediction="A")]),
        Trajectory(id="dup-1", prompt="p", gold_answer="A", steps=[TrajectoryStep(step=0, prediction="A")]),
    ]
    boot_dup = paired_bootstrap_delta(dup_base, dup_target, metric="accuracy", n_boot=50, seed=0)
    assert boot_dup["n_paired"] == 1


def test_cli_adversarial_exit_codes_and_error_messages(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Nonexistent input file -> nonzero exit code (2) + single-line human-readable error on stderr.
    rc_missing = cli_main(["run", "--input", str(tmp_path / "nonexistent.jsonl")])
    err_missing = capsys.readouterr().err.strip()
    assert rc_missing != 0
    assert "\n" not in err_missing
    assert "Error:" in err_missing

    # Malformed JSON file on `run` and `compare` -> nonzero exit code (2) + single-line error.
    bad_file = tmp_path / "bad.jsonl"
    bad_file.write_text("{corrupted json\n", encoding="utf-8")
    rc_bad_run = cli_main(["run", "--input", str(bad_file)])
    err_bad_run = capsys.readouterr().err.strip()
    assert rc_bad_run != 0
    assert "\n" not in err_bad_run

    rc_bad_cmp = cli_main(["compare", "--base", str(bad_file), "--target", str(bad_file)])
    err_bad_cmp = capsys.readouterr().err.strip()
    assert rc_bad_cmp != 0
    assert "\n" not in err_bad_cmp

    # `demo` with 0 items per archetype completes cleanly with exit code 0.
    rc_demo_zero = cli_main(["demo", "--n-per-archetype", "0"])
    capsys.readouterr()
    assert rc_demo_zero == 0


def test_adapter_input_validation_with_stubs(tmp_path: Path) -> None:
    # LLaDA transfer schedule validation.
    with pytest.raises(ValueError, match="steps_per_block must be > 0"):
        compute_transfer_schedule(256, 0)
    with pytest.raises(ValueError, match="block_length must be >= 0"):
        compute_transfer_schedule(-1, 4)

    # Extractors handle hostile strings cleanly without crashing.
    for hostile_text in ("", "   ", "\x00\xff", "#### NaN", "\\boxed{}", "Answer: E", "Option Z: Correct"):
        assert extract_verdict(hostile_text) in (None, "A", "B", "C", "D")
        assert extract_gsm8k_verdict(hostile_text) is None or isinstance(
            extract_gsm8k_verdict(hostile_text), str
        )

    # Empty history in history_to_trajectory returns an empty Trajectory cleanly.
    class _StubTok:
        mask_token_id = 999

        def decode(self, ids, skip_special_tokens=True):
            return ""

    empty_traj = history_to_trajectory(
        traj_id="empty-hist",
        prompt="Q?",
        gold_answer="A",
        history=[],
        tokenizer=_StubTok(),
        prompt_len=1,
        mask_token_id=999,
    )
    assert empty_traj.steps == []

    # Issue #40 finding: malformed item dict missing required keys raises KeyError in run_dream_arcc_smoke / run_llada_smoke.
    class _StubModel:
        def diffusion_generate(self, *args, **kwargs):
            class _Out:
                sequences = [1, 2]
                history = [[1, 2]]

            return _Out()

    cfg = DreamSmokeConfig(
        steps=1,
        max_new_tokens=1,
        out_jsonl=str(tmp_path / "out.jsonl"),
        out_ids=str(tmp_path / "ids.json"),
        out_log=str(tmp_path / "log.json"),
    )
    with pytest.raises(KeyError):
        run_dream_arcc_smoke([{"prompt": "missing id"}], cfg, model=_StubModel(), tokenizer=_StubTok(), device="cpu")
    with pytest.raises(KeyError):
        run_llada_smoke([{"prompt": "missing id"}], model=_StubModel(), tokenizer=_StubTok(), device="cpu")


def test_fuzz_10000_random_trajectories_zero_crashes() -> None:
    """Generate and score 10,000 seeded random adversarial trajectories end-to-end."""
    rng = random.Random(20261110)
    verdict_pool = [None, "A", "B", "C", "D", "E", "42", "-3.5", ""]
    gold_pool = ["A", "B", "C", "D", "42", ""]
    text_pool = [
        "",
        "Answer: A",
        "Step 1: 🤔 café Δx -> B",
        "Longer rationale with multiple clauses and numbers 1, 2, 3.",
    ]

    trajs: list[Trajectory] = []
    for i in range(10_000):
        n_steps = rng.randint(0, 8)
        shape = i % 6
        steps: list[TrajectoryStep] = []
        commit_from = rng.randint(0, max(1, n_steps))
        for s in range(n_steps):
            if shape == 0:
                pred = None
            elif shape == 1:
                pred = "A" if s % 2 == 0 else "B"
            elif shape == 2:
                pred = "A" if s < n_steps - 1 else "D"
            else:
                pred = rng.choice(verdict_pool)
            ans_comm: bool | None
            if shape == 3:
                ans_comm = None
                crf = None
            else:
                ans_comm = s >= commit_from
                crf = round(s / max(1, n_steps), 4)
            steps.append(
                TrajectoryStep(
                    step=s,
                    prediction=pred,
                    reasoning_text=rng.choice(text_pool),
                    confidence=round(rng.random(), 4) if shape != 4 else None,
                    answer_committed=ans_comm,
                    committed_reasoning_fraction=crf,
                )
            )
        trajs.append(
            Trajectory(
                id=f"fuzz-{i}",
                prompt=f"Fuzz prompt {i % 200}",
                gold_answer=rng.choice(gold_pool),
                steps=steps,
            )
        )

    assert len(trajs) == 10_000
    assert_all_metric_invariants(trajs)

    # Also verify compare_conditions on two 5,000-item halves of the 10,000 fuzz corpus.
    half_a = trajs[:5000]
    half_b = [
        Trajectory(id=f"fuzz-{i}", prompt=t.prompt, gold_answer=t.gold_answer, steps=t.steps)
        for i, t in enumerate(trajs[5000:])
    ]
    cmp_res = compare_conditions(half_a, half_b, n_boot=50, seed=42)
    assert cmp_res["base_aggregate"]["n_trajectories"] == 5000
    assert cmp_res["target_aggregate"]["n_trajectories"] == 5000
