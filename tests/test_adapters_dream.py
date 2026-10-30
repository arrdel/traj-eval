"""Tests for Dream adapter: verdict extraction, mask stripping, null handling.

No torch / transformers / weights. All model/tokenizer objects are fakes.
"""

import json

from traj_eval.adapters.dream import (
    decode_generation_region,
    extract_gsm8k_verdict,
    extract_verdict,
    history_to_trajectory,
    normalize_numeric_answer,
    strip_gsm8k_exemplars,
)
from traj_eval.io import load_trajectories, trajectories_to_dicts
from traj_eval.metrics import flip_rate, verdict_convergence_step
from traj_eval.types import Trajectory, TrajectoryStep


class FakeTokenizer:
    mask_token_id = 999

    def __init__(self, vocab):
        # vocab: token_id -> text piece
        self._vocab = vocab

    def decode(self, ids, skip_special_tokens=True):
        return "".join(self._vocab.get(int(i), "") for i in ids)


def test_extract_verdict_basic():
    assert extract_verdict("The answer is (B) because...") == "B"
    assert extract_verdict("Final answer: C") == "C"
    assert extract_verdict("Option D is correct") == "D"
    assert extract_verdict("I think B then C") == "C"  # last standalone wins
    assert extract_verdict("") is None
    assert extract_verdict("   ") is None
    assert extract_verdict("no letter here") is None


def test_extract_verdict_prefix_precedence():
    # Bare "A" as article should NOT beat an explicit answer prefix.
    assert extract_verdict("A long rationale. The answer is (B)") == "B"
    assert extract_verdict("Answer is A. Then B appears later as a letter") == "A"


def test_decode_strips_masks_and_prompt():
    tok = FakeTokenizer({1: "Q", 2: "A", 3: "B", 999: "<mask>"})
    # full = [prompt(1), gen(999, 2, 3)] -> prompt_len=1 -> gen [999,2,3] -> strip mask -> [2,3] -> "AB"
    assert decode_generation_region([1, 999, 2, 3], 1, tok) == "AB"
    # all masks -> ""
    assert decode_generation_region([1, 999, 999], 1, tok) == ""


def test_history_to_trajectory_null_then_letter():
    tok = FakeTokenizer({10: "P", 20: "Answer is (", 21: "B", 22: ")", 999: ""})
    # prompt_len=1. Step0: all masks -> "" -> None. Step1: "Answer is (B)" -> B.
    history = [
        [10, 999, 999],
        [10, 20, 21, 22],
    ]
    traj = history_to_trajectory(
        traj_id="q1",
        prompt="Q?",
        gold_answer="B",
        history=history,
        tokenizer=tok,
        prompt_len=1,
        mask_token_id=999,
    )
    assert len(traj.steps) == 2
    assert traj.steps[0].prediction is None
    assert traj.steps[1].prediction == "B"
    assert traj.final_prediction == "B"
    assert traj.is_correct is True


def test_null_not_a_flip():
    # [None, None, B, B, C] -> non-null [B,B,C] -> 1 flip / 2 = 0.5, NOT 3/4.
    steps = [
        TrajectoryStep(step=0, prediction=None, reasoning_text=""),
        TrajectoryStep(step=1, prediction=None, reasoning_text=""),
        TrajectoryStep(step=2, prediction="B", reasoning_text="b"),
        TrajectoryStep(step=3, prediction="B", reasoning_text="b"),
        TrajectoryStep(step=4, prediction="C", reasoning_text="c"),
    ]
    t = Trajectory(id="x", prompt="q", gold_answer="C", steps=steps)
    assert flip_rate(t) == 0.5
    # Convergence is first step of stable suffix of FINAL verdict C -> index 4.
    assert verdict_convergence_step(t) == 4
    # Same shape but final verdict B: [None, None, B, B, B] -> convergence 2.
    steps_b = [
        TrajectoryStep(step=0, prediction=None, reasoning_text=""),
        TrajectoryStep(step=1, prediction=None, reasoning_text=""),
        TrajectoryStep(step=2, prediction="B", reasoning_text="b"),
        TrajectoryStep(step=3, prediction="B", reasoning_text="b"),
        TrajectoryStep(step=4, prediction="B", reasoning_text="b"),
    ]
    tb = Trajectory(id="y", prompt="q", gold_answer="B", steps=steps_b)
    assert verdict_convergence_step(tb) == 2
    assert flip_rate(tb) == 0.0


def test_all_null_trajectory():
    steps = [
        TrajectoryStep(step=0, prediction=None, reasoning_text=""),
        TrajectoryStep(step=1, prediction=None, reasoning_text=""),
    ]
    t = Trajectory(id="u", prompt="q", gold_answer="B", steps=steps)
    assert verdict_convergence_step(t) is None
    assert flip_rate(t) == 0.0
    assert t.final_prediction is None
    assert t.is_correct is None


def test_io_roundtrip_preserves_null(tmp_path):
    traj = Trajectory(
        id="q1",
        prompt="Q?",
        gold_answer="B",
        steps=[
            TrajectoryStep(step=0, prediction=None, reasoning_text=""),
            TrajectoryStep(step=1, prediction="B", reasoning_text="Answer is (B)"),
        ],
    )
    p = tmp_path / "t.jsonl"
    p.write_text("\n".join(json.dumps(d) for d in trajectories_to_dicts([traj])) + "\n")
    # JSON must contain null, not "None".
    assert "null" in p.read_text()
    loaded = load_trajectories(p)
    assert loaded[0].steps[0].prediction is None
    assert loaded[0].steps[1].prediction == "B"


def test_history_to_trajectory_final_null_is_correct_none():
    tok = FakeTokenizer({10: "P", 999: ""})
    history = [[10, 999], [10, 999]]
    traj = history_to_trajectory(
        traj_id="q2",
        prompt="Q?",
        gold_answer="B",
        history=history,
        tokenizer=tok,
        prompt_len=1,
        mask_token_id=999,
    )
    assert traj.final_prediction is None
    assert traj.is_correct is None


def test_history_to_trajectory_tracks_answer_first_backfill():
    # Tokens: 10=prompt, 20="First reason. ", 21="Second reason. ", 30="Answer: ", 31="C", 999=mask
    tok = FakeTokenizer(
        {
            10: "P",
            20: "First reason. ",
            21: "Second reason. ",
            30: "Answer: ",
            31: "C",
            999: "",
        }
    )
    # Answer token (pos 3, tok 31) commits at step 1 while rationale tokens (pos 0, 1) commit at steps 2 and 3.
    history = [
        [10, 999, 999, 999, 999],
        [10, 999, 999, 30, 31],
        [10, 20, 999, 30, 31],
        [10, 20, 21, 30, 31],
    ]
    traj = history_to_trajectory(
        traj_id="af1",
        prompt="Q?",
        gold_answer="C",
        history=history,
        tokenizer=tok,
        prompt_len=1,
        mask_token_id=999,
    )
    assert [s.prediction for s in traj.steps] == [None, "C", "C", "C"]
    assert [s.answer_committed for s in traj.steps] == [False, True, True, True]
    # Rationale positions before ans_pos (index 3) are indices 0, 1, 2 (tokens 20, 21, 30).
    # At step 1: only token 30 (index 2) is committed -> 1/3 = 0.3333.
    # At step 2: tokens 20, 30 committed -> 2/3 = 0.6667.
    # At step 3: all 3 committed -> 1.0.
    assert [s.committed_reasoning_fraction for s in traj.steps] == [0.0, 0.3333, 0.6667, 1.0]


def test_gsm8k_numeric_normalization_and_extraction():
    assert normalize_numeric_answer("$1,200.00") == "1200"
    assert normalize_numeric_answer("-3.50") == "-3.5"
    assert normalize_numeric_answer("+42") == "42"

    # Intermediate operands without an answer cue must return None (never guess).
    assert extract_gsm8k_verdict("Janet has 16 eggs and eats 3 for breakfast.") is None
    # Explicit GSM8K hash marker, boxed answer, and answer prefix.
    assert extract_gsm8k_verdict("16 - 3 - 4 = 9. 9 * 2 = 18. #### 18") == "18"
    assert extract_gsm8k_verdict("So the total is \\boxed{1,250}.") == "1250"
    assert extract_gsm8k_verdict("After adding them up, the final answer is $72.00.") == "72"


def test_gsm8k_exemplar_stripping():
    # Echoed 2-shot exemplars followed by target response that has no verdict yet.
    echoed_partial = (
        "Question: What is 2 + 2?\nAnswer: 2 + 2 = 4. #### 4\n\n"
        "Question: What is 3 * 5?\nAnswer: 3 * 5 = 15. #### 15\n\n"
        "Question: Janet has 16 eggs...\nAnswer: She eats 3 and bakes 4"
    )
    assert extract_gsm8k_verdict(echoed_partial) is None

    # Target response completed, plus a hallucinated trailing Question turn.
    with_trailing_turn = (
        "She sells 9 * 2 = 18 dollars. #### 18\n\n"
        "Question: A train leaves at 3pm... #### 999"
    )
    assert strip_gsm8k_exemplars(with_trailing_turn) == "She sells 9 * 2 = 18 dollars. #### 18"
    assert extract_gsm8k_verdict(with_trailing_turn) == "18"
