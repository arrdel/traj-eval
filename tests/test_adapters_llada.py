# Copyright (C) 2026 Lenovo
# All Rights Reserved.
# Lenovo Confidential Restricted.

"""Tests for LLaDA adapter (LLaDA-8B-Instruct and LLaDA-1.5).

No torch / transformers / weights. All model and tokenizer objects are stubs.
"""

from pathlib import Path

import pytest

from traj_eval.adapters.llada import (
    LLADA_15_ID,
    LLADA_INSTRUCT_ID,
    LLADA_MASK_TOKEN_ID,
    LLaDASmokeConfig,
    compute_transfer_schedule,
    llada_history_to_trajectory,
    run_llada_smoke,
)
from traj_eval.io import load_trajectories
from traj_eval.metrics import (
    answer_commitment_step,
    stability_commitment_lag,
    unwritten_reasoning_at_convergence,
    verdict_convergence_step,
)


class FakeLLaDATokenizer:
    # Real LLaDA tokenizer has mask_token_id = None, so the adapter must
    # rely on LLADA_MASK_TOKEN_ID = 126336 rather than tokenizer.mask_token_id.
    mask_token_id = None

    def __init__(self, vocab: dict[int, str]):
        self._vocab = vocab

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        return "".join(self._vocab.get(int(i), "") for i in ids)

    def apply_chat_template(self, messages, tokenize: bool = False, add_generation_prompt: bool = True) -> str:
        return messages[0]["content"]

    def __call__(self, text: str, return_tensors=None):
        return {"input_ids": [[1]], "attention_mask": [[1]]}


class FakeLLaDAOutput:
    def __init__(self, sequences, history):
        self.sequences = sequences
        self.history = history


class FakeLLaDAModel:
    def diffusion_generate(self, input_ids, **kwargs):
        steps = int(kwargs.get("steps", 4))
        mask_id = int(kwargs.get("mask_id", LLADA_MASK_TOKEN_ID))
        # Simulate answer-first commitment: answer token (30="Answer: C") commits
        # at step 1 while rationale tokens (20="Step 1. ", 21="Step 2. ") backfill
        # at steps 2 and 3.
        history = []
        for s in range(steps):
            if s == 0:
                history.append([1, mask_id, mask_id, mask_id])
            elif s == 1:
                history.append([1, mask_id, mask_id, 30])
            elif s == 2:
                history.append([1, 20, mask_id, 30])
            else:
                history.append([1, 20, 21, 30])
        return FakeLLaDAOutput(sequences=history[-1], history=history)


def test_compute_transfer_schedule():
    assert compute_transfer_schedule(256, 256) == [1] * 256
    sched = compute_transfer_schedule(10, 3)
    assert sched == [4, 3, 3]
    assert sum(sched) == 10
    with pytest.raises(ValueError):
        compute_transfer_schedule(10, 0)


def test_llada_history_to_trajectory_with_none_tokenizer_mask_id():
    tok = FakeLLaDATokenizer({1: "Q", 20: "Reason A. ", 21: "Reason B. ", 30: "Answer: C"})
    m = LLADA_MASK_TOKEN_ID
    history = [
        [1, m, m, m],
        [1, m, m, 30],
        [1, 20, m, 30],
        [1, 20, 21, 30],
    ]
    traj = llada_history_to_trajectory(
        traj_id="llada-q1",
        prompt="Question?",
        gold_answer="C",
        history=history,
        tokenizer=tok,
        prompt_len=1,
    )
    assert [s.prediction for s in traj.steps] == [None, "C", "C", "C"]
    assert verdict_convergence_step(traj) == 1
    assert answer_commitment_step(traj) == 1
    assert stability_commitment_lag(traj) == 0
    # At T_stab (step 1), rationale tokens 20 and 21 are both still masked -> 100% unwritten.
    assert unwritten_reasoning_at_convergence(traj) == pytest.approx(1.0)


def test_run_llada_smoke_with_stub_models(tmp_path: Path):
    tok = FakeLLaDATokenizer({1: "Q", 20: "Step 1. ", 21: "Step 2. ", 30: "Answer: C"})
    mdl = FakeLLaDAModel()
    items = [
        {"id": "q1", "prompt": "P1", "gold_answer": "C"},
        {"id": "q2", "prompt": "P2", "gold_answer": "C"},
    ]
    for model_id in (LLADA_INSTRUCT_ID, LLADA_15_ID):
        slug = model_id.split("/")[-1].lower()
        cfg = LLaDASmokeConfig(
            model_id=model_id,
            steps=4,
            gen_length=3,
            block_length=3,
            out_jsonl=str(tmp_path / f"{slug}.jsonl"),
            out_ids=str(tmp_path / f"{slug}_ids.json"),
            out_log=str(tmp_path / f"{slug}_log.json"),
        )
        log = run_llada_smoke(items, cfg, model=mdl, tokenizer=tok, device="cpu")
        assert log["n_trajectories"] == 2
        loaded = load_trajectories(cfg.out_jsonl)
        assert len(loaded) == 2
        assert loaded[0].final_prediction == "C"
        assert loaded[0].steps[1].answer_committed is True
        assert loaded[0].steps[1].committed_reasoning_fraction == 0.0
        assert loaded[0].steps[3].committed_reasoning_fraction == 1.0


def test_llada_config_serializes_frontier_width():
    cfg = LLaDASmokeConfig(frontier_width=16)
    d = cfg.to_dict()
    assert d["frontier_width"] == 16
