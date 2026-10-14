"""Tests for IO: load/save round-trip and error handling."""

import json
from pathlib import Path

import pytest

from traj_eval.io import load_trajectories, save_results, trajectories_to_dicts
from traj_eval.synthetic import generate_synthetic_trajectories


def test_jsonl_roundtrip(tmp_path: Path):
    trajs = generate_synthetic_trajectories(n_per_archetype=2)
    p = tmp_path / "in.jsonl"
    p.write_text("\n".join(json.dumps(d) for d in trajectories_to_dicts(trajs)) + "\n")
    loaded = load_trajectories(p)
    assert len(loaded) == len(trajs)
    assert loaded[0].id == trajs[0].id
    assert loaded[0].steps[0].prediction == trajs[0].steps[0].prediction


def test_json_list_roundtrip(tmp_path: Path):
    trajs = generate_synthetic_trajectories(n_per_archetype=1)
    p = tmp_path / "in.json"
    p.write_text(json.dumps(trajectories_to_dicts(trajs)))
    loaded = load_trajectories(p)
    assert len(loaded) == 3


def test_json_wrapped_roundtrip(tmp_path: Path):
    trajs = generate_synthetic_trajectories(n_per_archetype=1)
    p = tmp_path / "wrapped.json"
    p.write_text(json.dumps({"trajectories": trajectories_to_dicts(trajs)}))
    assert len(load_trajectories(p)) == 3


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        load_trajectories("does-not-exist-12345.jsonl")


def test_empty_file(tmp_path: Path):
    p = tmp_path / "empty.jsonl"
    p.write_text("")
    with pytest.raises(ValueError, match="empty"):
        load_trajectories(p)


def test_bad_json_line(tmp_path: Path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"id": "x", "prompt": "q", "gold_answer": "B", "steps": []}\nnot json\n')
    with pytest.raises(ValueError, match="invalid JSON"):
        load_trajectories(p)


def test_missing_field(tmp_path: Path):
    p = tmp_path / "missing.jsonl"
    p.write_text('{"id": "x", "prompt": "q"}\n')
    with pytest.raises(ValueError, match="missing required"):
        load_trajectories(p)


def test_save_results(tmp_path: Path):
    out = tmp_path / "sub" / "results.json"
    saved = save_results({"aggregate": {"accuracy": 0.5}}, out)
    assert saved.exists()
    assert json.loads(saved.read_text())["aggregate"]["accuracy"] == 0.5
