"""Tests for the toy GRPO estimator."""

import numpy as np
import pytest

from experiments.grpo_minimal import (
    clipped_surrogate,
    group_advantages,
    sampled_kl,
    train_matching,
)


def test_normalized_advantage_is_zero_mean_unit_std():
    rewards = np.array([[0.0, 1.0, 0.0, 1.0]])
    adv = group_advantages(rewards, normalize=True)
    assert adv.mean() == pytest.approx(0.0)
    assert adv.std() == pytest.approx(1.0)


def test_constant_group_has_zero_advantage():
    rewards = np.ones((4, 8))
    assert np.allclose(group_advantages(rewards, normalize=True), 0.0)
    assert np.allclose(group_advantages(rewards, normalize=False), 0.0)


def test_unnormalized_advantage_is_centered_reward():
    rewards = np.array([0.0, 0.0, 1.0, 1.0])
    adv = group_advantages(rewards, normalize=False)
    assert np.allclose(adv, [-0.5, -0.5, 0.5, 0.5])


def test_clipped_surrogate_caps_positive_advantage():
    # ratio 2.0 is outside the clip window. Positive A keeps the clipped value.
    out = clipped_surrogate(np.array([2.0]), np.array([1.0]), clip_eps=0.2)
    assert out[0] == pytest.approx(1.2)


def test_sampled_kl_is_zero_when_policies_match():
    p = np.array([0.3, 0.7])
    assert np.allclose(sampled_kl(p, p), 0.0)


def test_matching_grpo_beats_chance():
    # Six contexts, chance is 1/6. Outcome reward only; no value head.
    stats = train_matching(steps=40, batch_size=32, group_size=8, n_ctx=6, seed=0)
    assert stats["greedy_accuracy"] == 1.0
