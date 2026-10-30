"""Tests for the toy GRPO estimator."""

import numpy as np
import pytest

from experiments.grpo_minimal import (
    clipped_surrogate,
    group_advantages,
    post_convergence_credit_fraction,
    sampled_kl,
    train_matching,
    trajectory_step_weights,
)
from traj_eval.types import Trajectory, TrajectoryStep


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


def test_trajectory_step_weights_and_post_convergence_credit():
    # Trajectory converges at step 1 (T_stab=1) and commits at step 2 (T_commit=2) out of 5 steps (0..4).
    steps = [
        TrajectoryStep(step=0, prediction=None, confidence=0.3, answer_committed=False),
        TrajectoryStep(step=1, prediction="B", confidence=0.8, answer_committed=False),
        TrajectoryStep(step=2, prediction="B", confidence=0.9, answer_committed=True),
        TrajectoryStep(step=3, prediction="B", confidence=0.9, answer_committed=True),
        TrajectoryStep(step=4, prediction="B", confidence=0.9, answer_committed=True),
    ]
    t = Trajectory(id="credit-1", prompt="q", gold_answer="B", steps=steps)

    # 1. Broadcast (diffu-GRPO): 3 of 5 steps are post-convergence -> 60% post-convergence credit.
    w_broad = trajectory_step_weights(t, scheme="broadcast")
    assert np.allclose(w_broad, [1.0, 1.0, 1.0, 1.0, 1.0])
    assert post_convergence_credit_fraction(t, scheme="broadcast") == pytest.approx(0.6)

    # 2. Pre-convergence (Paper 1 §6): 0% post-convergence credit.
    w_pre_conv = trajectory_step_weights(t, scheme="pre_convergence")
    assert np.allclose(w_pre_conv, [1.0, 1.0, 0.0, 0.0, 0.0])
    assert post_convergence_credit_fraction(t, scheme="pre_convergence") == 0.0

    # 3. Pre-commitment: active through step 2 (T_commit=2).
    w_pre_comm = trajectory_step_weights(t, scheme="pre_commitment")
    assert np.allclose(w_pre_comm, [1.0, 1.0, 1.0, 0.0, 0.0])

    # 4. StepRS-GRPO: linear half-decay from 1.0 to 0.5.
    w_step_rs = trajectory_step_weights(t, scheme="step_rs")
    assert w_step_rs[0] == pytest.approx(1.0)
    assert w_step_rs[-1] == pytest.approx(0.5)
    assert post_convergence_credit_fraction(t, scheme="step_rs") < post_convergence_credit_fraction(
        t, scheme="broadcast"
    )

    # 5. ForkGRPO (fork_only): only step 0 has confidence < 0.6.
    w_fork = trajectory_step_weights(t, scheme="fork_only", fork_threshold=0.6)
    assert np.allclose(w_fork, [1.0, 0.0, 0.0, 0.0, 0.0])
    assert post_convergence_credit_fraction(t, scheme="fork_only") == 0.0

    # 6. DTCA: positive confidence gains occur at steps 0->1 (+0.5) and 1->2 (+0.1), plus anchor at step 1.
    w_dtca = trajectory_step_weights(t, scheme="dtca")
    assert w_dtca[1] == pytest.approx(1.0)
    assert w_dtca[2] == 0.0
    assert w_dtca[3] == 0.0
    assert w_dtca[4] == 0.0
    assert post_convergence_credit_fraction(t, scheme="dtca") == 0.0
