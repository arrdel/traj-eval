"""Tests for the toy GRPO estimator."""

import numpy as np
import pytest

from experiments.grpo_minimal import (
    clipped_surrogate,
    dtca_pos_weights,
    dtca_timestep_score,
    dtca_token_credits,
    group_advantages,
    post_convergence_credit_fraction,
    sampled_kl,
    step_rs_advantage_multiplier,
    step_rs_schedule,
    train_matching,
    train_trajectory_credit,
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


def test_source_paper_equations_for_step_rs_fork_grpo_and_dtca():
    # 1. StepRS-GRPO (Yu et al. 2026, arXiv:2610.00661v3, Eq. 3.3, 3.6, and Lemma 3.1 Eq. 3.7):
    # Linear half-decay schedule b(u) = beta_min + (beta_max - beta_min) * u with beta_min = 0.5 * beta_max.
    u_grid = np.array([1.0, 0.75, 0.5, 0.25, 0.0])
    b_vals = step_rs_schedule(u_grid, beta_max=2.0, gamma=0.5)
    assert np.allclose(b_vals, [2.0, 1.75, 1.5, 1.25, 1.0])

    # Verify Lemma 3.1 Eq. 3.7 exact identity against Eq. 3.3 entropic advantage:
    # A_{qi}(beta) = (1 / beta) * (exp(beta * r_{qi}) / (G^{-1} sum_h exp(beta * r_{qh})) - 1)
    #              == c_beta(p_hat) * (r_{qi} - p_hat)
    rewards = np.array([1.0, 0.0, 1.0, 0.0, 0.0])  # G=5, solve_rate p_hat = 0.4
    p_hat = float(rewards.mean())
    centered = rewards - p_hat
    beta = 1.4
    entropic_adv = (1.0 / beta) * (np.exp(beta * rewards) / np.mean(np.exp(beta * rewards)) - 1.0)
    mult = float(step_rs_advantage_multiplier(beta, p_hat))
    expected_mult = (np.exp(beta) - 1.0) / (beta * (1.0 - p_hat + p_hat * np.exp(beta)))
    assert mult == pytest.approx(expected_mult)
    assert np.allclose(entropic_adv, mult * centered)

    # 2. ForkGRPO (Frkovic et al. 2026, arXiv:2609.39859v1, Algorithm 2 & Section 3.2):
    # Fallback step set F_i = {t : max_{k in M_{i,t}} c^k_{i,t} <= lambda}.
    # Check boundary equality c_t == lambda (0.6) is included in F_i while c_t > lambda is excluded.
    fork_traj = Trajectory(
        id="fork-boundary",
        prompt="q",
        gold_answer="A",
        steps=[
            TrajectoryStep(step=0, prediction=None, confidence=0.45),
            TrajectoryStep(step=1, prediction="A", confidence=0.60),  # == lambda -> fallback step
            TrajectoryStep(step=2, prediction="A", confidence=0.61),  # > lambda -> parallel step
            TrajectoryStep(step=3, prediction="A", confidence=0.95),  # > lambda -> parallel step
        ],
    )
    w_fork_b = trajectory_step_weights(fork_traj, scheme="fork_only", fork_threshold=0.60)
    assert np.allclose(w_fork_b, [1.0, 1.0, 0.0, 0.0])

    # 3. DTCA (Chen et al. 2026, arXiv:2610.35201v1, Section 3.1--3.3, Eq. 6, 7, 8):
    # Hand-built POS category means m = [1.5, 1.75, 2.0] -> Eq. 6 maps to [0.8, 1.0, 1.2].
    m_cat = np.array([1.5, 1.75, 2.0])
    w_cat = dtca_pos_weights(m_cat)
    assert np.allclose(w_cat, [0.8, 1.0, 1.2])

    # Eq. 7: q_t = sum_{i in A_t} w_{c_i} * (1 - p_i)
    # Suppose 2 tokens commit at step t with w = [1.2, 0.8] and confidences p = [0.5, 0.75]:
    # q_t = 1.2 * (1 - 0.5) + 0.8 * (1 - 0.75) = 0.6 + 0.2 = 0.8
    q_t = dtca_timestep_score(np.array([1.2, 0.8]), np.array([0.5, 0.75]))
    assert q_t == pytest.approx(0.8)

    # Eq. 8: a_{t,i} = w_{c_i} / ((1 / |B_t|) sum_{k in B_t} w_{c_k}), mean(a_{t,i}) == 1.0
    a_ti = dtca_token_credits(w_cat)
    assert np.allclose(a_ti, [0.8, 1.0, 1.2])
    assert float(a_ti.mean()) == pytest.approx(1.0)


def test_pre_convergence_credit_beats_broadcast_under_post_verdict_backfill():
    # On early-stabilized trajectories with 5 post-convergence backfill steps (horizon=6),
    # pre_convergence assigns 0% post-convergence credit and drives target answer probability
    # and accuracy higher than broadcast diffu-GRPO (which assigns 83.33% of credit to post-verdict backfill).
    res_pre = train_trajectory_credit(scheme="pre_convergence", steps=35, horizon=6, seed=0)
    res_broad = train_trajectory_credit(scheme="broadcast", steps=35, horizon=6, seed=0)
    assert res_pre["post_convergence_credit_fraction"] == 0.0
    assert res_broad["post_convergence_credit_fraction"] == pytest.approx(5 / 6, abs=1e-3)
    assert res_pre["greedy_accuracy"] == 1.0
    assert res_pre["target_prob_mean"] > res_broad["target_prob_mean"] + 0.25
