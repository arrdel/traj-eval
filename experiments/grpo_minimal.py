"""Minimal Group Relative Policy Optimization on a verifiable toy task.

This is the estimator from Shao et al., DeepSeekMath (arXiv:2402.03300), as
used by DeepSeek-R1 (arXiv:2501.12948, eq. 1--3). It is not a language-model
trainer. The policy emits one answer per question, so the single-step outcome
advantage has nowhere to be split unless a multi-step trajectory credit scheme
is applied.

Zhao et al., d1 (arXiv:2504.12216), copy one outcome advantage onto every
token and drop the standard-deviation normalization. ``normalize=False``
matches that unnormalized advantage. It still does not assign per-step credit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from traj_eval.metrics import answer_commitment_step, verdict_convergence_step
from traj_eval.types import Trajectory, TrajectoryStep


def group_advantages(rewards: np.ndarray, eps: float = 1e-8, normalize: bool = True) -> np.ndarray:
    """Group-relative advantage over the last axis.

    ``normalize=True`` is DeepSeek-R1: ``(r - mean) / std``.
    ``normalize=False`` is the unnormalized advantage used by diffu-GRPO:
    ``r - mean``. A constant group returns zeros in both cases.
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    centered = rewards - rewards.mean(axis=-1, keepdims=True)
    if not normalize:
        return centered
    std = rewards.std(axis=-1, keepdims=True)
    return centered / np.maximum(std, eps)


def clipped_surrogate(ratio: np.ndarray, advantage: np.ndarray, clip_eps: float) -> np.ndarray:
    """PPO-style pessimistic surrogate, per sample.

    ``min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)``.
    """
    ratio = np.asarray(ratio, dtype=np.float64)
    advantage = np.asarray(advantage, dtype=np.float64)
    unclipped = ratio * advantage
    clipped = np.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantage
    return np.minimum(unclipped, clipped)


def sampled_kl(ref_prob: np.ndarray, new_prob: np.ndarray) -> np.ndarray:
    """Schulman unbiased KL estimator on the sampled action.

    ``k = r - log(r) - 1`` with ``r = pi_ref(a) / pi_theta(a)``.
    This is the term DeepSeek-R1 subtracts inside the GRPO objective.
    """
    r = np.asarray(ref_prob, dtype=np.float64) / np.maximum(np.asarray(new_prob, dtype=np.float64), 1e-12)
    return r - np.log(r) - 1.0


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


@dataclass
class AdditionBatch:
    """Single-digit sums. Features are concatenated one-hots of the addends."""

    features: np.ndarray  # (B, 20)
    answers: np.ndarray  # (B,)


def sample_addition(n: int, rng: np.random.Generator) -> AdditionBatch:
    a = rng.integers(0, 10, size=n)
    b = rng.integers(0, 10, size=n)
    features = np.zeros((n, 20), dtype=np.float64)
    features[np.arange(n), a] = 1.0
    features[np.arange(n), 10 + b] = 1.0
    return AdditionBatch(features=features, answers=(a + b).astype(np.int64))


class AnswerPolicy:
    """Linear softmax policy. One logit vector per action."""

    def __init__(
        self,
        n_features: int,
        n_actions: int,
        rng: np.random.Generator | None = None,
    ) -> None:
        rng = rng or np.random.default_rng(0)
        self.n_actions = n_actions
        self.W = rng.normal(scale=0.01, size=(n_actions, n_features))

    def probs(self, features: np.ndarray) -> np.ndarray:
        return _softmax(features @ self.W.T)


def grpo_step(
    policy: AnswerPolicy,
    features: np.ndarray,
    actions: np.ndarray,
    advantages: np.ndarray,
    old_action_prob: np.ndarray,
    *,
    clip_eps: float = 0.2,
    lr: float = 0.5,
    ref_action_prob: np.ndarray | None = None,
    beta: float = 0.0,
) -> float:
    """One clipped GRPO update. Returns the mean surrogate.

    Gradient of the unclipped branch is ``A * ratio * d log pi``. The clipped
    branch is treated as constant, matching PPO. The KL term, when ``beta > 0``,
    is the sampled-action estimator and is subtracted from the objective.
    """
    probs = policy.probs(features)
    idx = np.arange(len(actions))
    new_p = probs[idx, actions]
    ratio = new_p / np.maximum(old_action_prob, 1e-12)
    surrogate = clipped_surrogate(ratio, advantages, clip_eps)

    unclipped = ratio * advantages
    clipped = np.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    active = unclipped <= clipped + 1e-12
    coef = np.where(active, advantages * ratio, 0.0)

    if beta and ref_action_prob is not None:
        r = ref_action_prob / np.maximum(new_p, 1e-12)
        # J = surrogate - beta * k, dk/d log pi = 1 - r.
        coef = coef - beta * (1.0 - r)

    grad_logits = -probs * coef[:, None]
    grad_logits[idx, actions] += coef
    policy.W += lr * (grad_logits.T @ features) / len(actions)
    return float(surrogate.mean())


def _rollout_update(
    policy: AnswerPolicy,
    features: np.ndarray,
    answers: np.ndarray,
    rng: np.random.Generator,
    group_size: int,
    lr: float,
    normalize: bool,
) -> float:
    """Sample a group per row, assign outcome reward, take one GRPO step."""
    probs = policy.probs(features)
    batch_size = len(answers)
    actions = np.vstack(
        [
            np.array([rng.choice(policy.n_actions, p=probs[i]) for i in range(batch_size)])
            for _ in range(group_size)
        ]
    )
    rewards = (actions == answers[None, :]).astype(np.float64)
    adv = group_advantages(rewards.T, normalize=normalize)
    grpo_step(
        policy,
        np.repeat(features, group_size, axis=0),
        actions.T.reshape(-1),
        adv.reshape(-1),
        np.take_along_axis(probs, actions.T, axis=1).reshape(-1),
        lr=lr,
    )
    return float(rewards.mean())


def train_matching(
    steps: int = 40,
    batch_size: int = 32,
    group_size: int = 8,
    n_ctx: int = 6,
    seed: int = 0,
    lr: float = 1.0,
    normalize: bool = True,
) -> dict[str, float]:
    """Context id is the correct action. Reward is exact match.

    This is the reproduction check: GRPO with a verifiable outcome reward
    should drive greedy accuracy well above ``1 / n_ctx``.
    """
    rng = np.random.default_rng(seed)
    policy = AnswerPolicy(n_features=n_ctx, n_actions=n_ctx, rng=rng)
    for _ in range(steps):
        ctx = rng.integers(0, n_ctx, size=batch_size)
        features = np.zeros((batch_size, n_ctx), dtype=np.float64)
        features[np.arange(batch_size), ctx] = 1.0
        _rollout_update(policy, features, ctx.astype(np.int64), rng, group_size, lr, normalize)
    eval_ctx = np.arange(n_ctx)
    eval_feat = np.eye(n_ctx)
    greedy = policy.probs(eval_feat).argmax(axis=1)
    return {"greedy_accuracy": float((greedy == eval_ctx).mean())}


def train_addition(
    steps: int = 80,
    batch_size: int = 64,
    group_size: int = 8,
    seed: int = 0,
    lr: float = 0.8,
    normalize: bool = True,
) -> dict[str, float]:
    """Single-digit addition. Harder than :func:`train_matching`.

    A linear softmax on concatenated one-hots does not reliably master all
    100 sums under this budget. Use it as a sandbox, not as the passing test.
    """
    rng = np.random.default_rng(seed)
    policy = AnswerPolicy(n_features=20, n_actions=19, rng=rng)
    acc_hist: list[float] = []
    for _ in range(steps):
        batch = sample_addition(batch_size, rng)
        acc_hist.append(
            _rollout_update(
                policy, batch.features, batch.answers, rng, group_size, lr, normalize
            )
        )
    eval_batch = sample_addition(2000, np.random.default_rng(seed + 1))
    greedy = policy.probs(eval_batch.features).argmax(axis=1)
    return {
        "final_train_accuracy": acc_hist[-1],
        "greedy_accuracy": float((greedy == eval_batch.answers).mean()),
        "steps": float(steps),
    }


def step_rs_schedule(
    u: np.ndarray | float,
    *,
    beta_max: float = 1.0,
    gamma: float = 0.5,
) -> np.ndarray:
    """Linear state-dependent risk schedule from StepRS-GRPO (Yu et al. 2026, Eq. 3.6).

    Source definition (arXiv:2610.00661v3, Section 3.2, Eq. 3.2 & 3.6):
    Given local maskedness coordinate ``u in [0, 1]`` (fraction of masked tokens
    in the active block, starting at ``u = 1`` when fully masked and decreasing
    toward ``0`` as tokens are revealed):
        b(u) = beta_min + (beta_max - beta_min) * u,   beta_min = gamma * beta_max
    The paper's default linear half-decay schedule sets ``gamma = 1/2``.
    """
    u_arr = np.asarray(u, dtype=np.float64)
    beta_min = gamma * beta_max
    return beta_min + (beta_max - beta_min) * u_arr


def step_rs_advantage_multiplier(
    beta: np.ndarray | float,
    solve_rate: float,
) -> np.ndarray:
    """Exact finite-group binary-reward multiplier ``c_beta(p)`` (Yu et al. 2026, Lemma 3.1 Eq. 3.7).

    Source definition (arXiv:2610.00661v3, Section 3.3, Lemma 3.1 Eq. 3.7):
    For binary rewards ``r_{qi} in {0, 1}`` with group solve rate
    ``p_hat_q = (1 / G) * sum_{i=1}^G r_{qi}`` and centered advantage
    ``A^0_{qi} = r_{qi} - p_hat_q``, the entropic risk-sensitive advantage
    ``A_{qi}(beta) = (1 / beta) * (exp(beta * r_{qi}) / (G^{-1} sum_h exp(beta * r_{qh})) - 1)``
    factors exactly as:
        A_{qi}(beta) = c_beta(p_hat_q) * A^0_{qi}
        c_beta(p) = (exp(beta) - 1) / (beta * (1 - p + p * exp(beta))),   c_0(p) = 1.
    """
    beta_arr = np.asarray(beta, dtype=np.float64)
    p = float(solve_rate)
    out = np.ones_like(beta_arr, dtype=np.float64)
    nonzero = np.abs(beta_arr) > 1e-12
    b_nz = beta_arr[nonzero]
    eb = np.exp(b_nz)
    out[nonzero] = (eb - 1.0) / (b_nz * (1.0 - p + p * eb))
    return out


def dtca_pos_weights(category_means: np.ndarray) -> np.ndarray:
    """Bounded POS category contribution weights ``w_c in [0.8, 1.2]`` (Chen et al. 2026, Eq. 6).

    Source definition (DTCA, arXiv:2610.35201v1, Section 3.1, Eq. 5--6):
    Given category-level average confidence gains ``m_c = (1 / N_c) sum_{i: c_i = c} Delta P_i``:
        w_c = 0.8 + 0.4 * (m_c - min_j m_j) / (max_j m_j - min_j m_j)
    If all categories have identical mean contribution, returns ``1.0`` for each category.
    """
    m = np.asarray(category_means, dtype=np.float64)
    if m.size == 0:
        return np.zeros(0, dtype=np.float64)
    m_min = float(m.min())
    m_max = float(m.max())
    if m_max - m_min <= 1e-12:
        return np.ones_like(m, dtype=np.float64)
    return 0.8 + 0.4 * (m - m_min) / (m_max - m_min)


def dtca_timestep_score(
    pos_weights: np.ndarray,
    commit_confidences: np.ndarray,
) -> float:
    """Timestep importance score ``q_t = sum_{i in A_t} w_{c_i} * (1 - p_i)`` (Chen et al. 2026, Eq. 7)."""
    w = np.asarray(pos_weights, dtype=np.float64)
    p = np.asarray(commit_confidences, dtype=np.float64)
    if w.size == 0:
        return 0.0
    return float(np.sum(w * (1.0 - p)))


def dtca_token_credits(pos_weights: np.ndarray) -> np.ndarray:
    """Normalized token credit weights ``a_{t,i}`` within a selected step (Chen et al. 2026, Eq. 8).

    Source definition (DTCA, arXiv:2610.35201v1, Section 3.3, Eq. 8--9):
        a_{t,i} = w_{c_i} / ((1 / |B_t|) * sum_{k in B_t} w_{c_k})
    keeping the mean credit across ``B_t`` equal to ``1.0``.
    """
    w = np.asarray(pos_weights, dtype=np.float64)
    if w.size == 0:
        return np.zeros(0, dtype=np.float64)
    mean_w = float(w.mean())
    if mean_w <= 1e-12:
        return np.ones_like(w, dtype=np.float64)
    return w / mean_w


def trajectory_step_weights(
    traj: Trajectory,
    scheme: str = "broadcast",
    *,
    fork_threshold: float = 0.6,
) -> np.ndarray:
    """Compute per-step RL credit weights ``w_t`` along a denoising trajectory.

    Source definitions from the 2025--2026 dLLM RL literature:

    * ``"broadcast"``: ``w_t = 1.0`` for every step ``t`` (Zhao et al., d1 /
      diffu-GRPO, arXiv:2504.12216). Broadcasts outcome advantage uniformly to
      all denoising steps, including post-convergence backfill and EOS padding.
    * ``"pre_convergence"``: ``w_t = 1.0`` for steps up to and including
      ``verdict_convergence_step(traj)`` and ``0.0`` afterwards.
    * ``"pre_commitment"``: ``w_t = 1.0`` for steps up to and including
      ``answer_commitment_step(traj)`` (or ``verdict_convergence_step`` when
      commitment is not logged) and ``0.0`` afterwards.
    * ``"step_rs"``: linear half-decay schedule ``b(u) = beta_min + (beta_max - beta_min) * u``
      with ``beta_min = 0.5 * beta_max`` (``gamma = 0.5``, normalized to ``beta_max = 1.0``)
      from ``1.0`` at the first denoising step (``u = 1.0``) to ``0.5`` at the
      final step (``u = 0.0``), matching StepRS-GRPO (Yu et al. 2026,
      arXiv:2610.00661v3, Section 3.2, Eq. 3.6).
    * ``"fork_only"``: ``w_t = 1.0`` only at fallback steps
      ``F_i = {t : max_{k in M_{i,t}} c^k_{i,t} <= lambda}`` where step confidence
      is at or below ``fork_threshold`` (``step.confidence <= fork_threshold``, or
      before verdict convergence when confidence is not logged), matching
      ForkGRPO (Frkovic, Jazbec, & Naesseth 2026, arXiv:2609.39859v1,
      Algorithm 2 & Section 3.2 Eq. 2).
    * ``"dtca"``: trajectory-level reduction of Denoising Token Credit Assignment
      (Chen et al. 2026, arXiv:2610.35201v1, Section 3.1 Eq. 3--4), weighting
      steps by ReLU-clipped confidence gain ``Delta P_t = max(0, c_{t+1} - c_t)``
      plus an anchor at verdict convergence. Token-level POS weight scaling
      (Eq. 6), timestep selection score (Eq. 7), and normalized token credit
      (Eq. 8) are provided by :func:`dtca_pos_weights`, :func:`dtca_timestep_score`,
      and :func:`dtca_token_credits`.
    """
    steps = traj.steps
    n = len(steps)
    if n == 0:
        return np.zeros(0, dtype=np.float64)

    conv_step = verdict_convergence_step(traj)
    comm_step = answer_commitment_step(traj)

    if scheme == "broadcast":
        return np.ones(n, dtype=np.float64)

    if scheme == "pre_convergence":
        if conv_step is None:
            return np.ones(n, dtype=np.float64)
        return np.array(
            [1.0 if s.step <= conv_step else 0.0 for s in steps], dtype=np.float64
        )

    if scheme == "pre_commitment":
        cutoff = comm_step if comm_step is not None else conv_step
        if cutoff is None:
            return np.ones(n, dtype=np.float64)
        return np.array(
            [1.0 if s.step <= cutoff else 0.0 for s in steps], dtype=np.float64
        )

    if scheme == "step_rs":
        if n == 1:
            return np.ones(1, dtype=np.float64)
        u = np.linspace(1.0, 0.0, n, dtype=np.float64)
        return step_rs_schedule(u, beta_max=1.0, gamma=0.5)

    if scheme == "fork_only":
        weights = np.zeros(n, dtype=np.float64)
        for i, s in enumerate(steps):
            if s.confidence is not None:
                if s.confidence <= fork_threshold:
                    weights[i] = 1.0
            elif conv_step is None or s.step <= conv_step:
                weights[i] = 1.0
        return weights

    if scheme == "dtca":
        weights = np.zeros(n, dtype=np.float64)
        for i in range(n - 1):
            c_now = steps[i].confidence
            c_next = steps[i + 1].confidence
            if c_now is not None and c_next is not None and c_next > c_now:
                weights[i] = float(c_next - c_now)
        if conv_step is not None:
            for i, s in enumerate(steps):
                if s.step == conv_step:
                    weights[i] = max(weights[i], 1.0)
        if weights.sum() == 0.0:
            return trajectory_step_weights(traj, scheme="pre_convergence")
        return weights / weights.max()

    raise ValueError(f"Unsupported trajectory credit scheme: {scheme!r}")


def post_convergence_credit_fraction(
    traj: Trajectory,
    scheme: str = "broadcast",
    *,
    fork_threshold: float = 0.6,
) -> float:
    """Fraction of total RL credit weight assigned strictly after verdict convergence.

    On trajectories that stabilize early, an outcome-broadcast RL method
    (``scheme="broadcast"``) assigns most of its update mass to steps after
    the verdict is already fixed, whereas ``scheme="pre_convergence"`` assigns
    ``0.0``.
    """
    steps = traj.steps
    if not steps:
        return 0.0
    conv_step = verdict_convergence_step(traj)
    if conv_step is None:
        return 0.0
    w = trajectory_step_weights(traj, scheme=scheme, fork_threshold=fork_threshold)
    total = float(w.sum())
    if total <= 0.0:
        return 0.0
    post_mass = float(
        sum(w[i] for i, s in enumerate(steps) if s.step > conv_step)
    )
    return round(post_mass / total, 4)


def train_trajectory_credit(
    *,
    scheme: str = "pre_convergence",
    steps: int = 35,
    horizon: int = 6,
    batch_size: int = 24,
    group_size: int = 6,
    n_ctx: int = 6,
    seed: int = 0,
    lr: float = 0.8,
    normalize: bool = True,
) -> dict[str, float]:
    """Multi-step denoising trajectory GRPO sandbox comparing credit-assignment rules.

    Each rollout has ``horizon`` denoising steps (``t = 0 .. horizon - 1``):
    * At ``t = 0``, the policy observes the true prompt context feature (one-hot
      of ``ctx``) and samples a verdict ``a_0`` (which stabilizes as the
      trajectory's verdict at ``t = 0``).
    * At post-convergence steps ``t = 1 .. horizon - 1``, the verdict stays fixed
      at ``a_0`` while the model elaborates post-verdict backfill tokens under
      uninformative distractor context features and actions.

    Outcome reward is ``1.0`` iff the final verdict equals ``ctx``.
    * ``scheme="broadcast"`` (diffu-GRPO / d1) copies the outcome advantage
      uniformly onto all ``horizon`` steps, injecting ``horizon - 1`` spurious
      post-convergence updates per rollout.
    * ``scheme="pre_convergence"`` uses :func:`trajectory_step_weights`
      to zero out credit after ``verdict_convergence_step``, updating only on the
      step(s) that determined the verdict.
    """
    rng = np.random.default_rng(seed)
    policy = AnswerPolicy(n_features=n_ctx, n_actions=n_ctx, rng=rng)

    # Canonical early-stabilized trajectory template (verdict converges at step 0,
    # followed by horizon - 1 post-convergence backfill steps).
    template_traj = Trajectory(
        id="template",
        prompt="q",
        gold_answer="0",
        steps=[
            TrajectoryStep(
                step=t,
                prediction="0",
                confidence=0.3 if t == 0 else 0.95,
                answer_committed=(t >= 0),
            )
            for t in range(horizon)
        ],
    )
    step_w = trajectory_step_weights(template_traj, scheme=scheme)

    for _ in range(steps):
        ctx = rng.integers(0, n_ctx, size=batch_size)
        true_feat = np.zeros((batch_size, n_ctx), dtype=np.float64)
        true_feat[np.arange(batch_size), ctx] = 1.0

        probs_0 = policy.probs(true_feat)
        verdict_actions = np.vstack(
            [
                np.array([rng.choice(n_ctx, p=probs_0[i]) for i in range(batch_size)])
                for _ in range(group_size)
            ]
        )  # (G, B)
        rewards = (verdict_actions == ctx[None, :]).astype(np.float64)
        adv_group = group_advantages(rewards.T, normalize=normalize).reshape(-1)  # (B * G,)

        all_feats: list[np.ndarray] = []
        all_acts: list[np.ndarray] = []
        all_advs: list[np.ndarray] = []
        all_old_p: list[np.ndarray] = []

        for t in range(horizon):
            w_t = float(step_w[t])
            if w_t <= 0.0:
                continue
            if t == 0:
                f_t = np.repeat(true_feat, group_size, axis=0)
                a_t = verdict_actions.T.reshape(-1)
                p_t = np.take_along_axis(probs_0, verdict_actions.T, axis=1).reshape(-1)
            else:
                # Post-convergence steps condition on the same prompt feature true_feat
                # while emitting post-verdict rationale/EOS backfill tokens (shifted away
                # from the answer token). Broadcasting outcome advantage onto these steps
                # reinforces non-answer tokens on the prompt context.
                f_t = np.repeat(true_feat, group_size, axis=0)
                a_t = (verdict_actions.T.reshape(-1) + (t % (n_ctx - 1)) + 1) % n_ctx
                probs_t = policy.probs(f_t)
                p_t = probs_t[np.arange(batch_size * group_size), a_t]

            all_feats.append(f_t)
            all_acts.append(a_t)
            all_advs.append(adv_group * w_t)
            all_old_p.append(p_t)

        if all_feats:
            grpo_step(
                policy,
                np.vstack(all_feats),
                np.concatenate(all_acts),
                np.concatenate(all_advs),
                np.concatenate(all_old_p),
                lr=lr,
            )

    eval_ctx = np.arange(n_ctx)
    eval_feat = np.eye(n_ctx)
    eval_probs = policy.probs(eval_feat)
    greedy = eval_probs.argmax(axis=1)
    return {
        "greedy_accuracy": float((greedy == eval_ctx).mean()),
        "target_prob_mean": float(eval_probs[eval_ctx, eval_ctx].mean()),
        "post_convergence_credit_fraction": post_convergence_credit_fraction(
            template_traj, scheme=scheme
        ),
    }
