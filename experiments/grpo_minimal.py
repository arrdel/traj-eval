"""Minimal Group Relative Policy Optimization on a verifiable toy task.

This is the estimator from Shao et al., DeepSeekMath (arXiv:2402.03300), as
used by DeepSeek-R1 (arXiv:2501.12948, eq. 1--3). It is not a language-model
trainer. The policy emits one answer per question, so the advantage has
nowhere to be split along a trajectory. That split is Paper 2.

Zhao et al., d1 (arXiv:2504.12216), copy one outcome advantage onto every
token and drop the standard-deviation normalization. ``normalize=False``
matches that unnormalized advantage. It still does not assign per-step credit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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
