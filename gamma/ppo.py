"""PPO for a factored, masked action space.

Written here rather than pulled from a library, as decided in
`docs/decisions/0005-in-repo-ppo.md`: Stable-Baselines3 has no clean support for
autoregressive multi-head actions with per-head masks, which is the one thing this project
cannot compromise on.

Kept to a single readable file on purpose, in the spirit of CleanRL. An algorithm you have
implemented is one you can debug.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import torch
from torch import nn

from gamma.net import PolicyNet


#: The four choices one action is made of, in the order the network returns them.
HEADS = ("type", "block", "position", "rotation")


def _grad_norm(net: nn.Module, term: torch.Tensor, optimiser) -> float:
    """The norm of the gradient one loss term puts on the policy, on its own."""
    optimiser.zero_grad()
    term.backward(retain_graph=True)
    total = 0.0
    for parameter in net.parameters():
        if parameter.grad is not None:
            total += float(parameter.grad.detach().pow(2).sum())
    optimiser.zero_grad()
    return total ** 0.5


@dataclass
class PPOConfig:
    """Defaults are CleanRL's, which are sane for a first run on a new environment."""

    steps_per_env: int = 128
    epochs: int = 4
    minibatches: int = 4

    learning_rate: float = 2.5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    entropy_coef: float = 0.03
    value_coef: float = 0.5
    max_grad_norm: float = 0.5

    #: Annealed to zero over training. Left constant, the policy keeps taking large steps
    #: long after it should be refining.
    anneal_lr: bool = True

    #: Scales rewards by a running estimate of return standard deviation.
    #:
    #: Without it the critic dominates the update and the policy stops exploring. Measured
    #: on the endless task: value loss around 800 against a policy loss of 0.26, and
    #: entropy falling from 2.85 to zero within five updates. The agent had committed to a
    #: single action before it had seen anything.
    normalise_rewards: bool = True

    #: Higher than the usual 0.01, because this action space is wide and collapsing early
    #: is the failure mode actually observed here.
    entropy_coef_start: float = 0.03

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class RolloutBuffer:
    """One batch of experience, shaped (steps, envs, ...)."""

    spatial: torch.Tensor
    globals_: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    values: torch.Tensor
    masks: dict[str, torch.Tensor] = field(default_factory=dict)


class RunningScale:
    """Running estimate of the standard deviation of discounted returns.

    Rewards are divided by it, which keeps the critic's targets on a stable scale no
    matter what the task hands out. A reward function that pays 5 per wave and 0.01 per
    ore otherwise forces the value head to span three orders of magnitude, and its
    gradient then drowns out everything else in the update.
    """

    def __init__(self, gamma: float) -> None:
        self.gamma = gamma
        self.mean = 0.0
        self.var = 1.0
        self.count = 1e-4
        self._returns: np.ndarray | None = None

    def update(self, rewards: np.ndarray, dones: np.ndarray) -> float:
        if self._returns is None:
            self._returns = np.zeros(rewards.shape[-1], dtype=np.float64)

        for step in range(rewards.shape[0]):
            self._returns = self._returns * self.gamma * (1.0 - dones[step]) + rewards[step]
            batch_mean = float(self._returns.mean())
            batch_var = float(self._returns.var())
            batch_count = self._returns.size

            delta = batch_mean - self.mean
            total = self.count + batch_count
            self.mean += delta * batch_count / total
            m_a = self.var * self.count
            m_b = batch_var * batch_count
            self.var = (m_a + m_b + delta ** 2 * self.count * batch_count / total) / total
            self.count = total

        return float(np.sqrt(self.var) + 1e-8)


class PPO:
    """Trains a PolicyNet against a set of environments driven by the caller."""

    def __init__(self, net: PolicyNet, config: PPOConfig | None = None) -> None:
        self.config = config or PPOConfig()
        self.net = net.to(self.config.device)
        self.optimiser = torch.optim.Adam(
            self.net.parameters(), lr=self.config.learning_rate, eps=1e-5
        )
        self.updates = 0
        self.scale = RunningScale(self.config.gamma)

    # Acting ----------------------------------------------------------------------

    @torch.no_grad()
    def act(self, observation: dict[str, np.ndarray], masks: dict[str, np.ndarray]):
        """Choose actions for a batch of environments."""
        spatial = torch.as_tensor(observation["spatial"], device=self.config.device)
        globals_ = torch.as_tensor(observation["global"], device=self.config.device)
        torch_masks = {k: torch.as_tensor(v, device=self.config.device) for k, v in masks.items()}
        return self.net.act(spatial, globals_, torch_masks)

    # Learning --------------------------------------------------------------------

    def advantages(self, buffer: RolloutBuffer, last_value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Generalised advantage estimation, walked backwards through the rollout."""
        config = self.config
        steps = buffer.rewards.shape[0]
        advantages = torch.zeros_like(buffer.rewards)
        running = torch.zeros_like(last_value)

        for step in reversed(range(steps)):
            if step == steps - 1:
                next_value, next_alive = last_value, 1.0 - buffer.dones[step]
            else:
                next_value, next_alive = buffer.values[step + 1], 1.0 - buffer.dones[step]

            delta = buffer.rewards[step] + config.gamma * next_value * next_alive - buffer.values[step]
            running = delta + config.gamma * config.gae_lambda * next_alive * running
            advantages[step] = running

        return advantages, advantages + buffer.values

    def update(self, buffer: RolloutBuffer, last_value: torch.Tensor) -> dict[str, float]:
        config = self.config

        if config.normalise_rewards:
            std = self.scale.update(
                buffer.rewards.detach().cpu().numpy().astype(np.float64),
                buffer.dones.detach().cpu().numpy().astype(np.float64),
            )
            buffer.rewards = buffer.rewards / std

        advantages, returns = self.advantages(buffer, last_value)

        flat = lambda tensor: tensor.reshape(-1, *tensor.shape[2:])  # noqa: E731
        b_spatial = flat(buffer.spatial)
        b_globals = flat(buffer.globals_)
        b_actions = flat(buffer.actions)
        b_log_probs = flat(buffer.log_probs)
        b_advantages = flat(advantages)
        b_returns = flat(returns)
        b_values = flat(buffer.values)
        b_masks = {k: flat(v) for k, v in buffer.masks.items()}

        batch = b_spatial.shape[0]
        size = batch // config.minibatches
        indices = np.arange(batch)

        stats = {"grad_policy": 0.0, "grad_entropy": 0.0, "grad_total": 0.0,
                 **{f"entropy_{name}": 0.0 for name in HEADS},
                 "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "clip_fraction": 0.0,
                 "reward_scale": float(np.sqrt(self.scale.var)) if config.normalise_rewards else 1.0}
        passes = 0

        for _ in range(config.epochs):
            np.random.shuffle(indices)
            for start in range(0, batch, size):
                chunk = indices[start:start + size]

                _, log_prob, entropy, value, spread = self.net.act(
                    b_spatial[chunk], b_globals[chunk],
                    {k: v[chunk] for k, v in b_masks.items()},
                    action=b_actions[chunk], per_head=True,
                )

                ratio = (log_prob - b_log_probs[chunk]).exp()

                # Normalised per minibatch, which keeps the update scale stable when a
                # few episodes produce far larger rewards than the rest.
                chunk_advantages = b_advantages[chunk]
                chunk_advantages = (chunk_advantages - chunk_advantages.mean()) / (
                    chunk_advantages.std() + 1e-8
                )

                unclipped = -chunk_advantages * ratio
                clipped = -chunk_advantages * torch.clamp(ratio, 1 - config.clip, 1 + config.clip)
                policy_loss = torch.max(unclipped, clipped).mean()

                # Clipped value loss, the same trust region idea applied to the critic.
                value_clipped = b_values[chunk] + torch.clamp(
                    value - b_values[chunk], -config.clip, config.clip
                )
                value_loss = 0.5 * torch.max(
                    (value - b_returns[chunk]) ** 2,
                    (value_clipped - b_returns[chunk]) ** 2,
                ).mean()

                entropy_loss = entropy.mean()
                loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy_loss

                if passes == 0:
                    # Which term is actually steering, measured once per update.
                    #
                    # Comparing the two losses is worthless and was tried: the policy loss
                    # sits around 0.006 while the entropy bonus is 0.03 x 11.6 = 0.35, and
                    # concluding anything from that ratio compares values where only
                    # gradients matter. A large flat term contributes nothing; a small
                    # sharp one can dominate. So both are backed through the network and
                    # their gradient norms compared.
                    stats["grad_policy"] = _grad_norm(self.net, policy_loss, self.optimiser)
                    stats["grad_entropy"] = _grad_norm(
                        self.net, -config.entropy_coef * entropy_loss, self.optimiser)

                self.optimiser.zero_grad()
                loss.backward()
                # The return value is the norm BEFORE clipping, which is free to keep and
                # says how much of each update is being thrown away. A run whose gradient
                # is routinely several times the ceiling is taking the same size step in a
                # direction it barely controls, every time.
                before_clip = nn.utils.clip_grad_norm_(
                    self.net.parameters(), config.max_grad_norm)
                stats["grad_total"] += float(before_clip)
                self.optimiser.step()

                with torch.no_grad():
                    stats["clip_fraction"] += ((ratio - 1).abs() > config.clip).float().mean().item()
                stats["policy_loss"] += policy_loss.item()
                stats["value_loss"] += value_loss.item()
                stats["entropy"] += entropy_loss.item()
                for name, value_ in zip(HEADS, spread.tolist()):
                    stats[f"entropy_{name}"] += value_
                passes += 1

        self.updates += 1
        # Measured once per update rather than accumulated, so they must not be averaged
        # over the minibatches like the rest.
        once = {key: stats.pop(key) for key in ("reward_scale", "grad_policy", "grad_entropy")}
        out = {k: v / max(1, passes) for k, v in stats.items()}
        out.update(once)
        return out

    def set_learning_rate(self, fraction_done: float) -> float:
        """Linear decay to zero across the run."""
        if not self.config.anneal_lr:
            return self.config.learning_rate
        lr = self.config.learning_rate * max(0.0, 1.0 - fraction_done)
        for group in self.optimiser.param_groups:
            group["lr"] = lr
        return lr

    # Persistence ------------------------------------------------------------------

    def save(self, path) -> None:
        torch.save({"net": self.net.state_dict(), "updates": self.updates}, path)

    def load(self, path) -> None:
        state = torch.load(path, map_location=self.config.device)
        self.net.load_state_dict(state["net"])
        self.updates = state.get("updates", 0)
