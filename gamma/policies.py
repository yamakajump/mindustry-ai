"""Reference policies to measure learned agents against.

`RandomPolicy` is the floor. `MaskedRandomPolicy` is the honest floor: it only picks
actions the environment says are legal, so beating it means the agent learned something
beyond the rules it was handed.
"""

from __future__ import annotations

from typing import Any

import numpy as np



class RandomPolicy:
    """Uniform over the whole action space, masks ignored."""

    def __init__(self, action_space, seed: int | None = None) -> None:
        self.action_space = action_space
        if seed is not None:
            self.action_space.seed(seed)

    def act(self, observation: dict[str, np.ndarray], info: dict[str, Any]) -> np.ndarray:
        return self.action_space.sample()


class MaskedRandomPolicy:
    """Uniform over legal actions only.

    This is the baseline that matters. A learned agent that cannot beat it has learned
    nothing except what the masks already told it.
    """

    def __init__(self, action_space, seed: int | None = None, env=None) -> None:
        self.action_space = action_space
        self.rng = np.random.default_rng(seed)
        # The set of action types depends on whether the agent has a body, so it is read
        # from the environment rather than from a module constant that only knew about one.
        self.types = tuple(env.action_types) if env is not None else ("noop", "place", "break")

    def act(self, observation: dict[str, np.ndarray], info: dict[str, Any]) -> np.ndarray:
        mask = info.get("action_mask", {})
        action = np.zeros(5, dtype=np.int64)

        legal_types = np.flatnonzero(mask.get("type", np.ones(len(self.types), bool)))
        kind = int(self.rng.choice(legal_types)) if legal_types.size else 0
        action[0] = kind

        if self.types[kind] in ("noop", "unload"):
            return action

        legal_blocks = np.flatnonzero(mask.get("block", np.ones(1, bool)))
        if legal_blocks.size:
            action[1] = int(self.rng.choice(legal_blocks))

        # Mining needs an ore tile, not merely a free one.
        key = "mineable" if self.types[kind] == "mine" and "mineable" in mask else "position"
        positions = mask.get(key)
        if positions is not None and positions.any():
            ys, xs = np.nonzero(positions)
            pick = int(self.rng.integers(len(xs)))
            action[2], action[3] = int(xs[pick]), int(ys[pick])

        action[4] = int(self.rng.integers(4))
        return action


def run_episode(env, policy, max_steps: int | None = None) -> dict[str, Any]:
    """Play one episode and return what happened.

    Reports refusals as well as reward: an agent that scores zero because every action
    was rejected is a different problem from one that acted and failed.
    """
    observation, info = env.reset()
    total = 0.0
    applied = refused = 0
    limit = max_steps or env.task.max_steps

    for step in range(limit):
        action = policy.act(observation, info)
        observation, reward, terminated, truncated, info = env.step(action)
        total += reward

        outcome = info.get("action")
        if outcome is not None:
            applied += bool(outcome.get("applied"))
            refused += not outcome.get("applied")

        if terminated or truncated:
            break

    raw = info.get("raw", {})
    return {
        "reward": total,
        "steps": step + 1,
        "applied": applied,
        "refused": refused,
        "solved": bool(env.task.succeeded(raw)),
        "items": raw.get("items", {}),
    }
