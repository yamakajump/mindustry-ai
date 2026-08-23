"""The Gymnasium environment behaves like one, and the baselines rank as expected."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from gymnasium import spaces

from gamma import tasks
from gamma.alpha import AlphaPolicy
from gamma.env import ACTION_TYPES, GLOBAL_FIELDS, MindustryEnv
from gamma.policies import MaskedRandomPolicy, RandomPolicy, run_episode

def test_spaces_are_available_before_the_first_reset(env: MindustryEnv) -> None:
    """Gymnasium callers build policies from the spaces, so asking early must work."""
    assert isinstance(env.action_space, spaces.MultiDiscrete)
    assert isinstance(env.observation_space, spaces.Dict)
    assert env.action_space.nvec[0] == len(ACTION_TYPES)
    assert env.action_space.nvec[1] == len(env.blocks)


def test_reset_returns_an_observation_in_the_space(env: MindustryEnv) -> None:
    obs, info = env.reset()
    assert env.observation_space.contains(obs)
    assert obs["global"].shape == (len(GLOBAL_FIELDS),)
    assert "action_mask" in info


def test_step_returns_the_gym_five_tuple(env: MindustryEnv) -> None:
    env.reset()
    obs, reward, terminated, truncated, info = env.step(np.zeros(5, dtype=np.int64))
    assert env.observation_space.contains(obs)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_masks_cover_every_head(env: MindustryEnv) -> None:
    _, info = env.reset()
    mask = info["action_mask"]
    assert mask["type"].shape == (len(ACTION_TYPES),)
    assert mask["block"].shape == (len(env.blocks),)
    assert mask["position"].ndim == 2
    assert mask["type"][ACTION_TYPES.index("noop")], "doing nothing is always legal"


def test_observation_keys_survive_losing_the_core(env: MindustryEnv) -> None:
    """Fields must not appear and disappear.

    `items` used to be omitted when there was no core, which crashed consumers at the
    exact moment an episode was ending. A policy fed a vector that changes shape learns
    nothing from it either.
    """
    _, info = env.reset()
    raw = info["raw"]
    for field in ("items", "has_core", "core_x", "core_y", "map_width", "wave"):
        assert field in raw, f"{field} missing from the observation"


def test_illegal_action_is_reported_not_raised(env: MindustryEnv) -> None:
    env.reset()
    far = np.array([ACTION_TYPES.index("place"), 0, 0, 0, 0], dtype=np.int64)
    _, _, _, _, info = env.step(far)
    assert info["action"] is not None
    assert info["action"]["applied"] is False


def test_truncation_happens_at_the_step_limit(env: MindustryEnv) -> None:
    """Uses a shortened copy of the task so the check does not cost 450 steps."""
    from dataclasses import replace

    env.task = replace(tasks.T1_COPPER, max_steps=3)
    try:
        env.reset()
        flags = [env.step(np.zeros(5, dtype=np.int64))[3] for _ in range(3)]
        assert flags[-1] is True, "the episode never truncated"
        assert not any(flags[:-1]), "truncated too early"
    finally:
        env.task = tasks.T1_COPPER


@pytest.mark.perf
def test_alpha_beats_random_on_t1(env: MindustryEnv) -> None:
    """The baseline ordering the whole curriculum rests on.

    Marked perf because it plays several full episodes. If this inverts, either Alpha
    broke or the task stopped measuring anything.
    """
    alpha = AlphaPolicy(env)
    alpha.reset()
    scripted = run_episode(env, alpha)
    chaotic = run_episode(env, MaskedRandomPolicy(env.action_space, seed=0))

    assert scripted["reward"] > chaotic["reward"]
    assert scripted["applied"] > chaotic["applied"]
    assert scripted["items"].get("copper", 0) > chaotic["items"].get("copper", 0)
