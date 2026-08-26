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
    # Against the environment's own list, not the module constant: the list grows with
    # what the environment offers, and comparing to the constant is how a whole action
    # type stayed unreachable for a day.
    assert env.action_space.nvec[0] == len(env.action_types)
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
    # Against `env.action_types` and not the module constant, which is what let a whole
    # action type go missing. The list grows a `stamp` entry as soon as a design is
    # loaded; the mask did not, and the network takes its type count from the mask.
    assert mask["type"].shape == (len(env.action_types),)
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
    # Both policies play the same world. Without the pin they do not: Mindustry paints
    # ore on at load time and re-randomises it every load, so this compared two policies
    # on two different maps and duly inverted, with random scoring 39 against Alpha's 16.
    from dataclasses import replace

    env.task = replace(tasks.T1_COPPER, world_seed=20260824)
    try:
        alpha = AlphaPolicy(env)
        alpha.reset()
        scripted = run_episode(env, alpha)
        chaotic = run_episode(env, MaskedRandomPolicy(env.action_space, seed=0, env=env))
    finally:
        env.task = tasks.T1_COPPER

    assert scripted["reward"] > chaotic["reward"]
    assert scripted["applied"] > chaotic["applied"]
    assert scripted["items"].get("copper", 0) > chaotic["items"].get("copper", 0)


def test_a_loaded_design_is_reachable(env: MindustryEnv) -> None:
    """A design the policy cannot choose is a design that does not exist.

    `action_types` grows a `stamp` entry as soon as a design is loaded and the mask did
    not, so the head had six outputs while the environment named seven and the seventh was
    unreachable. Measured over 54 archived episodes: zero stamps in 39,456 actions. Not a
    choice the policy made, an option it never had, on every run that passed --designs.

    It is the mechanism built for the problem the project keeps hitting, which is that a
    conveyor line placed one tile at a time never gets finished: 5,719 conveyors across
    177 episodes and one line that ever met end to end.
    """
    from gamma.env import STAMP
    from gamma.library import load as load_designs

    designs = tuple(load_designs(Path("docs/designs.json")))
    assert designs, "no design to test with"

    was = env.designs
    try:
        env.designs = designs
        _, info = env.reset()
        types = env.action_types
        assert STAMP in types
        assert info["action_mask"]["type"].shape == (len(types),)
    finally:
        env.designs = was


def test_breaking_can_aim_at_something_breakable(env: MindustryEnv) -> None:
    """One position head serves four action types, so its mask must cover all of them.

    It was `free` alone: buildable, empty, not solid. Right for building and exactly
    inverted for breaking, because `free` and `owned` are disjoint by construction, so
    every tile the agent could aim at while choosing `break` was guaranteed to hold
    nothing. Measured over 30 archived episodes: 6,660 demolitions, 23 of which hit a
    building the agent had placed.
    """
    import numpy as np

    _, info = env.reset()
    mask = info["action_mask"]

    spatial = info["raw"]["spatial"]
    channels = env._bridge.channels
    owned = spatial[channels.index("block_ally")] > 0

    assert owned.any(), "the core is an ally building, so something must be breakable"
    assert (mask["position"] & owned).any(), (
        "no tile holding an ally building is aimable, so breaking cannot ever work"
    )


def test_every_way_of_building_is_remembered(env: MindustryEnv) -> None:
    """The churn penalty watches tiles the agent built, so it must know all of them.

    Only manual placements were noted at first and the penalty never fired once: measured
    over 30 episodes, not one demolition landed on a tile the agent had `place`d, because
    it had stopped placing by hand. Nearly everything it built came from `connect`, forty
    blocks at a time, and those were the buildings it was tearing down. A rule watching the
    wrong door is indistinguishable from no rule.
    """
    env._built_at.clear()
    env._steps = 10

    env._remember([(4, 5, "conveyor", 0), (6, 7, "mechanical-drill", 0)])
    assert env._built_at[(4, 5)] == 10 and env._built_at[(6, 7)] == 10

    # Breaking one of them soon after is churn; breaking it much later is revision.
    assert env._note_churn("break", 4, 5) is True

    env._remember([(8, 9, "conveyor", 0)])
    env._steps = 10 + env.CHURN_WINDOW + 1
    assert env._note_churn("break", 8, 9) is False


def _settle(env, applied: bool) -> None:
    env._settle_churn({"action": {"applied": applied}})


def test_a_refused_demolition_is_not_charged_and_the_tile_is_not_forgotten(
        env: MindustryEnv) -> None:
    """Charging a refused break prices an attempt that changed nothing.

    Worse, popping the tile on the way would let the demolition that does land a moment
    later go free, so a single refusal would launder the very behaviour being priced.
    """
    env._built_at.clear()
    env._steps = 10
    env._remember([(3, 3, "conveyor", 0)])

    env._churned = env._note_churn("break", 3, 3)
    env._breaking = (3, 3)
    assert env._churned is True
    _settle(env, applied=False)
    assert env._churned is False, "nothing was torn down, so nothing is owed"
    assert (3, 3) in env._built_at, "the tile still stands and is still the agent's"

    env._churned = env._note_churn("break", 3, 3)
    env._breaking = (3, 3)
    _settle(env, applied=True)
    assert env._churned is True
    assert (3, 3) not in env._built_at, "it is gone now, so the ledger lets it go"


def test_a_refused_placement_does_not_go_on_the_ledger(env: MindustryEnv) -> None:
    """Otherwise the agent is later charged for clearing a building it never owned."""
    env._built_at.clear()
    env._steps = 10

    env._note_churn("place", 7, 7)
    _settle(env, applied=False)
    assert (7, 7) not in env._built_at

    env._note_churn("place", 7, 7)
    _settle(env, applied=True)
    assert (7, 7) in env._built_at
