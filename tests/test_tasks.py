"""What each curriculum task pays for.

A reward is the whole of what an agent is being asked to do, and it is the one part of a
run that no amount of watching will tell you is wrong: a policy learning nothing looks
exactly like a policy learning slowly. These run without a server, because the question is
arithmetic and not simulation.
"""

from __future__ import annotations

import pytest

from gamma import tasks

Observation = dict


def state(**fields) -> Observation:
    """A Ground Zero observation with everything at its starting value."""
    base = {"wave": 1, "core_health": 1100.0, "items": {"copper": 300, "lead": 300}}
    items = {**base["items"], **fields.pop("items", {})}
    return {**base, **fields, "items": items}


@pytest.fixture
def reward():
    return tasks.get("GZ_capture").reward


def test_standing_still_earns_nothing(reward) -> None:
    """The failure this task had for twenty-six generations: every episode scored the
    same, because the only term was a wave counter running on a timer."""
    assert reward(state(), state()) == 0.0


def test_a_wave_survived_is_worth_one(reward) -> None:
    assert reward(state(), state(wave=2)) == pytest.approx(1.0)


def test_damage_to_the_core_costs(reward) -> None:
    """The term that answers whether the defence held."""
    assert reward(state(), state(core_health=800.0)) < 0


def test_losing_the_core_costs_more_than_the_waves_it_bought(reward) -> None:
    """Otherwise letting the enemy through and collecting waves is a winning strategy."""
    lost = reward(state(), state(core_health=0.0, wave=2))
    assert lost < -2, lost


def test_a_repaired_core_pays_nothing_back(reward) -> None:
    """A core does not heal, but nothing should reward the appearance of it either: a
    term that pays both ways is a term an agent can pump."""
    assert reward(state(core_health=800.0), state()) == pytest.approx(0.0)


def test_banking_ore_is_worth_a_fraction_of_a_wave(reward) -> None:
    """Enough to give the first thousand steps a gradient, not enough to beat surviving."""
    banked = reward(state(), state(items={"copper": 400}))
    assert 0 < banked < reward(state(), state(wave=2))


def test_spending_ore_is_not_punished_into_paralysis(reward) -> None:
    """Building costs resources. A penalty as large as the reward would teach the agent to
    hoard, which is the opposite of the point."""
    spent = reward(state(), state(items={"copper": 200}))
    assert spent > -1


def test_capturing_the_sector_is_surviving_past_wave_ten(reward) -> None:
    task = tasks.get("GZ_capture")
    assert not task.succeeded({"wave": 10})
    assert task.succeeded({"wave": 11})
