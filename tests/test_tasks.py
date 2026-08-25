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


def test_no_task_pays_for_standing_still():
    """A step in which nothing changed must earn nothing, on every task.

    This is the invariant that separates a rate from a level, and it is not academic. The
    reward once paid `generation * 0.05` per step, meaning a generator bought once printed
    points for the remaining three thousand steps of the episode. The agent found it and
    the run looked healthy while it did: mean score climbing smoothly from +21 to +205 over
    two hundred updates, best episode +1577, and across 167 archived episodes exactly ONE
    conveyor line that ever reached a core.

    The rewards themselves gave it away, being exact multiples of 0.05 for thousands of
    consecutive steps, which is generation burning steadily while nothing else happened at
    all. Any term that reads a level rather than a difference fails here, immediately, on
    a static observation, without needing a training run to notice.
    """
    still = {
        "items": {"copper": 300, "lead": 300},
        "stats": {"enemy_units_destroyed": 4, "buildings_destroyed": 2},
        "produced": {"copper": 500},
        "placed": {"production": 3, "distribution": 9, "power": 2, "crafting": 1},
        "crafting": 12.5,
        "power": 11.0,
        "core_health": 900.0,
        "wave": 6,
        "has_core": True,
        "unit": {"carrying": 30},
    }

    for name, task in tasks.CURRICULUM.items():
        if name == "frontier_shaped":
            # Genuinely exempt, not excused. Potential-based shaping pays
            # `gamma * phi(after) - phi(before)`, so a static step pays
            # `(gamma - 1) * phi`, which is negative and cannot be collected: the sum over
            # an episode telescopes to `gamma^T * phi(last) - phi(first)` no matter what
            # the agent does in between. That is the theorem the whole design rests on.
            assert task.reward(still, still) <= 0.0
            continue

        assert task.reward(still, still) == pytest.approx(0.0), (
            f"{name} pays {task.reward(still, still)} for a step in which nothing changed, "
            "which an agent collects every step of the episode for free"
        )


def test_each_caller_gets_its_own_ledger():
    """A stateful reward must not be shared between environments.

    The curriculum is a module-level dictionary, so `get` used to hand every caller the
    same `Task` and therefore the same reward object. Harmless while a reward was a pure
    function; not harmless once it kept a per-episode ledger, because a training run puts
    twenty-four environments on twenty-four threads of one process.

    It surfaced as nonsense in the accounts rather than as a crash. Across 225 archived
    episodes the breakdown reported 1,515 core losses, up to nine in one episode, some on
    steps whose reward was exactly zero: one environment's itemisation read while another
    environment's step had overwritten it.
    """
    one, other = tasks.get("frontier"), tasks.get("frontier")
    assert one.reward is not other.reward

    still = {
        "wave": 1, "has_core": True, "core_health": 4000.0, "items": {}, "produced": {},
        "crafting": 0.0, "placed": {}, "stats": {}, "unit": {"carrying": 0},
    }
    one.reward(still, {**still, "produced": {"copper": 40}})

    assert one.reward.credited == pytest.approx(40.0)
    assert other.reward.credited == pytest.approx(0.0), (
        "a delivery credited in one environment was visible in another"
    )
