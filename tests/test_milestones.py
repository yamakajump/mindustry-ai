"""What the milestone reward pays for, and what it refuses to pay for twice.

The whole design rests on two claims: that a rung of the ladder is paid once per episode
whatever the agent does afterwards, and that automating production always beats doing the
same work by hand. Both are arithmetic, so both can be checked without a server, and both
are exactly the sort of thing that breaks silently in a real run.
"""

from __future__ import annotations

import pytest

from gamma import tasks


def state(**fields) -> dict:
    """A frontier observation on the step after landing: a loadout and nothing else."""
    base = {
        "wave": 1,
        "has_core": True,
        "core_health": 4000.0,
        "core_max_health": 4000.0,
        "items": {"copper": 300, "lead": 300},
        "produced": {},
        "crafting": 0.0,
        "power": 0.0,
        "placed": {},
        "stats": {},
        "unit": {"carrying": 0},
    }
    for key, value in fields.items():
        base[key] = {**base[key], **value} if isinstance(base.get(key), dict) and isinstance(value, dict) else value
    return base


@pytest.fixture
def reward():
    return tasks.get("frontier").reward


# The ladder ---------------------------------------------------------------------------


def test_standing_still_earns_nothing(reward) -> None:
    assert reward(state(), state()) == 0.0


def test_the_first_drill_pays(reward) -> None:
    assert reward(state(), state(placed={"production": 1})) == pytest.approx(5.0)


def test_the_second_drill_pays_nothing(reward) -> None:
    """The rung is paid on the crossing, so a counter that keeps climbing earns nothing
    more. This is what makes the ladder unfarmable."""
    before = state(placed={"production": 1})
    assert reward(before, state(placed={"production": 2})) == 0.0
    assert reward(before, state(placed={"production": 40})) == 0.0


def test_breaking_and_rebuilding_pays_nothing(reward) -> None:
    """The counter is cumulative, so tearing a conveyor down and putting it back cannot
    take the milestone back and cannot earn it again. A count of what is standing would
    have paid every time."""
    once = state(placed={"distribution": 1})
    again = state(placed={"distribution": 2})
    assert reward(once, again) == 0.0


def test_the_first_automated_ore_is_the_biggest_single_step(reward) -> None:
    """The moment the project exists to reach: ore arrived without a hand carrying it."""
    step = reward(state(), state(produced={"copper": 1}))
    others = [
        reward(state(), state(placed={"production": 1})),
        reward(state(), state(placed={"turret": 1})),
        reward(state(), state(wave=5)),
    ]
    assert step > max(others)


def test_thresholds_crossed_in_one_step_all_pay(reward) -> None:
    """A step that jumps from nothing to a thousand ore collects every rung below it, not
    only the top one. Rewarding the highest crossed would make a slow climb worth more
    than a fast one."""
    once = reward(state(), state(produced={"copper": 1_000}))
    assert once == pytest.approx(30.0 + 30.0 + 60.0 + 1_000 * 0.1)


# Automation against hand mining -------------------------------------------------------


def test_a_machine_is_worth_fifty_hands(reward) -> None:
    """Same ore in the core, different origin. If hand mining ever paid as well, the agent
    would have no reason to build anything, and the gap widened when delivery was
    reweighted so that producing outearns hiding."""
    machine = reward(
        state(produced={"copper": 500}, items={"copper": 800}),
        state(produced={"copper": 600}, items={"copper": 900}),
    )
    hand = reward(
        state(produced={"copper": 500}, items={"copper": 800}),
        state(produced={"copper": 500}, items={"copper": 900}),
    )
    assert machine == pytest.approx(hand * 50)


def test_spending_on_a_drill_is_not_punished(reward) -> None:
    """Paying a hundred copper for a machine has to read as zero hand mining, not as
    negative. An agent penalised for building never builds."""
    before = state(items={"copper": 300})
    after = state(items={"copper": 200}, placed={"production": 1})
    assert reward(before, after) == pytest.approx(5.0)


# Holding the sector -------------------------------------------------------------------


def test_a_kill_pays_and_keeps_paying(reward) -> None:
    """Unlike the ladder, killing is a rate: a turret that keeps working keeps earning."""
    assert reward(state(), state(stats={"enemy_units_destroyed": 1})) == pytest.approx(22.0)
    assert reward(
        state(stats={"enemy_units_destroyed": 5}),
        state(stats={"enemy_units_destroyed": 6}),
    ) == pytest.approx(2.0)


def test_losing_the_core_costs_more_than_any_episode_earns_by_luck(reward) -> None:
    lost = reward(state(), state(has_core=False, core_health=0.0))
    assert lost < -50.0


def test_a_repaired_core_pays_nothing_back(reward) -> None:
    """Damage is one-way. Otherwise letting the core be hit and repairing it is income."""
    hurt = state(core_health=3_000.0)
    assert reward(hurt, state(core_health=4_000.0)) == 0.0


def test_buildings_lost_cost_something(reward) -> None:
    assert reward(state(), state(stats={"buildings_destroyed": 4})) == pytest.approx(-2.0)


# The control ---------------------------------------------------------------------------


def test_both_frontier_tasks_share_a_world_pool() -> None:
    """The A/B is only a comparison if the only difference is the reward."""
    milestones = tasks.get("frontier")
    potential = tasks.get("frontier_shaped")
    assert milestones.reward is not potential.reward
    for field in ("procedural", "threat_limit", "loadout", "ticks_per_step", "max_steps"):
        assert getattr(milestones, field) == getattr(potential, field)


def test_every_milestone_reads_a_counter_that_only_climbs() -> None:
    """The safety property, stated as a test: a rung whose counter can fall would be paid
    again every time it recovered."""
    empty = state()
    full = state(
        produced={"copper": 20_000, "lead": 5_000, "coal": 100},
        placed={c: 9 for c in ("production", "distribution", "defense", "turret",
                               "power", "crafting", "units")},
        stats={"enemy_units_destroyed": 99},
        wave=30,
    )
    for stone in tasks.MILESTONES:
        assert stone.read(empty) < stone.threshold <= stone.read(full), stone.name


# Work at the consumer, wherever the consumer is -------------------------------------------


def test_a_factory_doing_work_pays_without_anything_reaching_the_core(reward) -> None:
    """The hypothesis this reward used to rest on, and it was wrong: value is not arrival
    at the core. A conveyor feeding a graphite press delivers nothing there, and it is the
    point of the game."""
    assert reward(state(), state(crafting=12.0)) == pytest.approx(6.0)


def test_a_running_generator_pays_nothing_per_step(reward) -> None:
    """The opposite of what this test used to assert, because the old version was wrong.

    It read "instantaneous rather than cumulative: it says a fuel line exists right now",
    and asserted 0.2 for generation of 4. Both the sentence and the number describe an
    annuity: generation is a level, so a machine bought once collects that 0.2 on every
    remaining step, three thousand times over a full episode. A policy trained against it
    reached a mean of +205 and a best episode of +1577 while building, across 167 archived
    episodes, exactly one conveyor line that ever reached a core.

    Power is paid once by the `first_power` milestone and then through what it enables, a
    craft at a time and a kill at a time. Running is not an achievement it repeats.
    """
    assert reward(state(), state(power=4.0)) == pytest.approx(0.0)
    assert reward(state(power=4.0), state(power=40.0)) == pytest.approx(0.0)


def test_a_loop_of_conveyors_earns_nothing(reward) -> None:
    """Counting transfers would let a closed loop carrying one item forever pay without
    end. Counting work at the consumer cannot, because a loop has no consumer."""
    turning = state(items={"copper": 300}, produced={}, crafting=0.0, power=0.0)
    assert reward(turning, turning) == 0.0


def test_producing_outearns_hiding(reward) -> None:
    """The balance the old weights got backwards. Measured on thirty held-out episodes:
    the trained policy beat a coward's ceiling while delivering nothing, because a lost
    core cost fifty against a plausible ten for a whole episode of production."""
    produced_then_died = reward(
        state(), state(produced={"copper": 1_000}, has_core=False, core_health=0.0)
    )
    hid_and_survived = reward(state(), state(wave=8))

    assert produced_then_died > hid_and_survived
