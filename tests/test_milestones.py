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
    # Remis a zero entre les tests : la recompense tient un registre par episode, et un
    # test qui herite du registre du precedent echoue sur ce que le precedent a fait.
    scorer = tasks.get("frontier").reward
    scorer.reset()
    return scorer


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


def test_thresholds_crossed_along_the_way_all_pay(reward) -> None:
    """Every rung below the one reached is collected, not only the top one. Paying only
    the highest crossed would make a slow climb worth more than a fast one.

    Delivered a rung at a time, because a single step cannot be credited more than
    `DELIVERY_CAP` items, which is the whole point of the ledger.
    """
    total = 0.0
    before = state()
    for delivered in range(50, 1_051, 50):
        after = state(produced={"copper": delivered})
        total += reward(before, after)
        before = after

    # automation, automation_100 and automation_1k, plus a copper apiece for the ore.
    assert total == pytest.approx(30.0 + 30.0 + 60.0 + 1_050 * 0.1)


def test_a_windfall_cannot_buy_a_rung(reward) -> None:
    """The reason the ledger exists.

    Something in a generated sector hands the core a whole stockpile in one step: 107 of
    237 archived episodes contained exactly one, median 1,194 items and up to 5,251,
    arriving with metaglass and graphite the agent cannot make. Those single steps carried
    22.1% of every point the run scored, and one traced case paid 99.57 on a step whose
    only action was to move, 80 of it milestones.

    A thousand ore at once is now credited fifty, so it crosses the first rung and nothing
    above it. Delivered honestly, the same thousand crosses three.
    """
    windfall = reward(state(), state(produced={"copper": 1_000, "metaglass": 400}))

    reward.reset()
    honest = 0.0
    before = state()
    for delivered in range(50, 1_001, 50):
        after = state(produced={"copper": delivered})
        honest += reward(before, after)
        before = after

    assert windfall < honest / 3


# Automation against hand mining -------------------------------------------------------


def test_a_machine_is_worth_fifty_hands(reward) -> None:
    """Same ore in the core, different origin. If hand mining ever paid as well, the agent
    would have no reason to build anything, and the gap widened when delivery was
    reweighted so that producing outearns hiding."""
    def primed():
        """A ledger already past the low rungs, so neither branch collects one.

        Without this the first fifty ore of the episode also cross `automation`, worth
        thirty, which swamps the two-tenths under comparison.
        """
        scorer = tasks.get("frontier").reward
        scorer.reset()
        before = state(produced={})
        for delivered in range(50, 501, 50):
            after = state(produced={"copper": delivered})
            scorer(before, after)
            before = after
        return scorer

    machine = primed()(
        state(produced={"copper": 500}, items={"copper": 800}),
        state(produced={"copper": 550}, items={"copper": 850}),
    )
    hand = primed()(
        state(produced={"copper": 500}, items={"copper": 800}),
        state(produced={"copper": 500}, items={"copper": 850}),
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
        credited=20_000.0,
        credited_variety=3,
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
    produced_then_died = 0.0
    before = state()
    for delivered in range(50, 1_001, 50):
        after = state(produced={"copper": delivered})
        produced_then_died += reward(before, after)
        before = after
    produced_then_died += reward(before, state(produced={"copper": 1_000},
                                               has_core=False, core_health=0.0))

    reward.reset()
    hid_and_survived = reward(state(), state(wave=8))

    assert produced_then_died > hid_and_survived


def test_delivery_pays_by_what_the_ore_cost_to_get() -> None:
    """Ten sand is not ten titanium, and the reward used to say it was.

    The counter behind delivery is a plain sum over items, so a drill sitting on a sand
    patch earned exactly what a titanium line earned. Measured over 73 episodes of a real
    run, by what ended up in the core: copper 51.7%, lead 18.9%, sand 14.2%. A seventh of
    the income came from the one resource that needs nothing but a drill and bare ground.

    Read through `terms` rather than the total, because the total also carries the
    automation milestones, which fire on how many items arrived and would drown the very
    difference under test.
    """
    terms = tasks.get("frontier").reward.terms
    paid = lambda produced: terms(state(), state(produced=produced))["delivered"]

    assert paid({"sand": 10}) == pytest.approx(0.3)
    assert paid({"copper": 10}) == pytest.approx(1.0)
    assert paid({"titanium": 10}) == pytest.approx(6.0)
    assert paid({"thorium": 10}) == pytest.approx(12.0)


def test_sand_does_not_buy_the_variety_milestones() -> None:
    """`two_ores` is worth 20 points and means two supply lines, not two drills.

    Both branches deliver the same number of items, so both cross the same automation
    rungs; the only thing that can differ between them is the variety milestone.
    """
    scorer = tasks.get("frontier").reward

    scorer.reset()
    before = state(produced={"copper": 50})
    scorer(state(), before)
    with_sand = scorer(before, state(produced={"copper": 50, "sand": 40}))

    scorer.reset()
    scorer(state(), before)
    with_lead = scorer(before, state(produced={"copper": 50, "lead": 40}))

    assert with_lead - with_sand == pytest.approx(20.0 + 40 * 0.7 * 0.1)


def test_reached_reads_the_ledger_not_the_observation() -> None:
    """The rungs that matter do not live in the observation, and two callers forgot.

    `automation` and `two_ores` moved onto the reward's per-episode ledger when a windfall
    turned out to be able to buy them. Anything checking `stone.read(obs)` afterwards
    reports them as never reached however well the episode went, because no observation
    carries a ledger. The evaluator and the dashboard were both doing exactly that.
    """
    task = tasks.get("frontier")
    task.reward.reset()

    landed = state(produced={"copper": 400, "lead": 400})
    task.reward(state(), landed)

    assert "automation" in tasks.reached(task, landed)
    assert "automation" not in {
        stone.name for stone in tasks.MILESTONES if stone.read(landed) >= stone.threshold
    }, "the observation alone cannot know, which is the whole point"


def test_tearing_down_your_own_work_costs_something(reward) -> None:
    """Demolishing was free, and the agent found that out the moment it could aim.

    `lost` counts what the enemy destroys and says nothing about what the agent dismantles.
    That was invisible while breaking was also impossible: the position mask offered
    `break` only empty tiles, so across 6,660 attempts it hit natural walls and bare ground
    and 23 of its own buildings. Masking positions by action type made breaking work, and
    the first thing the agent did with it was cover its own drills in demolition orders.
    """
    before = state(stats={"buildings_deconstructed": 4})
    after = state(stats={"buildings_deconstructed": 6})

    assert reward(before, after) == pytest.approx(-0.2), "two buildings torn down, an ore each"

    # A nudge and not a prohibition: what a demolition really costs or earns is what the
    # base delivers afterwards, and pricing it like a building lost in combat would make
    # revising an implantation dearer than leaving it to rot.
    torn = tasks.get("frontier").reward.terms(before, after)["torn"]
    assert abs(torn) < 0.5, "a player pulls things down to lay them again better"


def test_undoing_fresh_work_costs_more_than_revising_old_work(reward) -> None:
    """Two demolitions that look the same to a counter and are not the same act.

    Pulling down a line that has been running is what a player does to lay it again better.
    Pulling down a drill placed four steps ago is undoing your own work, and it was the
    cheapest action on the board: measured over 72 episodes once the position mask let
    `break` reach a building, 2,996 demolitions against 2,673 placements, 80% of them on
    the agent's own buildings, and 812 place-then-break cycles on the same tile.
    """
    before = state(stats={"buildings_deconstructed": 4})
    revision = state(stats={"buildings_deconstructed": 5})
    undoing = state(stats={"buildings_deconstructed": 5}, churn=1)

    assert reward(before, revision) == pytest.approx(-0.1)

    scorer = tasks.get("frontier").reward
    scorer.reset()
    assert scorer(before, undoing) == pytest.approx(-2.1)
