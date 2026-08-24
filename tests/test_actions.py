"""Actions change the world, illegal ones are refused, and the economy is real."""

from __future__ import annotations

import numpy as np
import pytest

from gamma.bridge import Bridge


@pytest.fixture
def acting(bridge_server, bridge_ports):
    """A fresh match with tensors enabled, so effects can be seen on the map."""
    with Bridge(port=bridge_ports[0], tensor=True) as client:
        obs = client.reset("Ancient_Caldera", "survival")
        yield client, obs


def test_placing_a_block_puts_it_on_the_map(acting) -> None:
    client, obs = acting
    x, y = obs["core_x"] + 3, obs["core_y"]

    result = client.place("conveyor", x, y, rotation=2)
    assert result["action"]["applied"] is True

    ally = result["spatial"][client.channels.index("block_ally")]
    assert ally[y, x] == 1, "the block is not where it was placed"


def test_placing_deducts_the_cost(acting) -> None:
    """The economy has to be real, or the agent learns that building is free."""
    client, obs = acting
    before = obs["items"]["copper"]

    after = client.place("conveyor", obs["core_x"] + 3, obs["core_y"], rotation=2)
    assert after["action"]["applied"] is True
    assert after["items"]["copper"] == before - 1, "a conveyor costs one copper"


def test_breaking_removes_the_block(acting) -> None:
    client, obs = acting
    x, y = obs["core_x"] + 3, obs["core_y"]
    client.place("conveyor", x, y, rotation=2)

    result = client.demolish(x, y)
    assert result["action"]["applied"] is True
    ally = result["spatial"][client.channels.index("block_ally")]
    assert ally[y, x] == 0


@pytest.mark.parametrize(
    "action, expected",
    [
        ({"type": "place", "block": "unicorn", "x": 10, "y": 10}, "no such block"),
        ({"type": "place", "block": "conveyor", "x": 9999, "y": 9999}, "invalid placement"),
        ({"type": "break", "x": 9999, "y": 9999}, "nothing breakable"),
        ({"type": "fly"}, "unknown action"),
        ({"x": 1, "y": 1}, "missing 'type'"),
    ],
)
def test_illegal_actions_are_refused_not_raised(acting, action, expected) -> None:
    """Rejection is a normal outcome for an agent still learning what is legal.
    It must be data in the observation, never an exception that ends the episode."""
    client, _ = acting
    result = client.act(action)
    assert result["action"]["applied"] is False
    assert expected in result["action"]["reason"]


def test_cannot_build_on_top_of_the_core(acting) -> None:
    client, obs = acting
    result = client.act(
        {"type": "place", "block": "conveyor", "x": obs["core_x"], "y": obs["core_y"]}
    )
    assert result["action"]["applied"] is False


def test_affordable_blocks_reflect_the_wallet(acting) -> None:
    client, _ = acting
    affordable = client.affordable_blocks()
    assert "conveyor" in affordable
    assert "mechanical-drill" in affordable
    # A survival start cannot pay for late-game blocks.
    assert "thorium-reactor" not in affordable


@pytest.fixture
def building(bridge_server, bridge_ports):
    """A match with no waves in it, for the one test that has to run for ten game minutes.

    In survival the enemy arrives long before a mechanical drill has filled a conveyor,
    and the run this test does is long enough that the base was regularly destroyed
    mid-measurement. It then reported that the chain had delivered nothing, which was true
    and told us nothing: a test that fails because the map killed it is measuring the map.
    """
    with Bridge(port=bridge_ports[0], tensor=True) as client:
        obs = client.reset("Ancient_Caldera", "sandbox")
        yield client, obs


def test_a_built_chain_delivers_ore_to_the_core(building) -> None:
    """The end to end proof, and curriculum stage T1 in miniature.

    A drill on an ore patch, a conveyor line to the core, and copper must actually
    arrive. This exercises every layer at once: action execution, the economy, the
    simulation running under acceleration, and the tensor used to find the ore.

    Delivery is read off the engine's own transport counter rather than off the core's
    stock. The counter answers the question this test actually asks, which is whether ore
    travelled down the conveyor, and it answers it whether the core happened to be full or
    empty at the time.
    """
    client, obs = building
    core_x, core_y = obs["core_x"], obs["core_y"]

    copper = obs["spatial"][client.channels.index("ore_copper")]
    ys, xs = np.where(copper > 0)
    assert len(xs) > 0, "no copper on this map"

    # Closest first, but not the closest only. A mechanical drill covers two tiles by two,
    # so the nearest ore tile is regularly one whose footprint runs into a wall or another
    # ore body, and the engine refuses it. Insisting on that one tile made this test fail
    # on a map it had no quarrel with.
    order = np.argsort((xs - core_x) ** 2 + (ys - core_y) ** 2)
    refusals = []
    ore_x = ore_y = None
    for candidate in order[:40]:
        x, y = int(xs[candidate]), int(ys[candidate])
        outcome = client.act(
            {"type": "place", "block": "mechanical-drill", "x": x, "y": y}
        )["action"]
        if outcome["applied"]:
            ore_x, ore_y = x, y
            break
        refusals.append(f"({x},{y}) {outcome.get('reason', 'no reason')}")

    assert ore_x is not None, "no copper tile took a drill: " + "; ".join(refusals[:5])

    # Build the path first, then orient each conveyor towards the next tile on it.
    # Deriving rotation from the segment instead would leave the corner tile pointing
    # along the old axis, and a chain with one wrong corner delivers exactly nothing.
    path = []
    y, step = ore_y, (-1 if core_y < ore_y else 1)
    while y != core_y:
        y += step
        path.append((ore_x, y))
    x, step = ore_x, (1 if core_x > ore_x else -1)
    while x != core_x:
        x += step
        path.append((x, core_y))

    # Mindustry rotations: 0 right, 1 up, 2 left, 3 down.
    directions = {(1, 0): 0, (0, 1): 1, (-1, 0): 2, (0, -1): 3}
    conveyors = 0
    for index, (px, py) in enumerate(path):
        nx, ny = path[index + 1] if index + 1 < len(path) else (core_x, core_y)
        rotation = directions.get((np.sign(nx - px), np.sign(ny - py)))
        if rotation is None:
            continue
        outcome = client.act({
            "type": "place", "block": "conveyor",
            "x": px, "y": py, "rotation": int(rotation),
        })["action"]
        conveyors += bool(outcome["applied"])

    # Runs until delivery is observed rather than for a fixed budget. A drill's output
    # depends on how far the chain had to run, and a fixed number of steps made this pass
    # alone and fail in a full suite for no reason anyone could see.
    result = client.observe()
    for _ in range(120):
        result = client.step(repeat=300)
        if result["produced"].get("copper", 0) > 0:
            break

    delivered = result["produced"].get("copper", 0)
    assert delivered > 0, "the chain delivered nothing after 36000 ticks"

    # The same counters the frontier reward is built on, checked here against a real chain
    # rather than a made-up observation. `produced` counts only what came down a transport
    # block, never what a unit carried in by hand, which is the distinction the whole
    # reward rests on.
    # Counted against what the engine actually accepted, not against what was asked for.
    # A drill that lands next to the core needs no conveyor at all, and the run where this
    # test insisted on one was a test measuring where the ore happened to be.
    assert result["placed"].get("production", 0) == 1, "the drill was not counted as placed"
    assert result["placed"].get("distribution", 0) == conveyors, "conveyors miscounted"
