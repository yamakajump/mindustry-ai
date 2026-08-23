"""The spatial tensor describes the world the agent is actually looking at."""

from __future__ import annotations

import numpy as np
import pytest

from gamma.bridge import Bridge


@pytest.fixture
def seeing(bridge_server, bridge_ports):
    """A connection that negotiated spatial tensors, on a loaded map."""
    with Bridge(port=bridge_ports[0], tensor=True) as client:
        client.reset("Ancient_Caldera", "survival")
        yield client


def test_tensor_is_absent_unless_requested(bridge_server, bridge_ports) -> None:
    """Tensors are hundreds of kilobytes. An agent that never asked must not pay."""
    with Bridge(port=bridge_ports[0], tensor=False) as blind:
        obs = blind.reset("Ancient_Caldera", "survival")
        assert "spatial" not in obs
        assert "tensor" not in obs


def test_tensor_shape_matches_the_map(seeing: Bridge) -> None:
    obs = seeing.observe()
    spatial = obs["spatial"]
    assert spatial.dtype == np.uint8
    assert spatial.ndim == 3
    channels, height, width = spatial.shape
    assert channels == len(seeing.channels)
    assert (width, height) == (obs["map_width"], obs["map_height"])


def test_channels_include_the_ores_present_on_the_map(seeing: Bridge) -> None:
    """Ore channels only exist once a map is loaded, so this also guards the bug
    where the handshake advertised a layout the tensor no longer matched."""
    assert "ore_copper" in seeing.channels
    assert "ore_lead" in seeing.channels
    assert len(seeing.channels) > 8, "no per-ore channels were appended"


def test_ore_channels_mark_real_deposits(seeing: Bridge) -> None:
    spatial = seeing.observe()["spatial"]
    copper = spatial[seeing.channels.index("ore_copper")]
    assert copper.max() == 1
    assert copper.sum() > 100, "a survival map should carry a fair amount of copper"


def test_the_core_appears_as_an_allied_building(seeing: Bridge) -> None:
    obs = seeing.observe()
    spatial = obs["spatial"]
    block = spatial[seeing.channels.index("block")]
    ally = spatial[seeing.channels.index("block_ally")]
    enemy = spatial[seeing.channels.index("block_enemy")]

    assert ally.sum() > 0, "the starting core is missing from the tensor"
    assert enemy.sum() == 0, "survival starts with no enemy buildings"
    # Ownership channels must partition the block channel, not overlap it.
    assert np.array_equal(block, np.maximum(ally, enemy))

    # And it sits where the scalar observation says it does.
    assert ally[obs["core_y"], obs["core_x"]] == 1


def test_building_health_is_full_at_the_start(seeing: Bridge) -> None:
    spatial = seeing.observe()["spatial"]
    health = spatial[seeing.channels.index("block_health")]
    ally = spatial[seeing.channels.index("block_ally")]
    assert health[ally > 0].min() == 255, "an untouched core should read as full health"
    assert health[ally == 0].max() == 0, "health leaked onto tiles with no building"


def test_solid_and_buildable_disagree(seeing: Bridge) -> None:
    """A sanity check on the terrain channels: walls are not buildable floor."""
    spatial = seeing.observe()["spatial"]
    solid = spatial[seeing.channels.index("solid")]
    buildable = spatial[seeing.channels.index("buildable")]
    assert solid.sum() > 0
    assert buildable.sum() > 0
    assert solid.shape == buildable.shape


def test_tensor_is_stable_while_the_world_is_frozen(seeing: Bridge) -> None:
    first = seeing.observe()["spatial"]
    second = seeing.observe()["spatial"]
    assert np.array_equal(first, second), "the tensor changed with no ticks in between"
