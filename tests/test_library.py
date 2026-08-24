"""Storing a discovered design, and putting it back on a world it has never seen.

The whole value of the search is whether what it found survives leaving the bench. These
run without a server, because every question here is about coordinates.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from gamma.adapt import best_anchor, lay, route, split
from gamma.library import Design, Placement, from_evolution, load, save


def bench_report(tmp_path, cells, origin=(120, 21), core=(130, 32)):
    """A report shaped like the one `tools/evolve_layout.py` writes."""
    path = tmp_path / "run.json"
    path.write_text(json.dumps({
        "map": "Ancient_Caldera", "genome": "parts", "world_seed": 16, "item": "copper",
        "rectangle": [13, 13], "origin": list(origin), "core": list(core), "ticks": 1800,
        "best": {"delivered": 24, "blocks": len(cells), "layout": "", "cells": cells},
    }), encoding="utf-8")
    return path


def trunk() -> Design:
    """A drill beside a conveyor that carries away from it. The shape the search found."""
    return Design(
        name="trunk",
        placements=(
            Placement(-2, -8, "mechanical-drill", 0),
            Placement(0, -8, "mechanical-drill", 0),
            Placement(-1, -8, "conveyor", 1),
            Placement(-1, -7, "conveyor", 1),
            Placement(-1, -6, "conveyor", 1),
        ),
        delivered=24, item="copper", world_seed=16,
    )


# Storing it -----------------------------------------------------------------------------


def test_a_design_is_stored_relative_to_the_core(tmp_path) -> None:
    """The rectangle it was evolved in was scaffolding. Stored against that, a design
    could not be placed on a world whose core sits elsewhere, which is every world."""
    report = bench_report(tmp_path, [[10, 3, "mechanical-drill", 0]])
    design = from_evolution(report)

    # The cell sat at (130, 24) on the map, which is the core's column, eight tiles below.
    assert design.placements == (Placement(0, -8, "mechanical-drill", 0),)


def test_a_stored_design_comes_back_unchanged(tmp_path) -> None:
    path = tmp_path / "library.json"
    save([trunk()], path)
    [restored] = load(path)

    assert restored.placements == trunk().placements
    assert (restored.delivered, restored.item, restored.world_seed) == (24, "copper", 16)


def test_a_run_that_delivered_nothing_is_not_a_design(tmp_path) -> None:
    path = tmp_path / "empty.json"
    path.write_text(json.dumps({"origin": [0, 0], "core": [0, 0], "best": None}),
                    encoding="utf-8")
    with pytest.raises(ValueError, match="holds no design"):
        from_evolution(path)


def test_drills_are_placed_before_anything_else() -> None:
    """A drill needs two clear tiles by two. A conveyor laid first on a tile the drill
    wanted makes the drill impossible and leaves a line fed by nothing."""
    kinds = [a["block"] for a in trunk().actions((100, 100))]
    assert kinds.index("mechanical-drill") < kinds.index("conveyor")
    assert all(k == "mechanical-drill" for k in kinds[:2])


# Moving it ------------------------------------------------------------------------------


def test_a_design_splits_into_a_shape_and_a_distance() -> None:
    """Measured on three unseen worlds, a design anchored on the core delivered 264 items
    on every one and zero copper on any: the transport travels, the position does not."""
    shape = split(trunk())

    assert len(shape.producers) == 2
    assert all("drill" in p.block for p in shape.producers)
    # Re-centred on its own outlet, so placing it is "put the handover here".
    assert shape.outlet == (0, 0)
    assert sorted((p.dx, p.dy) for p in shape.producers) == [(-1, 0), (1, 0)]


def test_a_route_faces_where_it_sends() -> None:
    """Rotation comes from the direction of the next tile. Taken from the current leg, the
    corner points along the old axis and the line delivers nothing at all."""
    laid = route((4, 0), (4, 3))
    # On the start, up to but not including the goal: the goal is the core, and the last
    # conveyor has to point at it rather than stand on it.
    assert [(x, y, r) for x, y, _, r in laid] == [(4, 0, 1), (4, 1, 1), (4, 2, 1)]

    cornered = route((0, 0), (2, 2))
    assert cornered[-1][3] == 0, "the last tile should head towards the goal"
    assert (2, 2) not in [(x, y) for x, y, _, _ in cornered], "built on the goal"


def test_the_shape_lands_where_the_ore_is() -> None:
    ore = np.zeros((40, 40), dtype=np.uint8)
    ore[20, 9:12] = 1

    shape = split(trunk())
    anchor = best_anchor(ore, shape, core=(20, 30))

    assert anchor is not None
    covered = sum(1 for p in shape.producers if ore[anchor[1] + p.dy, anchor[0] + p.dx])
    assert covered == 2


def test_a_nearer_patch_wins_when_the_ore_is_equal() -> None:
    """Ranking on ore alone sends the structure across the map for the same copper, and
    pays for the extra line twice: once to build it and once in the time it takes to
    fill."""
    ore = np.zeros((60, 60), dtype=np.uint8)
    ore[30, 29:32] = 1
    ore[55, 29:32] = 1

    anchor = best_anchor(ore, split(trunk()), core=(30, 34))
    assert anchor is not None and abs(anchor[1] - 34) < 10


def test_a_world_with_no_ore_gets_nothing_built() -> None:
    """Better than laying a line to nowhere and paying for it."""
    assert lay(trunk(), np.zeros((30, 30), dtype=np.uint8), core=(15, 15)) == []


def test_copies_go_to_different_patches() -> None:
    """A second copy on top of the first is refused by the engine and wasted. One
    structure is also not a base: measured on three unseen worlds it delivered about 128
    against the scripted baseline's 134 to 159."""
    ore = np.zeros((60, 60), dtype=np.uint8)
    ore[25, 29:32] = 1
    ore[20, 39:42] = 1

    once = lay(trunk(), ore, core=(30, 30), copies=1)
    twice = lay(trunk(), ore, core=(30, 30), copies=2)

    drills = [(x, y) for x, y, block, _ in twice if "drill" in block]
    assert len(twice) > len(once)
    assert len(set(drills)) == len(drills), "two copies landed on the same tiles"


# The design as an action -----------------------------------------------------------------


def test_a_library_adds_one_action_type_and_no_more() -> None:
    """The policy gains "put a structure here". It keeps every primitive it had, so if it
    ever finds something better than the structure, nothing stops it."""
    from gamma.env import DIRECT_ACTION_TYPES, MindustryEnv

    bare = MindustryEnv.__new__(MindustryEnv)
    bare.embodied, bare.designs = False, ()
    assert bare.action_types == DIRECT_ACTION_TYPES

    stocked = MindustryEnv.__new__(MindustryEnv)
    stocked.embodied, stocked.designs = False, (trunk(),)
    assert stocked.action_types == DIRECT_ACTION_TYPES + ("stamp",)
    assert set(DIRECT_ACTION_TYPES) < set(stocked.action_types)


def test_more_designs_than_the_block_dimension_is_refused(tmp_path) -> None:
    """They share that dimension, and the mask is what sets the size of the network's
    head. Widening one without the other would surface as a shape error far from here."""
    from gamma.env import MindustryEnv

    with pytest.raises(ValueError, match="will not fit"):
        MindustryEnv(
            task=None, server_dir=str(tmp_path), blocks=("conveyor",),
            designs=(trunk(), trunk()),
        )
