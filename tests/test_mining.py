"""Placing drills on the ore that is there, and routing what they mine to the core.

The measurement these exist for: the one fixed design the agent had placed 21,330 drills
across 25 episodes, and 89.9% of them landed on bare ground. A pattern bred on one map is
a pattern for that map, and generated ore is a blob of arbitrary shape wherever the
generator put it. These run without a server, because the question is geometry.
"""

from __future__ import annotations

import numpy as np
import pytest

from gamma import mining


def blob(rows: int, columns: int, cells: list[tuple[int, int]], kind: int = 1) -> np.ndarray:
    ore = np.zeros((rows, columns), dtype=np.int16)
    for x, y in cells:
        ore[y, x] = kind
    return ore


def test_a_drill_is_worth_its_dominant_ore_not_its_ore() -> None:
    """A drill mines the commonest ore under it and ignores the rest.

    Scoring the total would rank a square split between two ores as if a drill could work
    both, and send it to a seam that pays half what the number promised.
    """
    ore = np.array([[1, 1], [2, 2]], dtype=np.int16)
    kind, tiles = mining.dominant(ore, 0, 0, 2)

    assert tiles == 2, "a two-two split is worth two tiles, not four"
    assert kind in (1, 2)


def test_bare_ground_is_worth_nothing() -> None:
    assert mining.dominant(np.zeros((4, 4), dtype=np.int16), 0, 0, 2) == (0, 0)


def test_a_drill_that_would_hang_off_the_edge_is_not_offered() -> None:
    """Its footprint has to fit, and the engine refuses it otherwise."""
    ore = blob(4, 4, [(3, 3)])
    assert mining.dominant(ore, 3, 3, 2) == (0, 0)


def test_drills_do_not_overlap() -> None:
    """Two drills on the same tile is one drill's worth of ore for two drills' copper."""
    ore = np.ones((8, 8), dtype=np.int16)
    chosen = mining.pack(ore, size=2, limit=16)

    covered = np.zeros_like(ore, dtype=bool)
    for x, y, _, _ in chosen:
        assert not covered[y:y + 2, x:x + 2].any()
        covered[y:y + 2, x:x + 2] = True


def test_the_fullest_squares_are_taken_first() -> None:
    """Best-first is the whole approach, so it has to actually be best-first."""
    ore = np.zeros((10, 10), dtype=np.int16)
    ore[0:2, 0:2] = 1          # a full square, worth four
    ore[5, 5] = 1              # a lone tile, worth one

    chosen = mining.pack(ore, size=2, minimum=1, limit=8)
    assert chosen[0][:2] == (0, 0)
    assert chosen[0][3] == 4


def test_a_floor_refuses_drills_that_do_not_pay() -> None:
    """A drill on one tile mines at a quarter of the rate and costs the same."""
    ore = np.zeros((10, 10), dtype=np.int16)
    ore[5, 5] = 1

    assert mining.pack(ore, size=2, minimum=1) != []
    assert mining.pack(ore, size=2, minimum=2) == []


def test_a_path_goes_round_a_wall_instead_of_through_it() -> None:
    """The old routing drew an L without looking, which lays a line through a cliff."""
    passable = np.ones((7, 7), dtype=bool)
    passable[:, 3] = False
    passable[6, 3] = True                       # one gap, at the bottom

    steps = mining.path(passable, (0, 0), (6, 0))
    assert steps is not None, "there is a way round and it was not found"
    assert all(passable[y, x] for x, y in steps)
    assert (3, 6) in steps, "the only gap is at the bottom, so the path must use it"


def test_a_walled_off_goal_gives_up_rather_than_lying() -> None:
    passable = np.ones((7, 7), dtype=bool)
    passable[:, 3] = False

    assert mining.path(passable, (0, 0), (6, 0)) is None


def test_the_goal_is_reachable_even_though_it_is_solid() -> None:
    """The core is not passable, and the line has to arrive at it anyway."""
    passable = np.ones((5, 5), dtype=bool)
    passable[2, 2] = False

    steps = mining.path(passable, (0, 2), (2, 2))
    assert steps is not None and steps[-1] == (2, 2)


def test_every_conveyor_points_at_the_next_one() -> None:
    """One corner taken from the leg rather than from the next tile delivers nothing."""
    steps = [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]
    laid = mining.belt(steps)

    assert len(laid) == len(steps) - 1, "the last tile is the core, not a conveyor"
    assert [p.rotation for p in laid] == [0, 0, 1, 1]


def test_the_line_stops_short_of_the_core() -> None:
    steps = [(0, 0), (1, 0), (2, 0)]
    assert [(p.x, p.y) for p in mining.belt(steps)] == [(0, 0), (1, 0)]


def test_a_real_blob_is_covered_rather_than_sprayed() -> None:
    """The whole point, stated as a number.

    A ragged patch of nineteen tiles, and the drills placed on it must sit on ore. The
    fixed design managed 10.1%.
    """
    cells = [(x, y) for y in range(4, 9) for x in range(4, 9) if (x + y) % 5]
    ore = blob(16, 16, cells)

    chosen = mining.pack(ore, size=2, minimum=2, limit=8)
    assert chosen, "a nineteen tile patch takes at least one drill"

    on_ore = sum(1 for x, y, _, _ in chosen if ore[y:y + 2, x:x + 2].any())
    assert on_ore == len(chosen), "every drill placed must touch the patch"
    assert sum(tiles for _, _, _, tiles in chosen) >= 8


def test_a_richer_ore_wins_over_more_tiles_of_a_poor_one() -> None:
    """Sand is minable from the bare ground generated maps are made of.

    A packer that counts tiles drills sand, because there is always more of it. Measured
    on a first run: 252 of the 288 tiles under its drills were darksand against 32 of
    titanium, and sand is worth three tenths of a copper to the reward where titanium is
    worth six.
    """
    ore = np.zeros((8, 8), dtype=np.int16)
    ore[0:2, 0:2] = 1          # four tiles of the cheap one
    ore[4:6, 4:6] = 2          # four tiles of the dear one

    by_tiles = mining.pack(ore, size=2, minimum=1, limit=1)
    by_worth = mining.pack(ore, size=2, minimum=1, limit=1, worth={1: 0.3, 2: 6.0})

    assert by_tiles[0][2] in (1, 2), "tied on tiles, so either is a fair answer"
    assert by_worth[0][2] == 2, "told what each is worth, it must take the titanium"


def test_worth_does_not_override_the_floor() -> None:
    """A precious ore under a single tile is still a drill running at a quarter speed."""
    ore = np.zeros((8, 8), dtype=np.int16)
    ore[5, 5] = 2

    assert mining.pack(ore, size=2, minimum=2, worth={2: 25.0}) == []


def test_an_open_run_of_sixty_tiles_is_found() -> None:
    """The cap was bounding the ordinary case, not the walled-off one.

    Breadth-first spreads in every direction at once, so a core sixty tiles away costs on
    the order of eleven thousand tiles explored against a cap of four thousand: a search
    that could not reach past about thirty-five tiles however open the ground. It reported
    "no route", which reads as terrain and was arithmetic. Measured over 183 episodes:
    17,542 connects refused for no route, the largest single cause of a refused action.
    """
    passable = np.ones((80, 80), dtype=bool)
    steps = mining.path(passable, (5, 40), (65, 40))

    assert steps is not None, "sixty tiles of open ground is not a walled-off goal"
    assert len(steps) == 61, "and the way there is still the shortest one"


def test_the_cap_still_bounds_a_hopeless_search() -> None:
    """It exists so a walled-off goal costs a bounded amount of time."""
    passable = np.ones((200, 200), dtype=bool)
    passable[:, 100] = False

    assert mining.path(passable, (0, 0), (199, 199)) is None


def test_a_three_by_three_core_can_be_arrived_at() -> None:
    """The goal exception covers the goal tile and nothing around it.

    A core is three tiles by three and every one of them is solid, so the only ways into
    the centre are the eight tiles surrounding it, which are the core as well. The search
    could reach the goal by exception and could never reach a neighbour of it: arriving at
    its own core was impossible by construction whenever the core sat inside the window.
    Outside it everything is assumed open, so the same connect succeeded or failed on
    where the agent happened to be standing. Measured after the walling was fixed, 9,849
    connects and 7,160 stamps were still refused for no route.
    """
    passable = np.ones((20, 20), dtype=bool)
    passable[9:12, 9:12] = False          # the core, solid through and through

    assert mining.path(passable, (2, 10), (10, 10)) is None, (
        "stated as it was: solid on all nine tiles and the centre is unreachable")

    passable[9:12, 9:12] = True           # allied ground, crossed rather than skirted
    steps = mining.path(passable, (2, 10), (10, 10))
    assert steps is not None and steps[-1] == (10, 10)
