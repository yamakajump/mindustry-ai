"""The search that invents layouts, checked without a game running.

Everything here is arithmetic over a genome, so it needs no server. What it protects is
the part that is easy to get subtly wrong and impossible to notice from a fitness curve:
a crossover that severs the structures it was meant to inherit, an elite that quietly
loses its measurement, a tie-break that never prefers the smaller design.
"""

from __future__ import annotations

import random

import pytest

from gamma.evolve import (
    PALETTE,
    Layout,
    Population,
    cross,
    fitness,
    mutate,
    random_layout,
)


def solid(width: int, height: int, block: str, rotation: int = 0) -> Layout:
    index = PALETTE.index(block)
    size = width * height
    return Layout(width, height, [index] * size, [rotation] * size)


def test_a_layout_refuses_a_genome_of_the_wrong_size() -> None:
    """Silent truncation here would shift every cell after the missing one, which reads
    as a mutation that never happened."""
    with pytest.raises(ValueError, match="wants 9 cells"):
        Layout(3, 3, [0, 0, 0], [0, 0, 0])


def test_cells_are_reported_in_game_coordinates() -> None:
    layout = Layout(3, 2, [0, PALETTE.index("conveyor"), 0, 0, 0, PALETTE.index("mechanical-drill")],
                    [0, 2, 0, 0, 0, 0])
    assert sorted(layout.cells()) == [
        (1, 0, "conveyor", 2),
        (2, 1, "mechanical-drill", 0),
    ]


def test_a_random_layout_leaves_room_to_build_in() -> None:
    """A solid rectangle of blocks has nowhere to put the drill that would feed it, and
    the engine refuses overlapping placements anyway."""
    rng = random.Random(0)
    layouts = [random_layout(8, 8, rng, density=0.5) for _ in range(20)]
    filled = sum(layout.used() for layout in layouts) / (20 * 64)
    assert 0.35 < filled < 0.65


def test_only_conveyors_carry_a_rotation() -> None:
    """A router faces every way at once and a junction passes items straight through.
    Rolling a rotation for either spends variation on something the game ignores."""
    rng = random.Random(1)
    layout = random_layout(12, 12, rng, density=1.0)
    for index, block in enumerate(layout.blocks):
        if PALETTE[block] not in ("conveyor",):
            assert layout.rotations[index] == 0, PALETTE[block]


def test_crossover_takes_cells_from_both_parents() -> None:
    """Uniform, not single-point. A layout is two-dimensional and a cut through a flat
    list slices it across rows, severing every line that runs down the grid."""
    first = solid(6, 6, "conveyor", rotation=0)
    second = solid(6, 6, "mechanical-drill")
    child = cross(first, second, random.Random(3))

    kinds = {PALETTE[block] for block in child.blocks}
    assert kinds == {"conveyor", "mechanical-drill"}


def test_crossover_leaves_its_parents_alone() -> None:
    first, second = solid(4, 4, "conveyor"), solid(4, 4, "router")
    before = list(first.blocks)
    cross(first, second, random.Random(0))
    assert first.blocks == before


def test_mutation_can_turn_a_conveyor_without_replacing_it() -> None:
    """A line that is right except for one tile facing the wrong way is one rotation from
    working. A mutation that could only swap the whole cell would have to rediscover it."""
    layout = solid(10, 10, "conveyor", rotation=0)
    turned = mutate(layout, random.Random(2), rate=1.0)

    kept = [index for index, block in enumerate(turned.blocks)
            if PALETTE[block] == "conveyor" and turned.rotations[index] != 0]
    assert kept, "no conveyor ever changed direction in place"


def test_an_unmeasured_layout_never_wins() -> None:
    assert fitness(solid(3, 3, "conveyor")) == float("-inf")


def test_the_smaller_of_two_equal_designs_wins() -> None:
    """Without this the population settles on whichever sprawling mess was found first
    and never tidies it up, because everything that works scores the same."""
    big = solid(4, 4, "conveyor")
    big.delivered = 100
    small = Layout(4, 4, [0] * 16, [0] * 16)
    small.blocks[0] = PALETTE.index("conveyor")
    small.delivered = 100

    assert fitness(small) > fitness(big)


def test_the_elite_keep_their_measurement_across_a_generation() -> None:
    """Re-running a layout already measured costs a whole evaluation to learn something
    known. At roughly a second each, that is the difference between twenty generations in
    an hour and fifteen."""
    population = Population(4, 4, size=8, elite=2, rng=random.Random(0))
    population.seed()
    for index, layout in enumerate(population.members):
        layout.delivered = index * 10

    population.advance()
    survivors = population.members[:2]

    assert [layout.delivered for layout in survivors] == [70, 60]


def test_a_generation_keeps_its_size() -> None:
    population = Population(5, 5, size=12, elite=3, rng=random.Random(0))
    population.seed()
    for layout in population.members:
        layout.delivered = 0

    for _ in range(3):
        population.advance()
        assert len(population.members) == 12
        for layout in population.members:
            layout.delivered = 1


def test_the_best_is_the_best_measured_not_the_best_guessed() -> None:
    population = Population(3, 3, size=4, rng=random.Random(0))
    population.seed()
    population.members[0].delivered = 5
    population.members[1].delivered = 40
    # The other two were never run, and must not win by default.
    assert population.best() is population.members[1]


def test_a_population_where_nothing_worked_still_breeds() -> None:
    """The first generations deliver nothing at all. A selection that cannot choose
    between zeros would stall before the search ever starts."""
    population = Population(6, 6, size=10, rng=random.Random(4))
    population.seed()
    for layout in population.members:
        layout.delivered = 0

    population.advance()
    assert len(population.members) == 10
    assert population.generation == 1


def test_an_elite_keeps_its_cost_as_well_as_its_delivery() -> None:
    """Copying only the delivery left every survivor billed for nothing, so a design
    charged for seventy blocks came back free and outranked the honest candidates."""
    population = Population(4, 4, size=6, elite=2, rng=random.Random(0))
    population.seed()
    for index, layout in enumerate(population.members):
        layout.delivered = index
        layout.cost = 10 * index

    population.advance()
    assert [(l.delivered, l.cost) for l in population.members[:2]] == [(5, 50), (4, 40)]


def test_building_nothing_is_never_the_best_answer() -> None:
    """At 0.05 per block a seventy-block design cost 3.5 against the 3 ore it delivered,
    so the search was paid to build nothing and the empty layout won its generation."""
    empty = Layout(13, 13, [0] * 169, [0] * 169)
    empty.delivered, empty.cost = 0, 0

    working = solid(13, 13, "conveyor")
    working.delivered, working.cost = 3, 70

    assert fitness(working) > fitness(empty)
