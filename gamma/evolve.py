"""Layouts the agent invents for itself.

A conveyor line from a drill to the core is a structure that pays nothing until it is
complete. Measured over 177 archived episodes of a real run: 4,481 drills placed, 5,719
conveyors placed, and **one** episode where the tiles ever met end to end. A policy
choosing tiles one at a time is not going to find this, and no amount of training time
changes that, because the thing it is searching for has probability near zero of being
stumbled into.

So this searches for the structure directly, and separately. A population of layouts is
generated, each one is stamped into a real game and judged on what it actually delivers,
and the ones that deliver breed. Nothing here knows what a conveyor is for. The rules of
the game are the fitness function, and whatever comes out is the agent's own, not a
blueprint copied off somebody who already knew the answer.

The approach is not invented here either. [Reid et al.
(2021)](https://arxiv.org/abs/2102.04871) benchmarked simulated annealing, genetic
programming and evolutionary reinforcement learning on exactly this problem in Factorio,
the "logistic transport belt problem", and found search beating general-purpose learning
on it. This module is the search half; the learned policy keeps the half it is good at,
which is deciding where and when.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

#: The vocabulary a layout is written in. Deliberately tiny.
#:
#: Five blocks and four rotations is already 21 states per cell, so an 8x8 layout has
#: 21^64 of them. The point is not to make the space small enough to enumerate; it is to
#: make every candidate cheap to evaluate, so a population can cover ground the policy
#: never could.
PALETTE: tuple[str, ...] = ("air", "conveyor", "mechanical-drill", "junction", "router")

#: Blocks whose rotation means something. A router faces every way at once and a junction
#: passes items straight through, so rolling a rotation for either is wasted variation.
ROTATES: frozenset[str] = frozenset({"conveyor"})


@dataclass
class Layout:
    """One candidate: a rectangle of blocks and rotations.

    Stored flat, row-major, because every operation here is either a whole-genome walk or
    a single-cell change and neither wants the index arithmetic.
    """

    width: int
    height: int
    blocks: list[int]
    rotations: list[int]

    #: What it delivered, once measured. None until it has been run.
    delivered: int | None = None
    #: Blocks it costs to build, which breaks ties towards the smaller design.
    cost: int = 0

    def __post_init__(self) -> None:
        expected = self.width * self.height
        if len(self.blocks) != expected or len(self.rotations) != expected:
            raise ValueError(
                f"a {self.width}x{self.height} layout wants {expected} cells, "
                f"got {len(self.blocks)} blocks and {len(self.rotations)} rotations"
            )

    def cells(self):
        """Every non-empty cell, as (x, y, block name, rotation)."""
        for index, block in enumerate(self.blocks):
            if PALETTE[block] == "air":
                continue
            yield index % self.width, index // self.width, PALETTE[block], self.rotations[index]

    def used(self) -> int:
        return sum(1 for block in self.blocks if PALETTE[block] != "air")

    def copy(self) -> Layout:
        return Layout(self.width, self.height, list(self.blocks), list(self.rotations))


def random_layout(width: int, height: int, rng: random.Random, density: float = 0.5) -> Layout:
    """A layout drawn at random, with most cells left empty.

    Filling every cell sounds like more exploration and is less: a solid block of
    conveyors has no room for the drill that would feed it, and the engine refuses
    overlapping placements anyway. Leaving half of it empty makes the average candidate
    something that can physically exist.
    """
    size = width * height
    blocks, rotations = [], []
    for _ in range(size):
        if rng.random() > density:
            blocks.append(0)
            rotations.append(0)
            continue
        choice = rng.randrange(1, len(PALETTE))
        blocks.append(choice)
        rotations.append(rng.randrange(4) if PALETTE[choice] in ROTATES else 0)
    return Layout(width, height, blocks, rotations)


def cross(first: Layout, second: Layout, rng: random.Random) -> Layout:
    """One child, taking each cell from one parent or the other.

    Uniform rather than single-point on purpose. A layout is two-dimensional and a
    single-point cut on a flat list slices it across rows, which severs every line running
    down the grid and destroys precisely the structures worth inheriting.
    """
    child = first.copy()
    for index in range(len(child.blocks)):
        if rng.random() < 0.5:
            child.blocks[index] = second.blocks[index]
            child.rotations[index] = second.rotations[index]
    return child


def mutate(layout: Layout, rng: random.Random, rate: float = 0.04) -> Layout:
    """Change a few cells, so the population cannot settle on one answer forever.

    Both what a cell holds and which way it faces can change, and separately: a line that
    is right except for one tile facing the wrong way is one rotation from working, and a
    mutation that could only replace the whole cell would have to rediscover the line.
    """
    changed = layout.copy()
    for index in range(len(changed.blocks)):
        if rng.random() >= rate:
            continue
        if rng.random() < 0.4 and PALETTE[changed.blocks[index]] in ROTATES:
            changed.rotations[index] = rng.randrange(4)
            continue
        choice = rng.randrange(len(PALETTE))
        changed.blocks[index] = choice
        changed.rotations[index] = rng.randrange(4) if PALETTE[choice] in ROTATES else 0
    return changed


def fitness(layout: Layout, block_cost: float = 0.01) -> float:
    """What a layout is worth: what it delivered, less what it took to build.

    Charged on what the engine accepted rather than on what the genome asked for. A
    candidate drawn at random asks for plenty that cannot exist, and billing it for
    placements that were refused punishes it for something it never built.

    The coefficient is small and it has to be. A first run at 0.05 per block cost a
    seventy-block design 3.5 against the 3 ore it delivered, so the search was being paid
    to build nothing at all and the fittest layout in the population was the empty one.
    The cost is a tie-break between designs that work, never a reason not to work.
    """
    if layout.delivered is None:
        return float("-inf")
    return layout.delivered - block_cost * (layout.cost or layout.used())


@dataclass
class Population:
    """A generation of layouts, and the rules for making the next one."""

    width: int
    height: int
    size: int = 48
    elite: int = 6
    tournament: int = 3
    mutation: float = 0.04
    rng: random.Random = field(default_factory=random.Random)

    members: list[Layout] = field(default_factory=list)
    generation: int = 0

    def seed(self) -> list[Layout]:
        self.members = [random_layout(self.width, self.height, self.rng)
                        for _ in range(self.size)]
        return self.members

    def _pick(self, ranked: list[Layout]) -> Layout:
        """A parent, by tournament.

        Tournament rather than fitness-proportionate because early on almost every layout
        delivers exactly nothing, and a proportionate draw over a column of zeros is a
        uniform draw. A tournament still prefers the better of a few, whatever the scale.
        """
        contenders = self.rng.sample(ranked, min(self.tournament, len(ranked)))
        return max(contenders, key=fitness)

    def advance(self) -> list[Layout]:
        """Rank what was measured, keep the best, and breed the rest."""
        ranked = sorted(self.members, key=fitness, reverse=True)
        survivors = [layout.copy() for layout in ranked[:self.elite]]
        for layout, original in zip(survivors, ranked):
            # The elite keep their whole measurement, delivery and cost both. Copying only
            # the delivery left every survivor billed for nothing, so a design that had
            # been charged for seventy blocks came back as free and outranked the honest
            # candidates behind it.
            layout.delivered = original.delivered
            layout.cost = original.cost

        children = []
        while len(children) + len(survivors) < self.size:
            child = cross(self._pick(ranked), self._pick(ranked), self.rng)
            children.append(mutate(child, self.rng, self.mutation))

        self.members = survivors + children
        self.generation += 1
        return self.members

    def best(self) -> Layout | None:
        measured = [layout for layout in self.members if layout.delivered is not None]
        return max(measured, key=fitness) if measured else None
