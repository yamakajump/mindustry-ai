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
    #: Ore sitting inside the design, going nowhere. The difference between close and
    #: hopeless, and the only thing an incomplete line has to offer.
    stuck: int = 0

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


#: How much stuck ore the partial credit will look at, and no more.
#:
#: The cap is the whole safety of the term. Uncapped, a sprawling design full of conveyors
#: holds thousands of items and scores on all of them: measured over eighty generations,
#: the population settled at a mean fitness of 182 of which **89% was ore going nowhere**,
#: against about 21 actually delivered. The search had stopped building lines and started
#: hoarding. Sixty is roughly what a fifteen-tile line holds, so a design can still climb
#: out of zero on it and can never live on it.
STUCK_CAP = 60


def fitness(layout: Layout, block_cost: float = 0.01, stuck_worth: float = 0.05) -> float:
    """What a layout is worth: what it delivered, less what it took to build.

    Charged on what the engine accepted rather than on what the genome asked for. A
    candidate drawn at random asks for plenty that cannot exist, and billing it for
    placements that were refused punishes it for something it never built.

    The coefficient is small and it has to be. A first run at 0.05 per block cost a
    seventy-block design 3.5 against the 3 ore it delivered, so the search was being paid
    to build nothing at all and the fittest layout in the population was the empty one.
    The cost is a tie-break between designs that work, never a reason not to work.

    Ore stuck inside the design counts for a twentieth of ore delivered, up to a cap. The
    term is what makes the search climbable at all: a line one tile short of the core
    delivers nothing, exactly like an empty rectangle, so without it every candidate scores
    the same and the only pressure left is to build less. Measured before it existed:
    twenty-five generations, both genomes, nothing delivered, and the population shrank to
    four blocks and stopped.

    The cap is not a detail. Without it the same term was worth more than the objective:
    over eighty generations the population settled at a mean of 182 of which 89% was ore
    going nowhere, against 21 delivered, and the search had quietly stopped building lines
    in favour of hoarding. A hint that pays better than the goal is not a hint.
    """
    if layout.delivered is None:
        return float("-inf")
    return (layout.delivered
            + stuck_worth * min(getattr(layout, "stuck", 0), STUCK_CAP)
            - block_cost * (layout.cost or layout.used()))


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

# A second way of writing a layout ------------------------------------------------------

#: Direction of travel to Mindustry rotation: 0 right, 1 up, 2 left, 3 down.
_ROTATION = {(1, 0): 0, (0, 1): 1, (-1, 0): 2, (0, -1): 3}


@dataclass(frozen=True)
class Drill:
    """A drill, somewhere in the rectangle."""

    x: int
    y: int


@dataclass(frozen=True)
class Path:
    """A run of conveyors from one point to another, elbowed once.

    Rotations are derived from the direction of travel rather than drawn, which is the
    whole point: a path is correct by construction. The cell-by-cell genome had to roll
    four rotations right in a row to make three tiles of line, and a ten-tile line one
    time in a million. Here a line of any length costs one gene and is never wrong.

    This is geometry, not design. Which drills, which endpoints, how many, and whether any
    of it is worth building are left entirely to the search.
    """

    x0: int
    y0: int
    x1: int
    y1: int
    #: True to travel along x first, then y. The elbow is the only choice a straight
    #: corridor between two points still has, so it is worth a bit of the genome.
    horizontal_first: bool

    def tiles(self):
        """The line, as (x, y, rotation) in order of travel."""
        corner = (self.x1, self.y0) if self.horizontal_first else (self.x0, self.y1)

        points = [(self.x0, self.y0)]
        for tx, ty in (corner, (self.x1, self.y1)):
            x, y = points[-1]
            while x != tx:
                x += 1 if tx > x else -1
                points.append((x, y))
            while y != ty:
                y += 1 if ty > y else -1
                points.append((x, y))

        for index, (x, y) in enumerate(points):
            if index + 1 < len(points):
                nx, ny = points[index + 1]
                rotation = _ROTATION.get((nx - x, ny - y), 0)
            elif index:
                # The last tile keeps the heading that brought it here, so a line ending
                # against the core still hands its items over instead of stopping short.
                px, py = points[index - 1]
                rotation = _ROTATION.get((x - px, y - py), 0)
            else:
                rotation = 0
            yield x, y, rotation


@dataclass
class Design:
    """A layout written as parts rather than as cells.

    The cell genome asks the search to spell every line one square at a time and gets what
    that deserves: a rectangle of noise with a few accidental connections in it, stuck at
    the first score it finds. This one asks for drills and lines, so every candidate is at
    least the *kind* of thing that can work, and the search spends its budget on which and
    where instead of on rediscovering that conveyors point somewhere.
    """

    width: int
    height: int
    drills: list = field(default_factory=list)
    paths: list = field(default_factory=list)

    delivered: int | None = None
    cost: int = 0
    stuck: int = 0

    def copy(self):
        return Design(self.width, self.height, list(self.drills), list(self.paths))

    def to_layout(self) -> Layout:
        """Flatten to a grid, so it can be stamped and scored like any other candidate.

        Drills go down first and paths after, because a path crossing a drill should break
        around it rather than swallow it: the engine would refuse the conveyor anyway, and
        what stood is what gets billed.
        """
        size = self.width * self.height
        blocks = [0] * size
        rotations = [0] * size

        drill = PALETTE.index("mechanical-drill")
        for part in self.drills:
            if 0 <= part.x < self.width and 0 <= part.y < self.height:
                blocks[part.y * self.width + part.x] = drill

        conveyor = PALETTE.index("conveyor")
        for path in self.paths:
            for x, y, rotation in path.tiles():
                if not (0 <= x < self.width and 0 <= y < self.height):
                    continue
                index = y * self.width + x
                if blocks[index] == drill:
                    continue
                blocks[index] = conveyor
                rotations[index] = rotation

        return Layout(self.width, self.height, blocks, rotations)

    def used(self) -> int:
        return self.to_layout().used()

    def cells(self):
        return self.to_layout().cells()


def random_design(width: int, height: int, rng: random.Random,
                  drills: int = 3, paths: int = 3) -> Design:
    """A handful of drills and a handful of lines, all placed at random.

    Small on purpose. A design starting with thirty parts has no room to grow into
    anything and every mutation is lost in the crowd; one starting with a few can be added
    to when adding pays.
    """
    def point():
        return rng.randrange(width), rng.randrange(height)

    return Design(
        width, height,
        [Drill(*point()) for _ in range(rng.randint(1, drills))],
        [Path(*point(), *point(), rng.random() < 0.5) for _ in range(rng.randint(1, paths))],
    )


def cross_designs(first: Design, second: Design, rng: random.Random) -> Design:
    """A child taking each parent's parts with even odds, then trimmed.

    Taking every part from both would double the design each generation until a candidate
    is a solid block of conveyors, which scores badly and takes longest to evaluate.
    """
    drills = [d for d in first.drills + second.drills if rng.random() < 0.5]
    paths = [p for p in first.paths + second.paths if rng.random() < 0.5]
    return Design(first.width, first.height,
                  drills[:12] or [Drill(0, 0)], paths[:12])


def mutate_design(design: Design, rng: random.Random, rate: float = 0.35) -> Design:
    """Nudge a part, add one, or drop one.

    Nudging matters more than it looks. A drill one tile off its ore delivers nothing and
    is one step from delivering everything, and a mutation that could only replace it
    outright would have to find the patch again from scratch.
    """
    changed = design.copy()

    def point():
        return rng.randrange(changed.width), rng.randrange(changed.height)

    def nudge(value: int, limit: int) -> int:
        return min(max(value + rng.randint(-2, 2), 0), limit - 1)

    for index, drill in enumerate(changed.drills):
        if rng.random() < rate:
            changed.drills[index] = Drill(nudge(drill.x, changed.width),
                                          nudge(drill.y, changed.height))
    for index, path in enumerate(changed.paths):
        if rng.random() < rate:
            changed.paths[index] = Path(
                nudge(path.x0, changed.width), nudge(path.y0, changed.height),
                nudge(path.x1, changed.width), nudge(path.y1, changed.height),
                path.horizontal_first if rng.random() > 0.25 else not path.horizontal_first,
            )

    if rng.random() < 0.3 and len(changed.drills) < 12:
        changed.drills.append(Drill(*point()))
    if rng.random() < 0.3 and len(changed.paths) < 12:
        changed.paths.append(Path(*point(), *point(), rng.random() < 0.5))
    if rng.random() < 0.2 and len(changed.drills) > 1:
        changed.drills.pop(rng.randrange(len(changed.drills)))
    if rng.random() < 0.2 and len(changed.paths) > 1:
        changed.paths.pop(rng.randrange(len(changed.paths)))

    return changed


@dataclass
class DesignPopulation(Population):
    """The same tournament and elitism, over designs instead of grids."""

    def seed(self):
        self.members = [random_design(self.width, self.height, self.rng)
                        for _ in range(self.size)]
        return self.members

    def advance(self):
        ranked = sorted(self.members, key=fitness, reverse=True)
        survivors = []
        for original in ranked[:self.elite]:
            kept = original.copy()
            kept.delivered, kept.cost = original.delivered, original.cost
            survivors.append(kept)

        children = []
        while len(children) + len(survivors) < self.size:
            child = cross_designs(self._pick(ranked), self._pick(ranked), self.rng)
            children.append(mutate_design(child, self.rng))

        self.members = survivors + children
        self.generation += 1
        return self.members
