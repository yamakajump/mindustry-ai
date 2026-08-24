"""Put a discovered structure where the ore actually is.

A design comes out of the search anchored on the core, because that is what it was scored
against. Measured on three worlds it had never seen: it delivered **264 items on every one
of them and zero copper on any of them**. The transport it discovered transfers perfectly.
Where it sits does not, because the drills land on whatever happens to lie at a fixed
offset from the base, and on another map that is sand.

So a design has to be split into the two things it was conflating:

- **the part that sits on ore**, which is a shape and travels,
- **the part that reaches the core**, which is a distance and does not.

The first is kept exactly as the search found it. The second is recomputed on arrival by a
router, which is geometry rather than design: it decides no layout, it walks from one point
to another. That is the whole of what is added by hand here, and it is the same L-shaped
walk `gamma/alpha.py` has always used.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gamma.library import Design, Placement

#: Direction of travel to Mindustry rotation: 0 right, 1 up, 2 left, 3 down.
_ROTATION = {(1, 0): 0, (0, 1): 1, (-1, 0): 2, (0, -1): 3}

#: Blocks whose job is to carry, as opposed to to produce. The split a design gets cut on.
CARRIERS = frozenset({"conveyor", "titanium-conveyor", "junction", "router"})


@dataclass(frozen=True)
class Anchored:
    """A design taken apart into a shape that travels and a distance that does not."""

    #: Producers, kept at their offsets from each other, to be dropped onto ore.
    producers: tuple[Placement, ...]
    #: Where the producers hand over, relative to the same origin.
    outlet: tuple[int, int]

    def __len__(self) -> int:
        return len(self.producers)


def split(design: Design) -> Anchored:
    """Separate what a design produces with from what it carries with.

    The outlet is the carrier nearest the core, which is where the structure was handing
    over when it was measured. Everything the search decided about how the drills sit
    around it is kept; the run of conveyor between there and the base is thrown away,
    because its length was a fact about one map.
    """
    producers = tuple(p for p in design.placements if p.block not in CARRIERS)
    carriers = [p for p in design.placements if p.block in CARRIERS]

    if not producers:
        raise ValueError(f"{design.name} produces nothing: there is no shape to move")

    # The carrier the producers hand over to, which is the one nearest *them*, not the
    # one nearest the core. Taking the far end of the trunk instead leaves the drills two
    # tiles from where the new line starts, connected to nothing, since the trunk they
    # were flanking is exactly what gets thrown away and recomputed.
    def to_producers(carrier):
        return min(abs(carrier.dx - p.dx) + abs(carrier.dy - p.dy) for p in producers)

    nearest = min(carriers, key=to_producers) if carriers else None
    outlet = (nearest.dx, nearest.dy) if nearest else (producers[0].dx, producers[0].dy)

    # Re-centre the shape on its own outlet, so placing it is "put the outlet here".
    ox, oy = outlet
    return Anchored(
        producers=tuple(Placement(p.dx - ox, p.dy - oy, p.block, p.rotation)
                        for p in producers),
        outlet=(0, 0),
    )


def route(start: tuple[int, int], goal: tuple[int, int],
          block: str = "conveyor") -> list[tuple[int, int, str, int]]:
    """An L of conveyors from one point towards another, each facing where it sends.

    Lays on the start and on every tile up to but not including the goal, because the goal
    is the core: the last conveyor has to point *at* it, not stand on it.

    Rotation comes from the direction of the *next* tile, not of the current leg. Taking
    it from the leg leaves the corner pointing along the old axis, and one wrong corner
    means the line delivers nothing at all.
    """
    sx, sy = start
    gx, gy = goal

    # The whole walk, start included and goal included, so every tile knows what comes
    # next. The goal is dropped afterwards: it is the core, and building on it is both
    # refused and pointless.
    points: list[tuple[int, int]] = [(sx, sy)]
    x, y = sx, sy
    while y != gy:
        y += 1 if gy > y else -1
        points.append((x, y))
    while x != gx:
        x += 1 if gx > x else -1
        points.append((x, y))

    laid = []
    for index, (px, py) in enumerate(points[:-1]):
        nx, ny = points[index + 1]
        rotation = _ROTATION.get((int(np.sign(nx - px)), int(np.sign(ny - py))))
        if rotation is not None:
            laid.append((px, py, block, rotation))
    return laid


def best_anchor(ore: np.ndarray, shape: Anchored, core: tuple[int, int],
                reach: int = 24) -> tuple[int, int] | None:
    """Where to drop the shape so its producers sit on the most ore, closest to home.

    Both halves matter. A patch with more ore under the drills produces faster; a patch
    further away needs a longer line, which costs more to build and takes longer to fill.
    Ranking on ore alone sends the structure across the map for one extra tile of copper.
    """
    rows, columns = ore.shape
    cx, cy = core

    best: tuple[int, tuple[int, int]] | None = None
    for y in range(max(0, cy - reach), min(rows, cy + reach + 1)):
        for x in range(max(0, cx - reach), min(columns, cx + reach + 1)):
            covered = 0
            for producer in shape.producers:
                px, py = x + producer.dx, y + producer.dy
                if 0 <= px < columns and 0 <= py < rows and ore[py, px] > 0:
                    covered += 1
            if not covered:
                continue
            distance = max(abs(x - cx), abs(y - cy))
            score = covered * 100 - distance
            if best is None or score > best[0]:
                best = (score, (x, y))

    return None if best is None else best[1]


def lay(design: Design, ore: np.ndarray, core: tuple[int, int],
        reach: int = 24, copies: int = 1) -> list[tuple[int, int, str, int]]:
    """The whole thing, placed on a world it has never seen, once or several times.

    Drills first, then the line. A conveyor laid on a tile a drill wanted makes the drill
    impossible and leaves a run fed by nothing.

    Copies exist because one structure is not a base. Measured on three unseen worlds, a
    single one delivered about 128 copper against the scripted baseline's 134 to 159, at
    roughly twice the copper per block: more efficient and less ambitious, because the
    baseline works several patches and this worked one. How many to place, and where, is
    exactly the decision worth learning; this is the knob that decision would turn.
    """
    shape = split(design)
    remaining = ore.copy()
    laid: list[tuple[int, int, str, int]] = []

    for _ in range(max(1, copies)):
        anchor = best_anchor(remaining, shape, core, reach)
        if anchor is None:
            break

        ax, ay = anchor
        for producer in shape.producers:
            px, py = ax + producer.dx, ay + producer.dy
            laid.append((px, py, producer.block, producer.rotation))
            # Spent, so the next copy goes to a different patch rather than on top of
            # this one, where the engine would refuse it and the copy would be wasted.
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    if 0 <= py + dy < remaining.shape[0] and 0 <= px + dx < remaining.shape[1]:
                        remaining[py + dy, px + dx] = 0

        laid.extend(route(anchor, core))

    return laid
