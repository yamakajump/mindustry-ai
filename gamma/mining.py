"""Where to put drills on the ore that is actually there, and how to bring it home.

A fixed structure cannot do this, and the measurement is blunt: the one design the agent
had placed 21,330 drills across 25 episodes and 89.9% of them landed on bare ground. Ore
patches are generated blobs of arbitrary shape and position, so a five-drill pattern bred
on one map is a pattern for that map. It worked at all only because the agent stamped
hundreds of times an episode and roughly one shot in ten happened to touch ore.

Extraction has to be computed from the map. Factories do not: a graphite press is coal in
and graphite out, a geometry that owes nothing to the terrain, and a fixed design is
exactly the right tool there. That split is the whole reason this module exists alongside
the design library rather than replacing it.

The packing follows the approach the community mod [AutoDrill](https://github.com/Pointifix/AutoDrill)
describes: walk the candidate positions best-first by how much they would actually mine,
and keep a floor below which a drill is not worth placing. Both projects are GPL-3.0, so
reusing its code would have been allowed; this is written from the mechanic instead,
because the mod is Java for the game and this is Python for the trainer.

The mechanic, which is what the scoring has to respect: a drill covers a square, it mines
the ore type most common under it, and its speed rises with how many tiles of that type it
covers. So a drill's worth is the count of its dominant ore, not the count of ore.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np

#: Mindustry counts rotations anticlockwise from east.
_ROTATION = {(1, 0): 0, (0, 1): 1, (-1, 0): 2, (0, -1): 3}


@dataclass(frozen=True)
class Placement:
    """One block, where it goes and which way it faces."""

    x: int
    y: int
    block: str
    rotation: int = 0


def dominant(ore: np.ndarray, x: int, y: int, size: int) -> tuple[int, int]:
    """The ore a drill at (x, y) would mine, and how many tiles of it it covers.

    Returns `(kind, tiles)`, with kind zero and tiles zero when the square is bare. A drill
    mines the most common ore beneath it and ignores the rest, so a square split between
    two ores is worth the larger half rather than the whole.
    """
    rows, columns = ore.shape
    if x < 0 or y < 0 or x + size > columns or y + size > rows:
        return 0, 0

    patch = ore[y:y + size, x:x + size]
    kinds, counts = np.unique(patch[patch > 0], return_counts=True)
    if kinds.size == 0:
        return 0, 0
    best = int(np.argmax(counts))
    return int(kinds[best]), int(counts[best])


def pack(ore: np.ndarray, size: int = 2, minimum: int = 1, limit: int = 8,
         only: int | None = None,
         worth: dict[int, float] | None = None) -> list[tuple[int, int, int, int]]:
    """Non-overlapping drill positions, best first.

    Returns `(x, y, kind, tiles)` for each drill, ordered by what it mines.

    Greedy rather than optimal, and deliberately. Covering a blob with non-overlapping
    squares to maximise total yield is a packing problem, so the exact answer costs search
    time this runs inside an environment step and buys very little: on a blob of any size
    the best-first pass already takes the full squares first, and the squares it gives up
    are the ragged edges worth one or two tiles each.

    `minimum` is the floor below which a drill is not worth its copper. A drill on a single
    ore tile mines at a quarter of the rate of one on four and costs exactly the same.

    `worth` ranks the ores against each other, and leaving it out is a real choice rather
    than a default. Sand is minable from bare darksand, which generated maps are covered
    in, so a packer that counts tiles alone drills sand: measured on a first run, 252 of
    the 288 tiles under its drills were darksand against 32 of titanium. Sand is worth
    three tenths of a copper to the reward and titanium six, so that run was twenty times
    less useful than the same drills moved a few tiles.
    """
    rows, columns = ore.shape
    candidates: list[tuple[int, int, int, int]] = []
    for y in range(rows - size + 1):
        for x in range(columns - size + 1):
            kind, tiles = dominant(ore, x, y, size)
            if tiles < minimum or (only is not None and kind != only):
                continue
            score = tiles * (worth.get(kind, 1.0) if worth else 1.0)
            candidates.append((score, tiles, x, y, kind))

    candidates.sort(reverse=True)

    taken = np.zeros_like(ore, dtype=bool)
    chosen: list[tuple[int, int, int, int]] = []
    for _, tiles, x, y, kind in candidates:
        if len(chosen) >= limit:
            break
        if taken[y:y + size, x:x + size].any():
            continue
        taken[y:y + size, x:x + size] = True
        chosen.append((x, y, kind, tiles))
    return chosen


def path(passable: np.ndarray, start: tuple[int, int],
         goal: tuple[int, int], limit: int = 4000) -> list[tuple[int, int]] | None:
    """The shortest way from one tile to another, around whatever is in between.

    A* with a Manhattan heuristic, which on a four-connected grid of equal steps never
    overestimates, so the first path found is still the shortest. Short matters twice: a
    conveyor costs a copper per tile and every extra tile is another tile of line to build
    before a single ore moves.

    It was a plain breadth-first search, and the cap meant to bound a walled-off goal was
    bounding the ordinary case instead. Breadth-first spreads in every direction at once,
    so reaching a core sixty tiles away costs on the order of eleven thousand tiles
    explored, and the cap sat at four thousand: a search that could not reach past about
    thirty-five tiles no matter how open the ground was. It reported "no route" either way,
    which reads as terrain and was arithmetic. Measured over 183 episodes: 17,542 connects
    refused for no route, the single largest cause of a refused action in the run.

    The heuristic pulls the search towards the goal instead of around it, so open ground
    costs about the length of the path rather than its square, and the cap goes back to
    doing what it was for.
    """
    rows, columns = passable.shape
    sx, sy = start
    gx, gy = goal
    if not (0 <= sx < columns and 0 <= sy < rows):
        return None

    came: dict[tuple[int, int], tuple[int, int] | None] = {(sx, sy): None}
    cost = {(sx, sy): 0}
    queue = [(abs(sx - gx) + abs(sy - gy), 0, sx, sy)]
    seen = 0

    while queue:
        _, spent, x, y = heapq.heappop(queue)
        if spent > cost.get((x, y), spent):
            continue
        seen += 1
        if seen > limit:
            return None
        if (x, y) == (gx, gy):
            steps = [(x, y)]
            while came[steps[-1]] is not None:
                steps.append(came[steps[-1]])
            steps.reverse()
            return steps

        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if not (0 <= nx < columns and 0 <= ny < rows):
                continue
            # The goal is the core, which is not passable and does not need to be: the
            # last conveyor points at it rather than standing on it.
            if not passable[ny, nx] and (nx, ny) != (gx, gy):
                continue
            if spent + 1 >= cost.get((nx, ny), 1 << 30):
                continue
            cost[(nx, ny)] = spent + 1
            came[(nx, ny)] = (x, y)
            heapq.heappush(queue, (spent + 1 + abs(nx - gx) + abs(ny - gy),
                                   spent + 1, nx, ny))

    return None


def belt(steps: list[tuple[int, int]], block: str = "conveyor") -> list[Placement]:
    """Conveyors along a path, each facing the tile it sends to.

    The last step is dropped: it is the destination, and a conveyor standing on the core
    is both refused and pointless. Rotation comes from where a tile sends rather than from
    the leg it belongs to, because a corner taken from the leg points along the old axis
    and one wrong corner delivers nothing at all.
    """
    laid: list[Placement] = []
    for index in range(len(steps) - 1):
        x, y = steps[index]
        nx, ny = steps[index + 1]
        rotation = _ROTATION.get((nx - x, ny - y))
        if rotation is not None:
            laid.append(Placement(x, y, block, rotation))
    return laid
