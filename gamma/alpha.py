"""Alpha: the scripted baseline.

It plays badly but predictably, and it is the yardstick everything else is measured
against. A learned agent that cannot beat Alpha has not learned to play, it has learned
to be refused less often.

The strategy is the obvious one a human would try first: find the nearest copper, put a
drill on it, run a conveyor line back to the core, then repeat on the next patch. No
defence, no refinement, no reaction to anything. That is the point: it sets a floor that
is clearly beatable, not a target that is hard to reach.

Alpha is also where the macro library described in
`docs/decisions/0001-full-action-space.md` starts. The routines here are exactly the
"place a drill on the richest reachable patch" and "route a conveyor from A to B" that
will later be exposed to a learned agent as optional macro actions.
"""

from __future__ import annotations

from typing import Any, Iterator

import numpy as np

from gamma.env import ACTION_TYPES

#: Mindustry rotations, indexed by (dx, dy) of the direction of travel.
_ROTATIONS = {(1, 0): 0, (0, 1): 1, (-1, 0): 2, (0, -1): 3}

NOOP = np.zeros(5, dtype=np.int64)


def _route(start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int, int]]:
    """An L-shaped path from start to goal, as (x, y, rotation) triples.

    Rotation is derived from the direction to the *next* tile on the path, not from the
    current leg. Deriving it per leg leaves the corner tile pointing along the old axis,
    and one wrong corner means the chain delivers nothing at all.
    """
    sx, sy = start
    gx, gy = goal

    path: list[tuple[int, int]] = []
    y, step = sy, (1 if gy > sy else -1)
    while y != gy:
        y += step
        path.append((sx, y))
    x, step = sx, (1 if gx > sx else -1)
    while x != gx:
        x += step
        path.append((x, gy))

    placements = []
    for index, (px, py) in enumerate(path):
        nx, ny = path[index + 1] if index + 1 < len(path) else (gx, gy)
        rotation = _ROTATIONS.get((int(np.sign(nx - px)), int(np.sign(ny - py))))
        if rotation is not None:
            placements.append((px, py, rotation))
    return placements


class AlphaPolicy:
    """Scripted mining: drill the nearest copper, wire it to the core, repeat."""

    #: Four patches is what measurement favoured: fewer under-produces, more spends so
    #: much copper on conveyor line that the stock never recovers within the budget.
    def __init__(self, env, ore: str = "ore_copper", patches: int = 4) -> None:
        self.env = env
        self.ore = ore
        self.patches = patches
        self._plan: Iterator[np.ndarray] | None = None
        self._used: set[tuple[int, int]] = set()

    # Planning --------------------------------------------------------------------

    def _block_index(self, name: str) -> int:
        return self.env.blocks.index(name)

    def _plan_actions(self, info: dict[str, Any]) -> Iterator[np.ndarray]:
        raw = info["raw"]
        spatial = raw["spatial"]
        channels = self.env._bridge.channels
        core = (int(raw["core_x"]), int(raw["core_y"]))

        if self.ore not in channels:
            return
        ore_map = spatial[channels.index(self.ore)]
        free = info["action_mask"]["position"]

        ys, xs = np.nonzero(ore_map & (free > 0))
        if len(xs) == 0:
            return

        order = np.argsort((xs - core[0]) ** 2 + (ys - core[1]) ** 2)
        chosen = 0
        for index in order:
            spot = (int(xs[index]), int(ys[index]))
            if spot in self._used:
                continue
            # Keep patches apart so a second drill does not land on the first one's line.
            if any(abs(spot[0] - ux) + abs(spot[1] - uy) < 6 for ux, uy in self._used):
                continue

            self._used.add(spot)
            chosen += 1

            yield np.array(
                [ACTION_TYPES.index("place"), self._block_index("mechanical-drill"),
                 spot[0], spot[1], 0],
                dtype=np.int64,
            )
            for x, y, rotation in _route(spot, core):
                yield np.array(
                    [ACTION_TYPES.index("place"), self._block_index("conveyor"),
                     x, y, rotation],
                    dtype=np.int64,
                )

            if chosen >= self.patches:
                return

    # Policy ----------------------------------------------------------------------

    def act(self, observation: dict[str, np.ndarray], info: dict[str, Any]) -> np.ndarray:
        if self._plan is None:
            self._plan = self._plan_actions(info)

        for action in self._plan:
            return action

        # Plan exhausted: stand still and let the factory run. Doing nothing is a real
        # strategy here, and an agent that keeps fidgeting only wastes copper.
        return NOOP

    def reset(self) -> None:
        self._plan = None
        self._used.clear()
