"""Alpha with a body: the scripted baseline for an agent that plays like a player.

The disembodied Alpha placed a drill and a conveyor line anywhere it liked, instantly.
With a body none of that is free. It has to fly to the ore, and it can only build within
about 27 tiles of wherever it currently is.

So this version plays the way a beginner actually does: fly to the nearest ore, mine by
hand until full, fly back, drop it in the core, repeat. Once it can afford one it drops a
drill on the patch it is standing on, because a drill keeps producing while the unit is
away, which is the first real lesson Mindustry teaches.

It is still deliberately unambitious. It never defends, never refines, never plans a
route. It exists to be beaten.
"""

from __future__ import annotations

from typing import Any

import numpy as np

#: Stop travelling when this close, in tiles. Overshooting wastes the whole trip.
ARRIVED = 3.0

#: Build range is about 27 tiles, so this leaves margin for the unit drifting.
BUILD_MARGIN = 20.0


class EmbodiedAlphaPolicy:
    """Fly, mine, carry, deposit. The loop a new player falls into."""

    def __init__(self, env, ore: str = "ore_copper") -> None:
        self.env = env
        self.ore = ore
        self.target: tuple[int, int] | None = None
        self.drilled: set[tuple[int, int]] = set()
        self.phase = "seek"

    def reset(self) -> None:
        self.target = None
        self.drilled.clear()
        self.phase = "seek"

    # Helpers ---------------------------------------------------------------------

    def _action(self, kind: str, x: int = 0, y: int = 0, block: int = 0, rot: int = 0):
        return np.array(
            [self.env.action_types.index(kind), block, int(x), int(y), rot],
            dtype=np.int64,
        )

    def _nearest_ore(self, info: dict[str, Any], unit: dict) -> tuple[int, int] | None:
        """Nearest tile of the ore this policy is after, not of any ore at all.

        The mineable mask covers every ore the unit can reach. Taking the nearest tile
        from it sends the unit to whatever happens to be closest, which near a core is
        usually sand or lead. It mines happily, banks happily, and the copper the task
        actually scores never moves.
        """
        mask = info["action_mask"].get("mineable")
        if mask is None or not mask.any():
            return None

        channels = self.env._bridge.channels if self.env._bridge else []
        if self.ore in channels:
            wanted = info["raw"]["spatial"][channels.index(self.ore)] > 0
            mask = mask & wanted
        if not mask.any():
            return None

        ys, xs = np.nonzero(mask)
        distance = (xs - unit["x"]) ** 2 + (ys - unit["y"]) ** 2
        best = int(np.argmin(distance))
        return int(xs[best]), int(ys[best])

    # Policy ----------------------------------------------------------------------

    def act(self, observation: dict[str, np.ndarray], info: dict[str, Any]) -> np.ndarray:
        raw = info["raw"]
        unit = raw.get("unit")
        if unit is None:
            return self._action("noop")

        carrying = int(unit.get("carrying", 0))
        capacity = int(unit.get("capacity", 30))
        core = (int(raw.get("core_x", 0)), int(raw.get("core_y", 0)))
        here = (float(unit["x"]), float(unit["y"]))

        # Bank at 80% rather than waiting for a full load. The engine starts refusing
        # items slightly before the stated capacity, so a policy that waits for
        # carrying == capacity never triggers and mines forever without ever depositing.
        if carrying >= capacity * 0.8:
            self.phase = "return"
        if self.phase == "return":
            if carrying == 0:
                self.phase = "seek"
            else:
                return self._action("unload")

        if self.phase == "seek":
            self.target = self._nearest_ore(info, unit)
            if self.target is None:
                return self._action("noop")
            self.phase = "travel"

        if self.target is None:
            return self._action("noop")

        distance = np.hypot(self.target[0] - here[0], self.target[1] - here[1])

        if self.phase == "travel":
            if distance > ARRIVED:
                return self._action("move", *self.target)
            self.phase = "mine"

        if self.phase == "mine":
            # A drill on the patch keeps producing while the unit is elsewhere, which is
            # worth far more than the copper it costs.
            if self.target not in self.drilled and distance < BUILD_MARGIN:
                affordable = info["action_mask"]["block"]
                try:
                    index = self.env.blocks.index("mechanical-drill")
                except ValueError:
                    index = -1
                if index >= 0 and affordable[index]:
                    self.drilled.add(self.target)
                    return self._action("build", *self.target, block=index)

            if carrying < capacity:
                return self._action("mine", *self.target)
            self.phase = "return"

        return self._action("noop")
