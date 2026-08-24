"""A Gymnasium environment wrapping one Mindustry instance.

The action space is factored rather than flat, as decided in
`docs/decisions/0001-full-action-space.md`: one action is a tuple of small choices
(type, block, x, y, rotation) instead of a single pick from an impossible space.

Legality is not enforced by the space. An illegal action is applied, refused by the
engine, and reported in `info["action"]`. That is deliberate: rejection is normal for an
agent still learning the rules, and masking is provided as data in `info["action_mask"]`
so any algorithm can use it without the environment silently rewriting what was chosen.
"""

from __future__ import annotations

import random
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from gamma.bridge import Bridge
from gamma.adapt import route, split
from gamma.sectors import SectorPool, build_pool
from gamma.server import ServerProcess, install_plugin
from gamma.server_setup import setup_server
from gamma.tasks import Task

#: Action types when the agent has no body: it edits the world directly.
DIRECT_ACTION_TYPES = ("noop", "place", "break")

#: Added on top of either space when the environment is given a design library.
STAMP = "stamp"

#: Action types when the agent inhabits a unit, which is what a player can do.
#: Building is queued rather than applied, and only completes once the unit is in range.
EMBODIED_ACTION_TYPES = ("noop", "move", "build", "mine", "unload", "break")

#: Kept for callers written before bodies existed.
ACTION_TYPES = DIRECT_ACTION_TYPES

#: Blocks the agent may place, in a fixed order so the index is stable across episodes.
#: A deliberately small catalogue for the early curriculum: the full list is hundreds of
#: entries, nearly all of them unbuildable at the start, and a head that wide would spend
#: most of training learning that they are unavailable.
DEFAULT_BLOCKS = (
    "conveyor",
    "junction",
    "router",
    "mechanical-drill",
    "copper-wall",
    "duo",
)

#: Global scalars exposed to the policy, in a fixed order, with the divisor that brings
#: each into roughly [0, 1].
#:
#: Normalising these is not cosmetic. Raw, the vector carries tick values in the thousands
#: and wave timers in the tens of thousands, next to item counts in the hundreds. Feeding
#: that to a network makes the value head's output explode: measured value loss above
#: 4,000 against a policy loss of 0.3, which drowns the policy gradient entirely and
#: collapses entropy to zero within five updates.
GLOBAL_FIELDS = (
    ("tick", 10_000.0),
    ("wave", 20.0),
    ("wave_time", 3_600.0),
    ("enemies", 20.0),
    ("core_health", 4_000.0),
    ("copper", 1_000.0),
    ("lead", 1_000.0),
    ("coal", 1_000.0),
    ("sand", 1_000.0),
    ("carrying", 30.0),
    ("unit_x", 256.0),
    ("unit_y", 256.0),
)

#: Fields read from the unit rather than the top level of the observation.
_UNIT_FIELDS = {"carrying": "carrying", "unit_x": "x", "unit_y": "y"}
_TOP_LEVEL = {"tick", "wave", "wave_time", "enemies", "core_health"}


class MindustryEnv(gym.Env):
    """One agent, one Mindustry server, one curriculum task."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        task: Task,
        server_dir: str | None = None,
        bridge_port: int = 7654,
        game_port: int = 6567,
        blocks: tuple[str, ...] = DEFAULT_BLOCKS,
        jar: str | None = None,
        speed: str = "max",
        embodied: bool = False,
        evaluating: bool = False,
        seed: int | None = None,
        designs: tuple = (),
    ) -> None:
        super().__init__()
        self.task = task
        #: Structures the search discovered, offered to the policy as single actions.
        #:
        #: A conveyor line from a drill to the core pays nothing until it is complete, and
        #: a policy choosing tiles one at a time never completes one: measured over 177
        #: archived episodes, 5,719 conveyors placed and one line that ever met end to end.
        #: Handed a structure as one action, the policy stops spelling and starts deciding
        #: which patch, how many, and when, which is the part worth learning.
        #:
        #: Nothing here is a human blueprint. Every design comes out of `gamma/evolve.py`,
        #: scored on what it delivered in a real game.
        self.designs = tuple(designs)
        # Designs share the block dimension of the action space rather than getting one of
        # their own. Widening the space without widening the mask would have been worse
        # than useless: the mask is what sets the size of the network's head, so the two
        # would have disagreed and the disagreement would have surfaced as a shape error
        # somewhere far from here.
        if len(self.designs) > len(blocks):
            raise ValueError(
                f"{len(self.designs)} designs will not fit in a block dimension of "
                f"{len(blocks)}: they share it"
            )
        # Drawing from the held-out half of the sector pool. Set only by the evaluator:
        # a training run that touched these would make its own score meaningless.
        self.evaluating = evaluating
        self.sector_index: int | None = None
        self.blocks = blocks
        # An embodied agent plays as a player: it must travel to what it builds and mine
        # by hand. Slower to train, and the only setting that matches the real game.
        self.embodied = embodied
        self.bridge_port = bridge_port
        self.game_port = game_port
        self.speed = speed

        self._dir = setup_server(server_dir or f"mindustry-env-{bridge_port}")
        if jar is not None:
            install_plugin(self._dir, jar)

        self._server: ServerProcess | None = None
        self._bridge: Bridge | None = None
        self._pool: SectorPool | None = None
        # Seeded per environment, so six of them running in parallel do not all draw the
        # same sector on the same episode and turn a pool of two hundred into a pool of
        # one.
        self._rng = random.Random(seed if seed is not None else bridge_port)
        self._steps = 0
        self._last_obs: dict[str, Any] = {}

        # Spaces depend on map dimensions, which are unknown until a map is loaded.
        # They are therefore built lazily, and the properties below load a map if asked
        # before the first reset. Declaring a guess in the constructor and correcting it
        # later would break any wrapper that read them in between.
        self._observation_space: spaces.Space | None = None
        self._action_space: spaces.Space | None = None

    # Lifecycle -------------------------------------------------------------------

    def _ensure_started(self) -> Bridge:
        if self._bridge is not None:
            return self._bridge

        self._server = ServerProcess(
            self._dir,
            jvm_args=[f"-Dmindustryai.port={self.bridge_port}"],
            port=self.game_port,
        )
        self._server.__enter__()
        self._server.wait_for(rf"listening on 127\.0\.0\.1:{self.bridge_port}", timeout=90)
        self._server.command(f"bridge-speed {self.speed}", r"speed set")

        # Generous on purpose: a step on a developed base, on a machine sharing eight
        # servers, can take far longer than the default. Timing out kills the
        # environment, and losing one stalls the whole run.
        self._bridge = Bridge(port=self.bridge_port, tensor=True, timeout=300.0)
        self._bridge.connect()
        return self._bridge

    def close(self) -> None:
        if self._bridge is not None:
            self._bridge.close()
            self._bridge = None
        if self._server is not None:
            self._server.__exit__()
            self._server = None

    # Spaces ----------------------------------------------------------------------

    @property
    def observation_space(self) -> spaces.Space:
        self._require_spaces()
        return self._observation_space

    @observation_space.setter
    def observation_space(self, value: spaces.Space) -> None:
        self._observation_space = value

    @property
    def action_space(self) -> spaces.Space:
        self._require_spaces()
        return self._action_space

    @action_space.setter
    def action_space(self, value: spaces.Space) -> None:
        self._action_space = value

    def _require_spaces(self) -> None:
        """Load a map if the spaces are still unknown.

        Gymnasium callers reasonably expect the spaces to exist before the first reset,
        and a policy is usually built from them. Loading a map costs a second, once.
        """
        if self._action_space is None:
            bridge = self._ensure_started()
            raw = self._load(bridge)
            self._build_spaces(raw)
            self._last_obs = raw

    def set_speed(self, speed: str) -> None:
        """Change the simulation speed of a running server.

        Uncapped speed means the engine's frame budget is zero, so its loop never sleeps.
        That is what makes a step fast, and it also means a server with nothing to do
        spins a core doing nothing at all. Twenty-four of them held the machine at 99%
        through a pause that was supposed to free it. Dropping to realtime hands the
        cores back; the pause raises it again on the way out.

        Safe to call from another thread only while the owning one is not stepping: the
        console channel is separate from the bridge socket, but neither is reentrant.
        """
        self.speed = speed
        if self._server is not None:
            self._server.command(f"bridge-speed {speed}", r"speed set", timeout=15.0)

    def _load(self, bridge: Bridge) -> dict[str, Any]:
        """Start a match: a generated sector, a named preset, or a custom map."""
        if self.task.procedural:
            raw = bridge.sector(index=self._next_sector(bridge), loadout=self.task.loadout,
                                seed=self.task.world_seed)
        elif self.task.sector is not None:
            raw = bridge.sector(self.task.sector, self.task.loadout,
                                seed=self.task.world_seed)
        else:
            raw = bridge.reset(self.task.map_name, self.task.mode,
                               seed=self.task.world_seed)
        if self.embodied:
            raw = bridge.embody()
        return raw

    def _next_sector(self, bridge: Bridge) -> int:
        """The world for this episode, drawn from the pool the task asked for.

        The listing is fetched once per environment: it describes the planet, which does
        not change, and asking for it on every episode would cost a round trip for an
        answer that is already known.
        """
        if self._pool is None:
            self._pool = build_pool(
                bridge.sectors(),
                threat_limit=self.task.threat_limit,
                worlds=self.task.worlds,
            )
        self.sector_index = self._pool.pick(self._rng, evaluating=self.evaluating)
        return self.sector_index

    def _build_spaces(self, obs: dict[str, Any]) -> None:
        spatial = obs["spatial"]
        channels, height, width = spatial.shape

        self._observation_space = spaces.Dict(
            {
                "spatial": spaces.Box(0, 255, shape=(channels, height, width), dtype=np.uint8),
                "global": spaces.Box(
                    -np.inf, np.inf, shape=(len(GLOBAL_FIELDS),), dtype=np.float32
                ),
            }
        )

        # (type, block, x, y, rotation). Sampling this uniformly is almost always an
        # illegal action, which is what info["action_mask"] is for.
        self._action_space = spaces.MultiDiscrete(
            [len(self.action_types), len(self.blocks), width, height, 4]
        )

    def _resize(self, obs: dict[str, Any]) -> None:
        """Follow the map when it changes size between episodes.

        Generated sectors are not all the same size, so this is expected rather than
        exceptional. It stays worth noticing because the spaces carry the dimensions: a
        caller reading `action_space` and caching it would then be addressing a map that
        no longer exists. Wrapped in a local window, which is how the policy sees the
        world, none of this reaches the network.
        """
        expected = self._observation_space["spatial"].shape[1:]
        if obs["spatial"].shape[1:] != expected:
            self._build_spaces(obs)

    @property
    def action_types(self) -> tuple[str, ...]:
        base = EMBODIED_ACTION_TYPES if self.embodied else DIRECT_ACTION_TYPES
        return base + (STAMP,) if self.designs else base

    # Conversion ------------------------------------------------------------------

    def _encode(self, obs: dict[str, Any]) -> dict[str, np.ndarray]:
        items = obs.get("items", {})
        unit = obs.get("unit", {})
        values = []

        for field, scale in GLOBAL_FIELDS:
            if field in _TOP_LEVEL:
                raw = float(obs.get(field, 0.0))
            elif field in _UNIT_FIELDS:
                raw = float(unit.get(_UNIT_FIELDS[field], 0))
            else:
                raw = float(items.get(field, 0))
            values.append(raw / scale)

        return {
            "spatial": obs["spatial"],
            "global": np.asarray(values, dtype=np.float32),
        }

    def _stamp(self, action: np.ndarray) -> None:
        """Lay a whole structure, one bridge action per block, before the world ticks.

        The policy chooses the design and where to put it. Everything after that is
        geometry: the drills go down first, because one needs two clear tiles by two and a
        conveyor laid on a tile it wanted makes it impossible, and the line to the core is
        recomputed for the distance it actually has to cover.

        Refusals are left alone. A structure that does not fit where it was asked for is
        an ordinary answer, and the policy should feel it as one rather than have it
        quietly corrected.
        """
        bridge = self._ensure_started()
        design = self.designs[int(action[1]) % len(self.designs)]
        anchor = (int(action[2]), int(action[3]))
        core = (int(self._last_obs.get("core_x", -1)), int(self._last_obs.get("core_y", -1)))
        if core[0] < 0:
            return

        cells = [(anchor[0] + p.dx, anchor[1] + p.dy, p.block, p.rotation)
                 for p in split(design).producers]
        cells += route(anchor, core)
        cells.sort(key=lambda cell: 0 if "drill" in cell[2] else 1)

        for x, y, block, rotation in cells:
            bridge.act({"type": "build" if self.embodied else "place",
                        "block": block, "x": x, "y": y, "rotation": rotation})

    def _decode(self, action: np.ndarray) -> dict[str, Any] | None:
        kind = self.action_types[int(action[0])]
        x, y = int(action[2]), int(action[3])

        if kind == "noop":
            return None
        if kind == STAMP:
            # Applied before the tick rather than as part of it: a structure is many
            # placements and the step protocol carries one.
            self._stamp(action)
            return None
        if kind in ("place", "build"):
            return {
                "type": kind,
                "block": self.blocks[int(action[1])],
                "x": x, "y": y, "rotation": int(action[4]),
            }
        if kind == "move":
            return {"type": "move", "x": x, "y": y}
        if kind == "mine":
            return {"type": "mine", "x": x, "y": y}
        if kind == "unload":
            return {"type": "unload"}
        if kind == "break":
            return {"type": "demolish" if self.embodied else "break", "x": x, "y": y}
        return None

    def _masks(self, obs: dict[str, Any]) -> dict[str, np.ndarray]:
        """Legality hints for the action heads.

        The block mask is exact. The position mask is an approximation: it marks tiles
        that are buildable floor with nothing on them, which is necessary but not
        sufficient, since a block larger than one tile or a build-radius rule can still
        refuse. Computing the exact mask would mean one per block type, and the engine
        already refuses illegal placements cheaply.
        """
        spatial = obs["spatial"]
        channels = self._bridge.channels if self._bridge else []

        def channel(name: str) -> np.ndarray:
            return spatial[channels.index(name)]

        affordable = set(self._bridge.affordable_blocks()) if self._bridge else set()
        block_mask = np.array([b in affordable for b in self.blocks], dtype=bool)

        free = (channel("buildable") > 0) & (channel("block") == 0) & (channel("solid") == 0)
        owned = channel("block_ally") > 0

        # The type head needs masking too, and it is the one that matters most in
        # practice: a broke agent that keeps choosing "place" spends the rest of the
        # episode being refused. Measured on a random policy, an unmasked type head
        # produced 98 refusals out of 120 once the starting copper ran out.
        if not self.embodied:
            type_mask = np.array(
                [True, bool(block_mask.any() and free.any()), bool(owned.any())],
                dtype=bool,
            )
            return {"type": type_mask, "block": block_mask, "position": free}

        unit = obs.get("unit", {})
        carrying = int(unit.get("carrying", 0))
        capacity = int(unit.get("capacity", 1))

        # Ore the unit is actually allowed to mine, by hardness. Masking this is not a
        # convenience: an agent that keeps ordering a tier-1 unit onto thorium learns
        # nothing except that mining fails.
        mineable = np.zeros_like(free)
        for name in channels:
            if name.startswith("ore_"):
                mineable |= channel(name) > 0

        # Ore under a building is not reachable, and the engine agrees: validMine requires
        # a bare tile. Without this the nearest ore to a unit standing on its core is the
        # core's own footprint, and the agent mines nothing for the whole episode while
        # every action is happily accepted.
        mineable &= channel("block") == 0
        mineable &= carrying < capacity

        type_mask = np.array(
            [
                True,                                   # noop
                True,                                   # move, always available
                bool(block_mask.any() and free.any()),  # build
                bool(mineable.any()),                   # mine
                carrying > 0,                           # unload, pointless when empty
                bool(owned.any()),                      # break
            ],
            dtype=bool,
        )

        return {
            "type": type_mask,
            "block": block_mask,
            "position": free,
            "mineable": mineable,
        }

    # Gym API ---------------------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        bridge = self._ensure_started()

        raw = self._load(bridge)
        if self._observation_space is None:
            self._build_spaces(raw)
        else:
            self._resize(raw)

        self._steps = 0
        self._last_obs = raw
        return self._encode(raw), {"action_mask": self._masks(raw), "raw": raw}

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        bridge = self._ensure_started()

        raw = bridge.step(repeat=self.task.ticks_per_step, action=self._decode(action))
        self._steps += 1

        reward = self.task.reward(self._last_obs, raw)
        won = self.task.succeeded(raw)
        lost = self.task.failed(raw)
        if won:
            reward += self.task.success_bonus

        self._last_obs = raw
        truncated = self._steps >= self.task.max_steps and not (won or lost)

        info = {
            "action_mask": self._masks(raw),
            "action": raw.get("action"),
            "raw": raw,
            "steps": self._steps,
        }
        return self._encode(raw), float(reward), bool(won or lost), bool(truncated), info
