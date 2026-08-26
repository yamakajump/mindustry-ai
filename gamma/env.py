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
from gamma import mining, tasks
from gamma.adapt import split
from gamma.sectors import SectorPool, build_pool
from gamma.server import ServerProcess, install_plugin
from gamma.server_setup import setup_server
from gamma.tasks import Task

#: Action types when the agent has no body: it edits the world directly.
DIRECT_ACTION_TYPES = ("noop", "place", "break")

#: Added on top of either space when the environment is given a design library.
STAMP = "stamp"

#: Put drills on the ore that is actually there, and run a belt home.
#:
#: The action the project needed and did not have. A conveyor line placed one tile at a
#: time is never finished: across 177 archived episodes the agent laid 5,719 conveyors and
#: completed one line. A fixed structure finishes, and lands on the wrong ground: the one
#: design it had put 21,330 drills down over 25 episodes with 89.9% of them on bare rock,
#: because generated ore is a blob of arbitrary shape and a pattern bred on one map is a
#: pattern for that map.
#:
#: So the geometry is computed from what is under the anchor, and only the geometry.
#: Measured over a thousand random anchors on real maps, the packer puts 100% of its
#: drills on ore and fills 3.95 of each drill's four tiles.
#:
#: What stays with the policy is every decision worth calling one: which patch, when, how
#: much of the episode to spend on economy before defence. What leaves it is spelling a
#: line out tile by tile, which is a motor skill rather than a choice, and which it cannot
#: do. Fixed designs remain, for factories, where a layout owes nothing to the terrain.
CONNECT = "connect"

#: The distinct sets of tiles an action can aim at, in the order they are sent.
#:
#: One position head serves every action type and cannot know which was chosen, so it was
#: given one mask for all of them: `free`, meaning buildable, empty and not solid. Right
#: for building and exactly inverted for breaking, since `free` and `owned` are disjoint by
#: construction. Measured over 30 archived episodes, 6,660 demolitions of which 23 hit a
#: building the agent had placed, while `break` was 30% of everything it did.
#:
#: Naming the sets lets the network pick the right one once it knows the type, which is why
#: the type is now sampled first. The union that came before removed the impossibility and
#: left the ambiguity: the agent could aim at a building when breaking, and nothing made it.
POSITION_SETS = ("free", "owned", "mineable", "any")

#: Which set each action type aims into. Anything unnamed falls back to `any`, which is
#: honest for an action whose target is not a tile at all.
POSITION_OF_TYPE = {
    "place": "free", "build": "free", "stamp": "free",
    "break": "owned",
    "mine": "mineable", "connect": "mineable",
    "move": "any", "noop": "any", "unload": "any",
}

#: Types for which the block and rotation choices mean anything.
#:
#: A factored action carries all four every step, so on a `move` the policy still picks a
#: block and a rotation that go nowhere. Their log probability enters the importance ratio
#: regardless, which is noise, and the entropy bonus keeps pushing them towards uniform
#: because no gradient ever tells them otherwise: measured over 1.5M steps, the rotation
#: head sat at 1.38 against a maximum of 1.386 and never moved. Masked to a single option
#: they contribute exactly zero to both.
BLOCK_MATTERS = frozenset({"place", "build", "stamp"})
ROTATION_MATTERS = frozenset({"place", "build"})

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
    # Where home is, which nothing in the observation used to say.
    #
    # The agent knew its own position and its core's health and never the core's position,
    # so it could not tell how far from home it was nor which way home lay. The window is
    # forty-eight tiles and follows the unit, so once it wandered the core was simply not
    # in the picture. It went on building defences where it happened to be standing:
    # measured over 174 lost cores, 9.4 turrets and 16.8 walls per episode, laid a median
    # of 60 tiles from the core and 105 on average. The defence existed and protected
    # nothing, and the agent had no way of knowing.
    ("core_dx", 256.0),
    ("core_dy", 256.0),
)

#: Fields read from the unit rather than the top level of the observation.
_UNIT_FIELDS = {"carrying": "carrying", "unit_x": "x", "unit_y": "y"}
_TOP_LEVEL = {"tick", "wave", "wave_time", "enemies", "core_health"}
#: Offsets from the unit to the core, which no single field holds.
_DERIVED = {"core_dx": ("core_x", "x"), "core_dy": ("core_y", "y")}


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
        window: int = 0,
        capture_scene: bool = False,
        mining: bool = True,
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
        #: Nothing here is a human blueprint. Every design comes out of the forge, scored
        #: on what it delivered in a real game.
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

        #: Offer the computed extraction macro. See `CONNECT`.
        self.mining = bool(mining)
        self.bridge_port = bridge_port
        self.game_port = game_port
        self.speed = speed

        #: Ask the bridge to send only the window the policy reads, not the whole map.
        #:
        #: Verified identical to cropping after the wire, byte for byte, on the same world
        #: and the same origin, and nine times faster: a step costs 1.0 ms against 9.6. The
        #: map is 14 by 432 by 432 and the window is 14 by 48 by 48, so eighty times more
        #: was being encoded, sent and decoded than anybody ever looked at.
        self.window = int(window)

        #: Fetch what moved on every step, for whoever wants to see it.
        #:
        #: One caller, deliberately. `scene()` returns everything that changed *since the
        #: last call*, so two consumers asking independently each receive half of what
        #: happened and neither can tell. The environment asks once and hands the answer
        #: to both the recorder and the dashboard through `info`.
        #:
        #: Measured cost: 3% on the step, and 741 bytes a step, which is about two
        #: megabytes an episode before compression.
        self.capture_scene = bool(capture_scene)

        #: Set by `_stamp` and `_connect` for the step they happened on, so a replay can
        #: show what was laid rather than a single unexplained action.
        self._last_stamp: dict[str, Any] | None = None
        self._last_connect: dict[str, Any] | None = None

        #: When each tile was last built on, so tearing down fresh work can be told from
        #: revising old work. See `CHURN_WINDOW`.
        self._built_at: dict[tuple[int, int], int] = {}
        self._breaking: tuple[int, int] = (0, 0)
        self._placing: tuple[int, int] | None = None
        self._churned = False

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
            jvm_args=[
                f"-Dmindustryai.port={self.bridge_port}",
                # Two dozen servers share one machine, and by default each one sizes its
                # thread pools from `availableProcessors()`, which reports every logical
                # core on the box. Measured on a live run: 29 JVMs holding 1,897 threads
                # between them, 65 apiece, on 32 logical processors. Nearly all of that is
                # GC and JIT workers for a machine each server believes it owns alone, and
                # they contend against the one thread that actually simulates the world.
                "-XX:ActiveProcessorCount=2",
                # Serial collection follows from the same fact. Parallel GC exists to
                # spend several cores to shorten a pause; here there are no spare cores to
                # spend, and the pause is one server's alone.
                "-XX:+UseSerialGC",
                # A default heap is a quarter of physical memory, so each of these was
                # entitled to 23 GB and none needs it. Bounding it keeps the live set
                # small enough to stay in cache instead of sprawling across 93 GB.
                "-Xms256m",
                "-Xmx1g",
            ],
            port=self.game_port,
        )
        self._server.__enter__()
        self._server.wait_for_bridge(self.bridge_port, timeout=90)
        self._server.command(f"bridge-speed {self.speed}", r"speed set")

        # Generous on purpose: a step on a developed base, on a machine sharing eight
        # servers, can take far longer than the default. Timing out kills the
        # environment, and losing one stalls the whole run.
        self._bridge = Bridge(port=self.bridge_port, tensor=True, timeout=300.0,
                              window=self.window)
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
                                seed=self.task.world_seed, ore_near=self.task.ore_near)
        elif self.task.sector is not None:
            raw = bridge.sector(self.task.sector, self.task.loadout,
                                seed=self.task.world_seed, ore_near=self.task.ore_near)
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
    def position_set_of_type(self) -> tuple[int, ...]:
        """For each action type, the index into `POSITION_SETS` it aims into."""
        return tuple(POSITION_SETS.index(POSITION_OF_TYPE.get(name, "any"))
                     for name in self.action_types)

    @property
    def block_of_type(self) -> tuple[bool, ...]:
        return tuple(name in BLOCK_MATTERS for name in self.action_types)

    @property
    def rotation_of_type(self) -> tuple[bool, ...]:
        return tuple(name in ROTATION_MATTERS for name in self.action_types)

    @property
    def action_types(self) -> tuple[str, ...]:
        base = EMBODIED_ACTION_TYPES if self.embodied else DIRECT_ACTION_TYPES
        base = base + (CONNECT,) if self.mining else base
        return base + (STAMP,) if self.designs else base

    # Conversion ------------------------------------------------------------------

    def _encode(self, obs: dict[str, Any]) -> dict[str, np.ndarray]:
        items = obs.get("items", {})
        unit = obs.get("unit", {})
        values = []

        for field, scale in GLOBAL_FIELDS:
            if field in _TOP_LEVEL:
                raw = float(obs.get(field, 0.0))
            elif field in _DERIVED:
                where, mine = _DERIVED[field]
                raw = float(obs.get(where, 0.0)) - float(unit.get(mine, 0))
            elif field in _UNIT_FIELDS:
                raw = float(unit.get(_UNIT_FIELDS[field], 0))
            else:
                raw = float(items.get(field, 0))
            values.append(raw / scale)

        return {
            "spatial": obs["spatial"],
            "global": np.asarray(values, dtype=np.float32),
        }

    #: How long a building has to stand before pulling it down counts as revision.
    #:
    #: Demolishing is a tool: a belt that dead-ends, a line that wants rerouting, a drill in
    #: the way of a better one. It is also the cheapest way to undo your own work, and once
    #: the position mask let `break` actually reach a building the agent did little else.
    #: Measured over 72 episodes of a live run: 2,996 demolitions against 2,673
    #: placements, 80% of them on its own buildings, 812 place-then-break cycles on the
    #: same tile and one tile touched eleven times.
    #:
    #: Pricing every demolition higher would stop that and stop revision with it, which is
    #: worse: an agent that cannot afford to change its mind builds once and lives with it.
    #: So the price falls on the pathology instead. Two hundred steps is a hundred seconds
    #: of game time, long enough for a drill to have delivered something if it was ever
    #: going to, and a structure older than that is being reconsidered rather than churned.
    CHURN_WINDOW = 200

    #: The most a single connect may commit to, in blocks.
    #:
    #: A structure the agent cannot pay for and its one unit cannot reach is not a
    #: structure, it is a queue. Measured on the fixed design, which routed to the core
    #: without asking either question: a quarter of its stamps committed to more than a
    #: hundred blocks, the ninety-ninth percentile to 405, and the worst to 561, on an
    #: agent starting with three hundred copper.
    CONNECT_BUDGET = 60

    def _connect(self, x: int, y: int) -> None:
        """Drills on the ore under the anchor, and a belt from them to the core.

        Nothing here is remembered from another map. The drills are packed onto the shape
        of the patch that is actually there, and the belt is the shortest way round what
        is actually in between. What the policy chose is the patch and the moment.
        """
        bridge = self._ensure_started()
        raw = self._last_obs
        core = (int(raw.get("core_x", -1)), int(raw.get("core_y", -1)))
        if core[0] < 0 or "spatial" not in raw:
            self._last_connect = {"applied": False, "type": CONNECT, "reason": "no world yet"}
            return

        spatial = raw["spatial"]
        channels = bridge.channels
        ore = np.zeros(spatial.shape[1:], dtype=np.int16)
        for index, name in enumerate(channels):
            if name.startswith("ore_"):
                ore[spatial[index] > 0] = index

        # The anchor arrives in world coordinates and the tensor is a window, so the two
        # have to be reconciled before anything is indexed. Reading world coordinates into
        # a window-shaped array is not an error that raises: it lands out of range, the
        # slice comes back empty, and every connect reports "no ore" while the map is
        # covered in it. Measured before this line existed: 182 refusals and not one
        # drill placed.
        origin = raw.get("window_origin") or (0, 0)
        ox, oy = int(origin[0]), int(origin[1])
        rows, columns = ore.shape

        half = 12
        x0 = max(0, min(x - ox - half, columns - 1))
        y0 = max(0, min(y - oy - half, rows - 1))
        patch = ore[y0:min(rows, y0 + 2 * half), x0:min(columns, x0 + 2 * half)]

        # Ranked by what the reward pays for each ore, so the packer and the scoring agree
        # on what is worth mining. Without it the packer drills sand, which every darksand
        # tile yields and which generated maps are made of: 252 of 288 tiles under the
        # drills on a first run, against 32 of titanium worth twenty times as much.
        worth = {index: tasks.ORE_VALUE.get(name[4:], 1.0)
                 for index, name in enumerate(channels) if name.startswith("ore_")}

        drills = mining.pack(patch, size=2, minimum=2, limit=6, worth=worth)
        if not drills:
            self._last_connect = {"applied": False, "type": CONNECT, "reason": "no ore"}
            return

        # Routed in world coordinates, on ground that only the window describes. Outside it
        # nothing is known, so nothing is claimed: unknown ground is treated as open, which
        # is what the old L-shaped routing assumed everywhere.
        ground = self._ground()
        if ground is None:
            self._last_connect = {"applied": False, "type": CONNECT, "reason": "no world yet"}
            return
        known, ours = ground

        # Back to world coordinates, which is what the bridge builds in.
        cells: list[tuple[int, int, str, int]] = []
        for dx, dy, _, _ in drills:
            cells.append((ox + x0 + dx, oy + y0 + dy, "mechanical-drill", 0))
        # A drill needs its whole square, so the line cannot be routed through one it is
        # about to lay.
        for cx, cy, _, _ in cells:
            known[cy:cy + 2, cx:cx + 2] = False

        # From the drill nearest the core, because that is the shortest line and the line
        # is what costs.
        head = min(cells, key=lambda cell: abs(cell[0] - core[0]) + abs(cell[1] - core[1]))

        steps = mining.path(known, (head[0], head[1] - 1), core)
        if steps is None:
            self._last_connect = {"applied": False, "type": CONNECT, "reason": "no route"}
            return

        # Nothing is laid where something of the agent's already stands: the line runs into
        # its own network rather than over it, which is what a player does and what keeps a
        # working belt from being overwritten with the wrong rotation.
        cells += [(p.x, p.y, p.block, p.rotation) for p in mining.belt(steps)
                  if not ours[p.y, p.x]]
        if len(cells) > self.CONNECT_BUDGET:
            self._last_connect = {
                "applied": False, "type": CONNECT, "reason": "too far",
                "asked": len(cells), "budget": self.CONNECT_BUDGET,
            }
            return

        laid = 0
        refusal = ""
        for cx, cy, block_name, rotation in cells:
            outcome = bridge.act({"type": "build" if self.embodied else "place",
                                  "block": block_name, "x": cx, "y": cy,
                                  "rotation": rotation})
            answer = outcome.get("action") or {}
            laid += bool(answer.get("applied"))
            # Kept from the first block the world turned down, because a connect that lays
            # nothing says nothing about why on its own: 2,357 of them were recorded as
            # refused with no reason at all, and the answer turned out to be that the
            # agent was broke.
            if not refusal and not answer.get("applied"):
                refusal = str(answer.get("reason", ""))

        # What the packer believed it was doing, kept beside what was asked for. The two
        # disagreeing is a coordinate bug, and a coordinate bug here is invisible: the
        # drills go down, the belt goes down, and they mine nothing.
        believed = sum(tiles for _, _, _, tiles in drills)

        self._last_connect = {
            "applied": laid > 0, "type": CONNECT, "x": x, "y": y,
            "origin": [ox, oy], "patch": [x0, y0],
            "believed_ore": believed,
            "drills": len(drills), "laid": laid, "asked": len(cells),
            "reason": "" if laid else (refusal or "nothing accepted"),
            "cells": [[cx, cy, block_name, rotation]
                      for cx, cy, block_name, rotation in cells],
        }
        self._remember(cells)

    def _route_to_core(self, start: tuple[int, int],
                       core: tuple[int, int]) -> list[tuple[int, int]] | None:
        """The shortest way home around whatever is in between.

        Passability is only known inside the window, so beyond it the ground is treated as
        open. That is exactly what the L-shaped routing assumed everywhere, and it is a
        much smaller assumption when it only covers the part nobody can see.
        """
        ground = self._ground()
        if ground is None:
            return None
        passable, _ = ground
        return mining.path(passable, start, core)

    def _ground(self) -> tuple[np.ndarray, np.ndarray] | None:
        """What a belt can cross, and which of it is already the agent's, map-wide.

        Its own buildings do not block it, and that is the whole correction. Everything
        built counted as a wall, so the first line to reach the core sealed the core: every
        tile touching it was then one of the agent's own conveyors, and no second line
        could ever arrive. The better the economy got, the more thoroughly it walled itself
        in. Measured over 179 episodes after the search itself was fixed, "no route" was
        still the leading refusal in the run, 15,181 connects and 9,523 stamps, and it was
        never the terrain.

        A player does not route around their own belt, they run into it. So allied ground
        is crossed, and the caller lays nothing on the tiles that are already occupied:
        the ore rides what is already there.
        """
        raw = self._last_obs
        spatial = raw.get("spatial")
        width = int(raw.get("map_width", 0))
        height = int(raw.get("map_height", 0))
        if spatial is None or width <= 0 or height <= 0:
            return None

        channels = self._bridge.channels if self._bridge else []
        passable = np.ones((height, width), dtype=bool)
        ours = np.zeros((height, width), dtype=bool)
        if "solid" in channels and "block" in channels:
            solid = spatial[channels.index("solid")] > 0
            block = spatial[channels.index("block")] > 0
            mine = (spatial[channels.index("block_ally")] > 0
                    if "block_ally" in channels else np.zeros_like(block))
            origin = raw.get("window_origin") or (0, 0)
            ox, oy = int(origin[0]), int(origin[1])
            rows, columns = block.shape
            # Allied ground is crossed whatever it is, and the core is the reason it has to
            # be. A core is three tiles by three and every one of them is solid, so the
            # only ways into the centre are the eight tiles around it, which are the core
            # as well. The search could reach the goal tile by exception and could never
            # reach a neighbour of it, which made arriving at its own core impossible by
            # construction the moment the core was inside the window. Outside the window
            # everything is assumed open, so the same connect succeeded or failed on where
            # the agent happened to be standing.
            passable[oy:oy + rows, ox:ox + columns] = ~((solid | block) & ~mine)
            ours[oy:oy + rows, ox:ox + columns] = (block | solid) & mine

        return passable, ours

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

        # Routed around what is in the way, and only if the way is short enough to pay for.
        #
        # It used to draw an L without looking and place however many conveyors that came
        # to. Measured over 34,341 stamps from a real run: a median of 43 blocks, but 23.5%
        # over a hundred, a ninety-ninth percentile of 405 and a worst of 561, on an agent
        # that starts with three hundred copper and one unit to build with. A structure it
        # can neither pay for nor reach is not a structure, it is a queue.
        ground = self._ground()
        steps = mining.path(ground[0], anchor, core) if ground else None
        if steps is None:
            self._last_stamp = {"applied": False, "type": STAMP, "reason": "no route"}
            return

        # As for a connect: the line runs into what is already the agent's rather than
        # over it, so a working belt is never overwritten with the wrong rotation.
        ours = ground[1]
        cells += [(p.x, p.y, p.block, p.rotation) for p in mining.belt(steps)
                  if not ours[p.y, p.x]]
        if len(cells) > self.CONNECT_BUDGET:
            self._last_stamp = {
                "applied": False, "type": STAMP, "reason": "too far",
                "asked": len(cells), "budget": self.CONNECT_BUDGET,
            }
            return

        cells.sort(key=lambda cell: 0 if "drill" in cell[2] else 1)

        laid = 0
        refusal = ""
        for x, y, block, rotation in cells:
            outcome = bridge.act({"type": "build" if self.embodied else "place",
                                  "block": block, "x": x, "y": y, "rotation": rotation})
            answer = outcome.get("action") or {}
            laid += bool(answer.get("applied"))
            # Why the first block was turned down, as for a connect. A stamp that lays
            # nothing was recorded as refused with no reason at all 14,384 times.
            if not refusal and not answer.get("applied"):
                refusal = str(answer.get("reason", ""))

        # Reported, because a stamp applies outside the step protocol and therefore
        # produced no outcome at all. The recorder only writes an action it can see
        # applied, so stamps were invisible in every replay: after the type head was
        # widened to reach them, the archives still showed zero, and the only way to tell
        # "never chosen" from "never recorded" was to read the shape of the network.
        self._last_stamp = {
            "applied": laid > 0, "type": STAMP,
            "design": int(action[1]) % len(self.designs),
            "x": anchor[0], "y": anchor[1], "laid": laid, "asked": len(cells),
            "reason": "" if laid else (refusal or "nothing accepted"),
            # The cells themselves, because a viewer derives what stands from the actions
            # it was given and a stamp is fifty placements reported as one. Without them a
            # replay of a stamping agent shows an empty map while it builds.
            "cells": [[x, y, block, rotation] for x, y, block, rotation in cells],
        }
        self._remember(cells)

    def _remember(self, cells) -> None:
        """Note every tile an action laid, whatever laid it.

        Only manual placements were noted at first, and the penalty for undoing fresh work
        never fired once: measured over 30 episodes, not one demolition landed on a tile the
        agent had `place`d. It had stopped placing by hand. Nearly everything it built came
        from `connect`, forty blocks at a time, and those were the buildings it was tearing
        down. A rule that watches the wrong door is indistinguishable from no rule.
        """
        for cell in cells:
            self._built_at[(int(cell[0]), int(cell[1]))] = self._steps

    def _note_churn(self, kind: str, x: int, y: int) -> bool:
        """Remember what was built where, and report tearing down something still fresh.

        The distinction the reward cannot make on its own: it sees a count of buildings
        deconstructed and nothing about which, so pulling down a drill laid four steps ago
        and reworking a line that has been running for a minute look identical to it.

        Judged on the intent here and settled once the world has answered, because a
        refused demolition tears nothing down. Charging it would price an attempt that
        changed nothing, and forgetting the tile on the way would let the demolition that
        does land a moment later go free.
        """
        self._placing = (x, y) if kind in ("place", "build") else None
        if self._placing is not None:
            self._built_at[(x, y)] = self._steps
            return False
        if kind != "break":
            return False

        built = self._built_at.get((x, y))
        return built is not None and self._steps - built < self.CHURN_WINDOW

    def _settle_churn(self, raw: dict[str, Any]) -> None:
        """Keep only what the world accepted, in both directions.

        A refused demolition tears nothing down, so charging it would price an attempt
        that changed nothing and forgetting the tile on the way would let the demolition
        that does land a moment later go free. A refused placement builds nothing either,
        and leaving it on the ledger would charge the agent for later clearing a building
        it never owned.
        """
        applied = bool((raw.get("action") or {}).get("applied", False))
        if self._placing is not None:
            if not applied:
                self._built_at.pop(self._placing, None)
            self._placing = None
            return
        if not self._churned:
            return
        if not applied:
            self._churned = False
            return
        self._built_at.pop(self._breaking, None)

    def _decode(self, action: np.ndarray) -> dict[str, Any] | None:
        kind = self.action_types[int(action[0])]
        x, y = int(action[2]), int(action[3])
        self._churned = self._note_churn(kind, x, y)
        self._breaking = (x, y)

        if kind == "noop":
            return None
        if kind == CONNECT:
            self._connect(x, y)
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

    def _affordable_design(self, affordable: set[str]) -> bool:
        """Whether any loaded design could be paid for as it stands."""
        for design in self.designs:
            parts = split(design).producers
            if parts and all(part.block in affordable for part in parts):
                return True
        return False

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
            legal = [True, bool(block_mask.any() and free.any()), bool(owned.any())]
            # Gated on paying for the line and for the design, exactly as in the embodied
            # branch. Two copies of one rule is how the direct mode came to offer actions
            # the other had already learned to withhold.
            affords_a_line = ("mechanical-drill" in affordable) and ("conveyor" in affordable)
            if self.mining:
                legal.append(bool(free.any() and affords_a_line))
            if self.designs:
                legal.append(bool(free.any() and self._affordable_design(affordable)))
            return {
                "type": np.array(legal, dtype=bool),
                "block": block_mask,
                # The union, kept for callers that take one mask for every type. The
                # network uses the named sets below and never sees this.
                "position": free | owned,
                "position_sets": np.stack([free, owned, np.zeros_like(free), free | owned]),
            }

        unit = obs.get("unit", {})
        carrying = int(unit.get("carrying", 0))
        capacity = int(unit.get("capacity", 1))

        # Ore the unit is actually allowed to mine, by hardness. Masking this is not a
        # convenience: an agent that keeps ordering a tier-1 unit onto thorium learns
        # nothing except that mining fails.
        #
        # This comment described the intent and the code below it took every ore channel
        # there was, which is worse than no comment: it read as done. Measured over 184
        # episodes, 9,584 mining orders were refused, the largest single source of refused
        # actions in the run once routing was fixed, and the whole cause was titanium and
        # thorium being offered to a unit that cannot touch them.
        tier = int((obs.get("unit") or {}).get("mine_tier", 99))
        hardness = list(getattr(self._bridge, "ore_hardness", []) or [])
        mineable = np.zeros_like(free)
        ore_index = 0
        for name in channels:
            if not name.startswith("ore_"):
                continue
            reachable = hardness[ore_index] <= tier if ore_index < len(hardness) else True
            if reachable:
                mineable |= channel(name) > 0
            ore_index += 1

        # Ore under a building is not reachable, and the engine agrees: validMine requires
        # a bare tile. Without this the nearest ore to a unit standing on its core is the
        # core's own footprint, and the agent mines nothing for the whole episode while
        # every action is happily accepted.
        mineable &= channel("block") == 0
        mineable &= carrying < capacity

        legal = [
            True,                                   # noop
            True,                                   # move, always available
            bool(block_mask.any() and free.any()),  # build
            bool(mineable.any()),                   # mine
            carrying > 0,                           # unload, pointless when empty
            bool(owned.any()),                      # break
        ]

        # The stamp, which was missing, and its absence was not a detail.
        #
        # `action_types` grows a seventh entry as soon as a design is loaded, but this
        # mask stopped at six, and the network takes its type count from the mask. So the
        # head had six outputs and index six was unreachable: measured over 54 recent
        # episodes, zero stamps in 39,456 actions. Not a choice the policy made, an option
        # it never had.
        #
        # That matters more than a missing action. Laying a structure in one decision is
        # the whole answer to the problem this project keeps hitting, which is that a
        # conveyor line placed one tile at a time never gets finished: 5,719 conveyors
        # across 177 archived episodes and one line that ever met end to end. The
        # mechanism built for it, the designs and the forge that breeds them, has been
        # switched off the entire time while every run passed --designs.
        # Offered when there is ore in view worth a drill, somewhere to put the belt, and
        # the copper to pay for both. Cheaper to check here than to let the agent choose it
        # and be refused: a refusal it cannot see coming teaches it nothing except that the
        # action is bad.
        #
        # Affordability was the part missing, and it was most of the episode. `connect`
        # asks the world for up to sixty blocks at once and reports failure only when not
        # one of them is accepted, so a broke agent standing on ore was offered it every
        # step, asked for sixty drills and conveyors it could not pay for, and had all
        # sixty refused. Measured over eleven episodes: 2,357 refusals out of 2,671
        # connects, 88% of them, and connect refusals alone were 80% of every refused
        # action in the run.
        affords_a_line = ("mechanical-drill" in affordable) and ("conveyor" in affordable)
        if self.mining:
            legal.append(bool(mineable.any() and free.any() and affords_a_line))

        # Gated on the blocks a design actually needs, not on affording any block at all.
        # A design is a fixed list of parts, so affording a conveyor says nothing about
        # affording the press the structure is built around, and the agent was offered the
        # stamp on the strength of the cheapest thing in the game.
        if self.designs:
            legal.append(bool(free.any() and self._affordable_design(affordable)))

        return {
            "type": np.array(legal, dtype=bool),
            "block": block_mask,
            # Everything any legal action could aim at, because there is one position head
            # for four action types and it cannot know which one was chosen.
            #
            # It used to be `free` alone, which is buildable, empty and not solid. That is
            # right for building and exactly inverted for breaking: `free` and `owned` are
            # disjoint by construction, so every tile the agent was allowed to aim at when
            # it chose `break` was guaranteed to have nothing on it. Measured over 30
            # archived episodes: 6,660 demolitions, of which 23 hit a building the agent
            # had placed. The other 99.7% hit natural walls or bare ground.
            #
            # The union is not the right answer either, only a much better wrong one: the
            # honest fix is a position mask conditioned on the type, which means sampling
            # the type first and is a change to the shape of the policy rather than to a
            # mask. This removes the impossibility; it does not remove the ambiguity.
            "position": free | owned | mineable,
            "mineable": mineable,
            # One plane per entry of POSITION_SETS, so the network can pick the right one
            # once it knows which action it is taking. `any` is everything a legal action
            # could target rather than literally every tile: a policy aiming a `move` at a
            # wall learns nothing except that walls refuse it.
            "position_sets": np.stack([free, owned, mineable, free | owned | mineable]),
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
        self._built_at.clear()

        # A reward may keep a ledger across a step, and a ledger that survives a reset
        # would carry one episode's delivery into the next one's milestones.
        if hasattr(self.task.reward, "reset"):
            self.task.reward.reset()

        return self._encode(raw), {"action_mask": self._masks(raw), "raw": raw}

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        bridge = self._ensure_started()

        self._last_stamp = None
        self._last_connect = None
        self._churned = False
        self._placing = None
        raw = bridge.step(repeat=self.task.ticks_per_step, action=self._decode(action))
        self._steps += 1
        if self._last_stamp is not None:
            # A stamp leaves no outcome in `raw`, having been applied before the tick.
            raw = {**raw, "action": self._last_stamp}
        elif self._last_connect is not None:
            raw = {**raw, "action": self._last_connect}

        # Stated to the reward, which sees only a count of buildings deconstructed and so
        # cannot tell undoing fresh work from reworking an old line.
        self._settle_churn(raw)
        raw = {**raw, "churn": 1 if self._churned else 0}
        reward = self.task.reward(self._last_obs, raw)
        won = self.task.succeeded(raw)
        lost = self.task.failed(raw)
        if won:
            reward += self.task.success_bonus
            # Folded into the itemisation too, or the accounts stop adding up exactly
            # where the run finally went well. Seven steps in one archive paid 51 while
            # their breakdown showed 1, which reads as a leak and is a bonus.
            terms = getattr(self.task.reward, "last_terms", None)
            if terms is not None:
                terms["won"] = terms.get("won", 0.0) + self.task.success_bonus

        self._last_obs = raw
        truncated = self._steps >= self.task.max_steps and not (won or lost)

        info = {
            "action_mask": self._masks(raw),
            "action": raw.get("action"),
            "raw": raw,
            "steps": self._steps,
        }
        if self.capture_scene:
            try:
                info["scene"] = bridge.scene()
            except Exception:
                # A dropped frame is a gap in a recording, not a reason to lose the
                # environment it came from. The next delta resynchronises anyway.
                info["scene"] = None
        return self._encode(raw), float(reward), bool(won or lost), bool(truncated), info
