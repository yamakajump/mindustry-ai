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

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from gamma.bridge import Bridge
from gamma.server import ServerProcess, install_plugin
from gamma.server_setup import setup_server
from gamma.tasks import Task

#: Action types, in the order the first action component indexes them.
ACTION_TYPES = ("noop", "place", "break")

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

#: Global scalars exposed to the policy, in a fixed order.
GLOBAL_FIELDS = (
    "tick",
    "wave",
    "wave_time",
    "enemies",
    "core_health",
    "copper",
    "lead",
    "coal",
    "sand",
)


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
    ) -> None:
        super().__init__()
        self.task = task
        self.blocks = blocks
        self.bridge_port = bridge_port
        self.game_port = game_port
        self.speed = speed

        self._dir = setup_server(server_dir or f"mindustry-env-{bridge_port}")
        if jar is not None:
            install_plugin(self._dir, jar)

        self._server: ServerProcess | None = None
        self._bridge: Bridge | None = None
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

        self._bridge = Bridge(port=self.bridge_port, tensor=True)
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

    def _load(self, bridge: Bridge) -> dict[str, Any]:
        """Start a match, from a campaign sector or a custom map."""
        if self.task.sector is not None:
            return bridge.sector(self.task.sector, self.task.loadout)
        return bridge.reset(self.task.map_name, self.task.mode)

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
            [len(ACTION_TYPES), len(self.blocks), width, height, 4]
        )

    # Conversion ------------------------------------------------------------------

    def _encode(self, obs: dict[str, Any]) -> dict[str, np.ndarray]:
        items = obs.get("items", {})
        values = []
        for field in GLOBAL_FIELDS:
            if field in ("tick", "wave", "wave_time", "enemies", "core_health"):
                values.append(float(obs.get(field, 0.0)))
            else:
                values.append(float(items.get(field, 0)))
        return {
            "spatial": obs["spatial"],
            "global": np.asarray(values, dtype=np.float32),
        }

    def _decode(self, action: np.ndarray) -> dict[str, Any] | None:
        kind = ACTION_TYPES[int(action[0])]
        if kind == "noop":
            return None
        if kind == "place":
            return {
                "type": "place",
                "block": self.blocks[int(action[1])],
                "x": int(action[2]),
                "y": int(action[3]),
                "rotation": int(action[4]),
            }
        return {"type": "break", "x": int(action[2]), "y": int(action[3])}

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
        type_mask = np.array(
            [
                True,                                          # noop is always legal
                bool(block_mask.any() and free.any()),         # place
                bool(owned.any()),                             # break
            ],
            dtype=bool,
        )

        return {"type": type_mask, "block": block_mask, "position": free}

    # Gym API ---------------------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        bridge = self._ensure_started()

        raw = self._load(bridge)
        if self._observation_space is None:
            self._build_spaces(raw)

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
