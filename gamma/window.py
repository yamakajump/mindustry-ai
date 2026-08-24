"""A local view of the world, centred on the agent.

The full map is 256x256 tiles. Feeding that to a convolutional network costs 28 times
more compute than a 48x48 window, and the position head would have to choose among 65,536
tiles instead of 2,304.

Cropping is not only cheaper, it is closer to how the game is actually played. A player
acts near themselves: they build within reach and mine what they are standing over. An
agent that can name any tile on the map is choosing mostly among tiles it cannot affect.

The window follows the agent's unit when it has one, and the core otherwise. Actions are
expressed in window coordinates and translated back to map coordinates on the way out, so
the policy never has to know where the window currently is.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces

#: Window side in tiles. Alpha's build range is about 27 tiles, so 48 covers everything
#: it can reach plus context on what lies just beyond.
DEFAULT_SIZE = 48


class LocalWindow:
    """Wraps an environment so observations and actions are local to the agent."""

    def __init__(self, env, size: int = DEFAULT_SIZE) -> None:
        self.env = env
        self.size = size
        self._origin = (0, 0)
        self._last_raw: dict[str, Any] = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    # Spaces ----------------------------------------------------------------------

    @property
    def observation_space(self) -> spaces.Space:
        inner = self.env.observation_space
        channels = inner["spatial"].shape[0]
        return spaces.Dict({
            "spatial": spaces.Box(0, 255, shape=(channels, self.size, self.size), dtype=np.uint8),
            "global": inner["global"],
        })

    @property
    def action_space(self) -> spaces.Space:
        nvec = self.env.action_space.nvec.copy()
        nvec[2] = self.size
        nvec[3] = self.size
        return spaces.MultiDiscrete(nvec)

    # Windowing -------------------------------------------------------------------

    def _centre(self, raw: dict[str, Any]) -> tuple[int, int]:
        unit = raw.get("unit")
        if unit:
            return int(unit.get("x", 0)), int(unit.get("y", 0))
        return int(raw.get("core_x", 0)), int(raw.get("core_y", 0))

    def _update_origin(self, raw: dict[str, Any]) -> None:
        centre_x, centre_y = self._centre(raw)
        width = int(raw.get("map_width", self.size))
        height = int(raw.get("map_height", self.size))
        half = self.size // 2
        # Clamped so the window never runs off the map, which would need padding and would
        # make the same tile appear at different offsets depending on where the agent is.
        self._origin = (
            max(0, min(centre_x - half, width - self.size)),
            max(0, min(centre_y - half, height - self.size)),
        )

    def _crop(self, observation: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        x0, y0 = self._origin
        spatial = observation["spatial"][:, y0:y0 + self.size, x0:x0 + self.size]

        if spatial.shape[1:] != (self.size, self.size):
            padded = np.zeros((spatial.shape[0], self.size, self.size), dtype=np.uint8)
            padded[:, :spatial.shape[1], :spatial.shape[2]] = spatial
            spatial = padded

        return {"spatial": np.ascontiguousarray(spatial), "global": observation["global"]}

    def _crop_mask(self, mask: np.ndarray) -> np.ndarray:
        x0, y0 = self._origin
        window = mask[y0:y0 + self.size, x0:x0 + self.size]
        if window.shape != (self.size, self.size):
            padded = np.zeros((self.size, self.size), dtype=bool)
            padded[:window.shape[0], :window.shape[1]] = window
            window = padded
        return window

    def _crop_info(self, info: dict[str, Any]) -> dict[str, Any]:
        masks = info.get("action_mask", {})
        local = dict(masks)
        for key in ("position", "mineable"):
            if key in masks and masks[key].ndim == 2:
                local[key] = self._crop_mask(masks[key])

        # A type is only legal if something in the window supports it. Otherwise the agent
        # picks "build", finds no legal tile nearby, and is refused for reasons it cannot
        # see from its own observation.
        if "type" in local:
            types = local["type"].copy()
            for index, name in enumerate(self.env.action_types):
                if name in ("place", "build") and "position" in local:
                    types[index] = types[index] and bool(local["position"].any())
                if name == "mine" and "mineable" in local:
                    types[index] = types[index] and bool(local["mineable"].any())
            local["type"] = types

        out = dict(info)
        out["action_mask"] = local
        out["window_origin"] = self._origin
        return out

    # Gym API ---------------------------------------------------------------------

    def reset(self, **kwargs: Any):
        observation, info = self.env.reset(**kwargs)
        self._last_raw = info["raw"]
        self._update_origin(self._last_raw)
        return self._crop(observation), self._crop_info(info)

    def step(self, action: np.ndarray):
        absolute = np.array(action, dtype=np.int64, copy=True)
        x0, y0 = self._origin
        absolute[2] = min(int(action[2]) + x0, max(0, int(self._last_raw.get("map_width", 1)) - 1))
        absolute[3] = min(int(action[3]) + y0, max(0, int(self._last_raw.get("map_height", 1)) - 1))

        observation, reward, terminated, truncated, info = self.env.step(absolute)
        self._last_raw = info["raw"]
        self._update_origin(self._last_raw)
        return self._crop(observation), reward, terminated, truncated, self._crop_info(info)

    def close(self) -> None:
        self.env.close()
