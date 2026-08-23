"""Recording matches so they can be watched afterwards.

One format, two transports, as decided in `docs/decisions/0003-one-replay-format.md`.
A header describes the map once, then every step appends only what changed.

The size argument is decisive: storing the raw observation tensor each step would cost
roughly 413 MB per episode. Block deltas plus a compressed terrain header cost about
200 KB, which is small enough to commit a showcase replay to the repository and serve it
from a static page.
"""

from __future__ import annotations

import base64
import gzip
import json
import zlib
from pathlib import Path
from typing import Any

import numpy as np

#: Bumped when the on-disk shape changes in a way a viewer must know about.
REPLAY_FORMAT = 1


def _encode_plane(plane: np.ndarray) -> str:
    """Compress a uint8 plane to base64, decodable in a browser."""
    return base64.b64encode(zlib.compress(np.ascontiguousarray(plane, dtype=np.uint8).tobytes(), 9)).decode("ascii")


class ReplayRecorder:
    """Wraps an environment and writes a replay of everything played through it.

    Deliberately a wrapper rather than a change to the environment: recording is a
    concern of the person watching, not of the agent, and an unrecorded run must stay
    exactly as fast as it was.
    """

    def __init__(self, env, path: str | Path, note: str = "") -> None:
        self.env = env
        self.path = Path(path)
        self.note = note
        self._file: gzip.GzipFile | None = None
        self._previous: np.ndarray | None = None
        self._step = 0

    # Plumbing --------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        # Anything not overridden here belongs to the wrapped environment.
        return getattr(self.env, name)

    def _write(self, record: dict[str, Any]) -> None:
        assert self._file is not None
        self._file.write((json.dumps(record, separators=(",", ":")) + "\n").encode("utf-8"))

    def _channel(self, spatial: np.ndarray, name: str) -> np.ndarray:
        return spatial[self.env._bridge.channels.index(name)]

    # Recording -------------------------------------------------------------------

    def reset(self, **kwargs: Any):
        observation, info = self.env.reset(**kwargs)
        raw = info["raw"]
        spatial = raw["spatial"]

        if self._file is not None:
            self._file.close()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = gzip.open(self.path, "wb")
        self._step = 0

        channels = self.env._bridge.channels
        ore_names = [c for c in channels if c.startswith("ore_")]

        # Ores collapse into one indexed plane: they never overlap, and one plane per ore
        # would multiply the header size for no gain in what a viewer can show.
        ores = np.zeros(spatial.shape[1:], dtype=np.uint8)
        for index, name in enumerate(ore_names, start=1):
            ores[self._channel(spatial, name) > 0] = index

        self._write({
            "type": "header",
            "format": REPLAY_FORMAT,
            "task": self.env.task.name,
            "description": self.env.task.description,
            "map": self.env.task.map_name,
            "note": self.note,
            "width": int(spatial.shape[2]),
            "height": int(spatial.shape[1]),
            "core": [int(raw.get("core_x", -1)), int(raw.get("core_y", -1))],
            "ore_names": [n.removeprefix("ore_") for n in ore_names],
            "solid": _encode_plane(self._channel(spatial, "solid")),
            "ores": _encode_plane(ores),
        })

        self._previous = self._channel(spatial, "block_ally").copy()
        self._write(self._frame(raw, reward=0.0))
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._step += 1
        if self._file is not None:
            self._write(self._frame(info["raw"], reward, info.get("action")))
            if terminated or truncated:
                self._write({
                    "type": "end",
                    "step": self._step,
                    "solved": bool(self.env.task.succeeded(info["raw"])),
                    "truncated": bool(truncated),
                })
        return observation, reward, terminated, truncated, info

    def _frame(self, raw: dict[str, Any], reward: float, action: dict | None = None) -> dict:
        current = self._channel(raw["spatial"], "block_ally")
        added: list[list[int]] = []
        removed: list[list[int]] = []

        if self._previous is not None:
            changed = np.argwhere(current != self._previous)
            for y, x in changed:
                (added if current[y, x] else removed).append([int(x), int(y)])
        self._previous = current.copy()

        frame: dict[str, Any] = {
            "type": "frame",
            "step": self._step,
            "tick": round(float(raw.get("tick", 0)), 1),
            "wave": int(raw.get("wave", 0)),
            "reward": round(float(reward), 4),
            "items": raw.get("items", {}),
        }
        if added:
            frame["added"] = added
        if removed:
            frame["removed"] = removed
        if action is not None and not action.get("applied", True):
            frame["refused"] = 1
        return frame

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
        self.env.close()


def record_episode(env, policy, path: str | Path, note: str = "") -> dict[str, Any]:
    """Play one episode through a recorder and return the same summary as run_episode."""
    from gamma.policies import run_episode

    recorder = ReplayRecorder(env, path, note=note)
    try:
        if hasattr(policy, "reset"):
            policy.reset()
        return run_episode(recorder, policy)
    finally:
        if recorder._file is not None:
            recorder._file.close()
            recorder._file = None
