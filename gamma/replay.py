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


def _encode_bytes(data: bytes) -> str:
    """Compress raw bytes to base64, decodable in a browser with DecompressionStream."""
    return base64.b64encode(zlib.compress(bytes(data), 9)).decode("ascii")


def _encode_plane(plane: np.ndarray) -> str:
    return _encode_bytes(np.ascontiguousarray(plane, dtype=np.uint8).tobytes())


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
        self._previous_raw: dict[str, Any] | None = None
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

        # The typed map is what lets a viewer draw the game with its own sprites. It is
        # fetched once per episode: 458 KB raw for a 256x256 map, a few dozen KB once
        # compressed, against 413 MB if the observation tensor were stored per step.
        typed = self.env._bridge.map()
        planes = typed["spatial"]
        tiles = typed["width"] * typed["height"]

        self._write({
            "type": "header",
            "format": REPLAY_FORMAT,
            "task": self.env.task.name,
            "description": self.env.task.description,
            "map": self.env.task.map_name,
            "note": self.note,
            # Enough to put the same world back on a server and re-run the episode for
            # real. A generated sector has no name, so the index is the only handle on
            # it, and without it `tools/watch.py` had nothing to load and the in-game
            # reader said "Map not found:" with nothing after the colon.
            "sector": self.env.task.sector,
            "sector_index": self.env.sector_index,
            "loadout": self.env.task.loadout,
            # Whether the agent had a body. A replay of an embodied episode re-run
            # without one shows blocks appearing out of nowhere instead of a unit flying
            # over to build them, which is not what happened.
            "embodied": bool(self.env.embodied),
            "ticks_per_step": self.env.task.ticks_per_step,
            "width": typed["width"],
            "height": typed["height"],
            "core": [int(raw.get("core_x", -1)), int(raw.get("core_y", -1))],
            "palette": typed["palette"],
            "blocks": list(self.env.blocks),
            # Each plane base64 of zlib, decodable in a browser with DecompressionStream.
            "floor": _encode_bytes(planes[0:tiles * 2]),
            "overlay": _encode_bytes(planes[tiles * 2:tiles * 4]),
            "block": _encode_bytes(planes[tiles * 4:tiles * 6]),
            "rotation": _encode_bytes(planes[tiles * 6:tiles * 7]),
        })

        self._previous = self._channel(spatial, "block_ally").copy()
        self._write(self._frame(raw, reward=0.0))
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._step += 1
        if self._file is not None:
            self._write(self._frame(info["raw"], reward, info.get("action"), action))
            if terminated or truncated:
                self._write({
                    "type": "end",
                    "step": self._step,
                    "solved": bool(self.env.task.succeeded(info["raw"])),
                    "truncated": bool(truncated),
                })
        return observation, reward, terminated, truncated, info

    def _frame(
        self,
        raw: dict[str, Any],
        reward: float,
        outcome: dict | None = None,
        action: Any = None,
    ) -> dict:
        current = self._channel(raw["spatial"], "block_ally")
        added: list[list[int]] = []
        removed: list[list[int]] = []

        if self._previous is not None and self._previous.shape != current.shape:
            # The map changed size under the recording, which the environment refuses on
            # the next reset. Nothing here can salvage the file, but the diff is not the
            # place to report it: a broadcast error from a delta hides the actual fault.
            self._previous = None

        if self._previous is not None:
            changed = np.argwhere(current != self._previous)
            for y, x in changed:
                (added if current[y, x] else removed).append([int(x), int(y)])
        self._previous = current.copy()

        # Where the points came from, not just how many. A total is not diagnosable: a
        # rising curve was once an annuity on a generator, and tracking down a steady
        # stream of delivery points afterwards took three separate probes against a live
        # server because no episode could say what it had been paid for. Only the non-zero
        # terms are written, so a quiet step costs nothing.
        # Read off what the reward actually paid, rather than recomputing it. The reward
        # keeps a per-episode ledger, so asking it a second time about the same step would
        # answer against a ledger that has already moved on.
        paid = getattr(self.env.task.reward, "last_terms", None)
        itemised = None
        if paid:
            itemised = {k: round(v, 4) for k, v in paid.items() if abs(v) > 1e-9}
        self._previous_raw = raw

        frame: dict[str, Any] = {
            "type": "frame",
            "step": self._step,
            "tick": round(float(raw.get("tick", 0)), 1),
            "wave": int(raw.get("wave", 0)),
            "reward": round(float(reward), 4),
            "items": raw.get("items", {}),
        }
        if itemised:
            frame["terms"] = itemised
        if added:
            frame["added"] = added
        if removed:
            frame["removed"] = removed
        if outcome is not None and not outcome.get("applied", True):
            frame["refused"] = 1

        # The action itself is recorded, not just which tiles changed. Deltas say a tile
        # became occupied; only the action says by what, and a viewer needs the block
        # identity to pick a sprite. It also costs less than the deltas it replaces.
        if action is not None and outcome is not None and outcome.get("applied"):
            # By name, never by index. The embodied action space puts `move` and `build`
            # where the direct one puts `place` and `break`, so a hardcoded index records
            # a move as a construction and the viewer draws a block that was never built.
            kind = self.env.action_types[int(action[0])]
            if kind in ("place", "build"):
                frame["act"] = {
                    "t": "place",
                    "b": self.env.blocks[int(action[1])],
                    "x": int(action[2]), "y": int(action[3]), "r": int(action[4]),
                }
            elif kind == "break":
                frame["act"] = {"t": "break", "x": int(action[2]), "y": int(action[3])}
            elif kind in ("move", "mine"):
                frame["act"] = {"t": kind, "x": int(action[2]), "y": int(action[3])}
            elif kind == "unload":
                # How an embodied agent banks what it carries. Left out, a re-run of the
                # episode never pays anything into the core, so it cannot afford the
                # blocks it went on to build and the replay diverges from what happened.
                frame["act"] = {"t": "unload"}
            elif kind == "stamp":
                # A whole structure in one decision, so the frame carries how much of it
                # the world actually accepted. `asked` against `laid` is the difference
                # between a design the policy placed well and one it dropped on a wall.
                frame["act"] = {
                    "t": "stamp",
                    "d": int(outcome.get("design", 0)),
                    "x": int(outcome.get("x", 0)), "y": int(outcome.get("y", 0)),
                    "laid": int(outcome.get("laid", 0)),
                    "asked": int(outcome.get("asked", 0)),
                }
        return frame

    def finish(self) -> None:
        """Close the current recording without closing the environment.

        A run that records episode after episode needs the file sealed before it can be
        renamed by its result, while the environment carries straight on into the next
        episode. Closing both together would end the run at the first episode.
        """
        if self._file is not None:
            self._file.close()
            self._file = None

    def close(self) -> None:
        self.finish()
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
