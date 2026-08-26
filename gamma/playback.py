"""Turn a recorded episode back into the thing the dashboard already knows how to draw.

There were two renderers for the same world, and only one of them was any good. The live
dashboard draws sprites, animates conveyors, follows units and shows what turrets are
aiming at; the replay viewer beside it derived what it could from the agent's actions and
drew that. Keeping both meant every improvement had to be made twice, and it was not: they
drifted until the same episode looked like two different games.

They do not have to differ at all. A replay is exactly the stream the dashboard consumes,
the same deltas produced by the same encoder, and the only thing it lacked was a way in.
So nothing here reimplements the protocol: it fills the same `MatchState` the trainer
fills and feeds `scene.apply` the frames off a disk instead of off a socket. The dashboard
cannot tell the difference, which is the point, because a second implementation of the
same accumulation would drift from the first exactly as the two renderers did.
"""

from __future__ import annotations

import gzip
import json
import threading
import time
from pathlib import Path
from typing import Any


def read(path: Path) -> tuple[dict, list[dict]]:
    """The header and the frames of a recorded episode."""
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        header = json.loads(handle.readline())
        frames = []
        for line in handle:
            record = json.loads(line)
            if record.get("type") == "frame":
                frames.append(record)
    return header, frames


def terrain_of(header: dict) -> dict:
    """The header is already the dashboard's terrain, field for field.

    Both were written by the same encoder from the same typed map, so there is nothing to
    convert. Worth stating because it is the reason this file is short.
    """
    return {
        "width": header["width"],
        "height": header["height"],
        "palette": header["palette"],
        "drop_zone_radius": header.get("drop_zone_radius", 0),
        "floor": header["floor"],
        "overlay": header["overlay"],
        "block": header["block"],
    }


def envelope(scene: dict, frame: dict, header: dict) -> dict:
    """Put back what the recorder left out, because it was already in the frame.

    A recorded scene carries only what moved. The fields around it, whether the match is
    playing and the tick and wave it is at, are dropped on the way to disk because the
    frame states them a few bytes earlier and repeating them costs about a hundred and
    twenty kilobytes over an episode.

    They still have to be there when the buffer is fed, and one of them decides everything:
    `apply` returns immediately on a frame that is not playing. Without this the world
    stays empty forever while every endpoint answers correctly, with nothing in it.
    """
    return {
        **scene,
        "playing": True,
        "tick": frame.get("tick", 0.0),
        "wave": frame.get("wave", 0),
        "width": header["width"],
        "height": header["height"],
    }


class Playback:
    """A cursor over a recording, which can be moved in either direction.

    Forwards is just reading on. Backwards cannot be: the scene is a stream of deltas, so a
    building placed at step ten and never touched again appears in step ten and in no step
    after it. Rewinding by replaying deltas from where you are would lose everything that
    has not moved recently, which on a base is most of it.

    So going back means starting the world again and replaying up to the target. It costs
    a pass over a few thousand small frames, nothing is drawn during it, and it is the only
    reading that cannot quietly lose a wall.
    """

    def __init__(self, header: dict, frames: list[dict], state: Any) -> None:
        self.header = header
        self.frames = frames
        self.state = state
        self.cursor = 0
        self.total = 0.0
        self.target: int | None = None

    def seek(self, step: int) -> None:
        """Asked for by the dashboard, honoured by the reader between two frames."""
        self.target = max(0, min(step, len(self.frames) - 1))

    def rewind(self, step: int) -> None:
        # Cleared rather than rebuilt: the version keeps climbing, so a browser holding an
        # old one is resynced with the whole world instead of handed deltas against a past
        # that no longer exists.
        self.state.scene.clear()
        self.total = 0.0
        for frame in self.frames[:step]:
            scene = frame.get("scene")
            if scene:
                self.state.scene.apply(envelope(scene, frame, self.header))
            self.total += float(frame.get("reward", 0.0))
        self.cursor = step

    def advance(self, frame: dict) -> None:
        """Fold one frame into the match the dashboard is reading."""
        state = self.state
        scene = frame.get("scene")
        if scene:
            state.scene.apply(envelope(scene, frame, self.header))

        self.total += float(frame.get("reward", 0.0))
        state.step = int(frame.get("step", 0))
        state.total_steps = state.step
        state.tick = float(frame.get("tick", 0.0))
        state.wave = int(frame.get("wave", 0))
        state.reward = self.total
        state.best_reward = max(state.best_reward, self.total)
        state.items = frame.get("items") or {}
        state.progress = state.step / max(1, len(self.frames))

        action = frame.get("act")
        if action:
            state.action = action.get("t", "")


def describe(state: Any, header: dict, frames: list[dict], name: str) -> None:
    """Fill in everything about the match that does not change while it plays."""
    state.policy = "replay"
    state.task = header.get("task", "")
    state.objective = header.get("description", "") or name
    state.max_steps = len(frames)
    state.core = header.get("core", [-1, -1])
    state.size = [header["width"], header["height"]]
    state.terrain = terrain_of(header)
    state.terrain_version += 1
    state.terrain_size = [header["width"], header["height"]]
    state.alive = True
    state.finished = 0


def play(monitor: Any, state: Any, header: dict, frames: list[dict],
         speed: float, stopping: threading.Event | None = None) -> None:
    """Walk a recording at a wall-clock pace, honouring pause and seek.

    A recorded step is thirty ticks, which is half a second of game time, so a speed of one
    plays it back at the pace it was lived at.
    """
    playback = Playback(header, frames, state)
    monitor.seeker = playback.seek
    monitor.length = len(frames)

    while playback.cursor < len(frames):
        if stopping is not None and stopping.is_set():
            break
        monitor.running.wait()
        if monitor.stopping.is_set():
            break

        if playback.target is not None:
            playback.rewind(playback.target)
            playback.target = None

        playback.advance(frames[playback.cursor])
        playback.cursor += 1
        time.sleep(0.5 / max(0.1, speed))

    state.alive = False
    state.finished = 1
