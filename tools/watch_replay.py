"""Play a recorded episode through the live dashboard.

    python tools/watch_replay.py showcase/connect.jsonl.gz

There were two renderers for the same thing, and only one of them was any good. The live
dashboard is 2,429 lines: it draws sprites, animates conveyors, follows units, shows what
turrets are aiming at. The replay viewer was 482 lines that derived the world from the
agent's actions and drew what it could work out from them.

That split made sense while a replay was a list of actions and the dashboard was a live
feed. It stopped making sense the moment the recorder started writing the scene, because a
replay is now exactly the stream the dashboard already consumes: the same deltas, in the
same shape, produced by the same encoder. The good renderer simply had no way to read a
file.

So nothing here reimplements the protocol. It builds the same `TrainingMonitor` the
trainer builds, fills the same `MatchState`, and feeds `scene.apply` the frames off the
disk instead of off a socket. The dashboard cannot tell the difference, and a second
implementation of `/state` that drifts from the first is a bug waiting to be found by
someone watching a replay that quietly lies.

Playback controls come free with that decision: the dashboard's pause button pauses the
run, and here the run is a file being read.
"""

from __future__ import annotations

import argparse
import gzip
import json
import threading
import time
import webbrowser
from pathlib import Path

from gamma.monitor import TrainingMonitor


def read(path: Path) -> tuple[dict, list[dict]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        header = json.loads(handle.readline())
        frames = [json.loads(line) for line in handle
                  if json.loads(line).get("type") == "frame"]
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
    stays empty forever and every endpoint answers correctly with nothing in it, which is
    exactly what it did until this function existed.
    """
    return {
        **scene,
        "playing": True,
        "tick": frame.get("tick", 0.0),
        "wave": frame.get("wave", 0),
        "width": header["width"],
        "height": header["height"],
    }


def play(monitor: TrainingMonitor, header: dict, frames: list[dict], speed: float) -> None:
    """Walk the recording, handing each frame to the same buffer the trainer feeds."""
    state = monitor.match(0)
    state.policy = "replay"
    state.task = header.get("task", "")
    state.objective = header.get("description", "")
    state.max_steps = len(frames)
    state.core = header.get("core", [-1, -1])
    state.size = [header["width"], header["height"]]
    state.terrain = terrain_of(header)
    state.terrain_version = 1
    state.terrain_size = [header["width"], header["height"]]
    state.alive = True

    total = 0.0
    for frame in frames:
        # The dashboard's pause button, which is the same button either way: there it
        # stops the trainer asking for steps, here it stops the reader handing them over.
        monitor.running.wait()
        if monitor.stopping.is_set():
            break

        scene = frame.get("scene")
        if scene:
            state.scene.apply(envelope(scene, frame, header))

        total += float(frame.get("reward", 0.0))
        state.step = int(frame.get("step", 0))
        state.total_steps = state.step
        state.tick = float(frame.get("tick", 0.0))
        state.wave = int(frame.get("wave", 0))
        state.reward = total
        state.best_reward = max(state.best_reward, total)
        state.items = frame.get("items") or {}
        state.progress = state.step / max(1, len(frames))

        action = frame.get("act")
        if action:
            state.action = action.get("t", "")

        # A recorded step is thirty ticks, which is half a second of game time.
        time.sleep(0.5 / max(0.1, speed))

    state.alive = False
    state.finished = 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay", type=Path)
    parser.add_argument("--port", type=int, default=8801,
                        help="kept off the trainer's port, so both can run at once")
    parser.add_argument("--speed", type=float, default=4.0)
    parser.add_argument("--no-open", dest="open", action="store_false", default=True)
    args = parser.parse_args()

    header, frames = read(args.replay)
    if not frames:
        raise SystemExit(f"{args.replay} has no frames")

    scened = sum(1 for f in frames if f.get("scene"))
    if not scened:
        raise SystemExit(
            f"{args.replay} carries no scene, so there is nothing to play: it was recorded "
            "before the world was written down, and only the agent's actions are in it."
        )

    monitor = TrainingMonitor(title=f"replay / {args.replay.name}")
    url = monitor.serve(args.port)
    print(f"{len(frames):,} frames, {scened:,} with a scene")
    print(f"watching at {url}")
    if args.open:
        webbrowser.open(url)

    thread = threading.Thread(target=play, args=(monitor, header, frames, args.speed),
                              daemon=True)
    thread.start()
    try:
        while thread.is_alive():
            thread.join(timeout=0.5)
    except KeyboardInterrupt:
        monitor.control("stop")


if __name__ == "__main__":
    main()
