"""Play a recorded episode through the live dashboard.

    python tools/watch_replay.py showcase/connect.jsonl.gz

A thin front for `gamma.playback`, which is where the reading lives so that the training
dashboard can offer the same thing without a second implementation. Use this when no run is
going; when one is, the dashboard has a replay button on every match.
"""

from __future__ import annotations

import argparse
import threading
import webbrowser
from pathlib import Path

from gamma import playback
from gamma.monitor import TrainingMonitor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay", type=Path)
    parser.add_argument("--port", type=int, default=8801,
                        help="kept off the trainer's port, so both can run at once")
    parser.add_argument("--speed", type=float, default=4.0)
    parser.add_argument("--no-open", dest="open", action="store_false", default=True)
    args = parser.parse_args()

    header, frames = playback.read(args.replay)
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

    state = monitor.match(0)
    playback.describe(state, header, frames, args.replay.name)

    thread = threading.Thread(
        target=playback.play, args=(monitor, state, header, frames, args.speed), daemon=True)
    thread.start()
    try:
        while thread.is_alive():
            thread.join(timeout=0.5)
    except KeyboardInterrupt:
        monitor.control("stop")


if __name__ == "__main__":
    main()
