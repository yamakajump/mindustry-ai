"""Watch an agent play inside the real Mindustry client.

The web viewer draws a map. This shows the game: conveyor animations, items travelling
along them, block connections, lighting, wave effects. Everything, because it *is* the
game rendering it.

No client mod is needed. The training server already speaks Mindustry's own network
protocol, so a stock client can simply join it.

    python tools/watch.py replays/showcase/alpha-t1.jsonl.gz

Then in Mindustry: Play, Join Game, 127.0.0.1:6567. The replay starts on its own once you
are in, and the server stays up afterwards so you can look around.
"""

from __future__ import annotations

import argparse
import gzip
import json
import time
from pathlib import Path

from gamma.bridge import Bridge
from gamma.server import ServerProcess, install_plugin
from gamma.server_setup import setup_server

BRIDGE_PORT = 7999
GAME_PORT = 6567


def read_replay(path: Path) -> tuple[dict, list[dict]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle]
    header = records[0]
    frames = [r for r in records if r.get("type") == "frame"]
    return header, frames


def someone_watching(server: ServerProcess) -> bool:
    """Whether a player has joined. The server says so in its status line."""
    try:
        line = server.command("status", r"[Nn]o players|players? connected", timeout=10)
        return "No players" not in line
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("replay", type=Path, help="a .jsonl.gz recorded episode")
    parser.add_argument("--speed", default="2", help="simulation speed, 1 is realtime")
    parser.add_argument("--port", type=int, default=GAME_PORT)
    parser.add_argument("--wait", type=int, default=300, help="seconds to wait for you to join")
    parser.add_argument("--start-anyway", action="store_true", help="do not wait for a player")
    parser.add_argument("--jar", type=Path, default=None)
    args = parser.parse_args()

    header, frames = read_replay(args.replay)
    jar = args.jar or next((Path("bridge") / "build" / "libs").glob("*.jar"))

    server_dir = setup_server("mindustry-watch")
    install_plugin(server_dir, jar)

    print(f"replay : {args.replay.name}")
    print(f"task   : {header['task']} on {header['map']}, {len(frames)} steps")
    print("starting server...")

    with ServerProcess(
        server_dir, jvm_args=[f"-Dmindustryai.port={BRIDGE_PORT}"], port=args.port
    ) as server:
        server.wait_for(rf"listening on 127\.0\.0\.1:{BRIDGE_PORT}", timeout=120)

        with Bridge(port=BRIDGE_PORT) as bridge:
            if header.get("sector"):
                bridge.sector(header["sector"], header.get("loadout"))
            else:
                bridge.reset(header["map"].replace(" ", "_"), "survival")

            # Realtime-ish on purpose: the point here is to watch, not to train.
            server.command(f"bridge-speed {args.speed}", r"speed set")

            print()
            print("=" * 64)
            print("  READY. In Mindustry:  Play  ->  Join Game  ->  + Add Server")
            print(f"  Address:  127.0.0.1:{args.port}")
            print("=" * 64)
            print()

            if not args.start_anyway:
                deadline = time.monotonic() + args.wait
                announced = False
                while time.monotonic() < deadline:
                    if someone_watching(server):
                        print("player joined, starting the replay")
                        break
                    if not announced:
                        print(f"waiting up to {args.wait}s for you to join...")
                        announced = True
                    time.sleep(3)
                else:
                    print("nobody joined, replaying anyway")

            for index, frame in enumerate(frames):
                action = frame.get("act")
                if action is None:
                    bridge.step(repeat=30)
                else:
                    payload = (
                        {
                            "type": "place", "block": action["b"],
                            "x": action["x"], "y": action["y"], "rotation": action.get("r", 0),
                        }
                        if action["t"] == "place"
                        else {"type": "break", "x": action["x"], "y": action["y"]}
                    )
                    bridge.step(repeat=30, action=payload)

                if index % 50 == 0:
                    print(f"  step {index}/{len(frames)}")

            print()
            print("replay finished. The server stays up: keep looking around, then close")
            print("this window (or press Ctrl+C) when you are done.")

            # Keep the world ticking so the client stays connected and animated.
            try:
                while True:
                    bridge.step(repeat=60)
            except KeyboardInterrupt:
                print("stopping")


if __name__ == "__main__":
    main()
