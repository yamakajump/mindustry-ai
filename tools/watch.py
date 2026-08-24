"""Watch an agent play inside the real Mindustry client.

The web viewer draws a map. This shows the game: conveyor animations, items travelling
along them, block connections, lighting, unit sprites, wave effects. Everything, because
it *is* the game rendering it.

No mod is needed on the client side. The training server already speaks Mindustry's own
network protocol, so a stock client can simply join it. This script starts a server, sets
a watchable speed, replays a recorded episode into it, and waits for you to connect.

    python tools/watch.py replays/showcase/alpha-t1.jsonl.gz

Then in Mindustry: Play, Join Game, add 127.0.0.1:6567, and fly around while it happens.
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("replay", type=Path, help="a .jsonl.gz recorded episode")
    parser.add_argument("--speed", default="2", help="simulation speed: 1 is realtime")
    parser.add_argument("--port", type=int, default=GAME_PORT, help="game port for the client")
    parser.add_argument("--wait", type=int, default=45, help="seconds to wait for you to join")
    parser.add_argument("--jar", type=Path, default=None)
    args = parser.parse_args()

    header, frames = read_replay(args.replay)
    jar = args.jar or next((Path("bridge") / "build" / "libs").glob("*.jar"))

    server_dir = setup_server("mindustry-watch")
    install_plugin(server_dir, jar)

    print(f"replay : {args.replay.name}")
    print(f"task   : {header['task']} on {header['map']}")
    print(f"frames : {len(frames)}")

    with ServerProcess(
        server_dir, jvm_args=[f"-Dmindustryai.port={BRIDGE_PORT}"], port=args.port
    ) as server:
        server.wait_for(rf"listening on 127\.0\.0\.1:{BRIDGE_PORT}", timeout=90)

        with Bridge(port=BRIDGE_PORT) as bridge:
            sector = header.get("sector")
            if sector:
                bridge.sector(sector, header.get("loadout"))
            else:
                bridge.reset(header["map"].replace(" ", "_"), "survival")

            # Realtime-ish, because the point is to watch. Acceleration is for training.
            server.command(f"bridge-speed {args.speed}", r"speed set")

            print()
            print("=" * 62)
            print(f"  Open Mindustry  ->  Play  ->  Join Game  ->  127.0.0.1:{args.port}")
            print(f"  Starting in {args.wait}s, or press Enter once you have joined.")
            print("=" * 62)
            print()

            # Give a human time to alt-tab and connect before anything happens.
            deadline = time.monotonic() + args.wait
            while time.monotonic() < deadline:
                if "players" in server.command("status", r"player", timeout=10).lower():
                    line = server.command("status", r"player", timeout=10)
                    if "No players" not in line:
                        print("player connected, starting")
                        break
                time.sleep(2)

            print("replaying...")
            for index, frame in enumerate(frames):
                action = frame.get("act")
                if action is not None:
                    payload = (
                        {"type": "place", "block": action["b"], "x": action["x"],
                         "y": action["y"], "rotation": action.get("r", 0)}
                        if action["t"] == "place"
                        else {"type": "break", "x": action["x"], "y": action["y"]}
                    )
                    bridge.step(repeat=30, action=payload)
                else:
                    bridge.step(repeat=30)

                if index % 50 == 0:
                    print(f"  step {index}/{len(frames)}")

            print("replay finished, server stays up so you can look around")
            input("press Enter to stop the server\n")


if __name__ == "__main__":
    main()
