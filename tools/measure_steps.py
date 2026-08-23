"""Measure agent decisions per second through the bridge socket.

This is the figure that governs training, not raw ticks. One step costs a socket round
trip, a pause and unpause of the game state, and the ticks themselves. Raw tick throughput
flatters the system by ignoring the first two.
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

from gamma.bridge import Bridge
from gamma.server import ServerProcess, install_plugin
from gamma.server_setup import setup_server

BRIDGE_PORT = 7901
GAME_PORT = 6901
MAP = "Ancient_Caldera"


def measure(bridge: Bridge, repeat: int, steps: int) -> dict[str, float]:
    bridge.reset(MAP, "survival")

    latencies = []
    start = time.perf_counter()
    for _ in range(steps):
        t0 = time.perf_counter()
        bridge.step(repeat=repeat)
        latencies.append((time.perf_counter() - t0) * 1000.0)
    elapsed = time.perf_counter() - start

    latencies.sort()
    return {
        "repeat": repeat,
        "steps_per_second": steps / elapsed,
        "game_seconds_per_second": steps * repeat / 60.0 / elapsed,
        "median_ms": statistics.median(latencies),
        "p99_ms": latencies[int(len(latencies) * 0.99) - 1],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--repeats", type=int, nargs="*", default=[1, 5, 15, 30, 60])
    parser.add_argument("--root", type=Path, default=Path("mindustry-bench"))
    args = parser.parse_args()

    jar = next((Path("bridge") / "build" / "libs").glob("*.jar"))
    server_dir = setup_server(args.root / "steps")
    install_plugin(server_dir, jar)

    rows = []
    with ServerProcess(
        server_dir, jvm_args=[f"-Dmindustryai.port={BRIDGE_PORT}"], port=GAME_PORT
    ) as server:
        server.wait_for(rf"listening on 127\.0\.0\.1:{BRIDGE_PORT}", timeout=60)
        server.command("bridge-speed max", r"speed set")

        with Bridge(port=BRIDGE_PORT) as bridge:
            for repeat in args.repeats:
                row = measure(bridge, repeat, args.steps)
                rows.append(row)
                print(
                    f"  repeat={repeat:<4} {row['steps_per_second']:>8.0f} steps/s  "
                    f"{row['game_seconds_per_second']:>7.1f}x realtime  "
                    f"median {row['median_ms']:.2f} ms  p99 {row['p99_ms']:.2f} ms"
                )

    print("\n| Ticks per step | Steps/s | Game time ratio | Median latency | p99 |")
    print("|---|---|---|---|---|")
    for r in rows:
        print(
            f"| {r['repeat']} | {r['steps_per_second']:,.0f} | "
            f"{r['game_seconds_per_second']:.1f}x | {r['median_ms']:.2f} ms | {r['p99_ms']:.2f} ms |"
        )


if __name__ == "__main__":
    main()
