"""Measure simulation throughput and write the results table.

Reports two independent figures per speed setting:

- **tps**, game updates per wall-clock second, counted inside the engine.
- **game time ratio**, how much game time elapsed per second of real time, derived
  from `state.tick`.

They must agree. Ticks alone could be inflated by a loop that spins without advancing
the world; game time alone could be inflated by a larger timestep. Only both together
prove the simulation genuinely runs faster.
"""

from __future__ import annotations

import argparse
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from gamma.server import ServerProcess, install_plugin
from gamma.server_setup import setup_server

BENCH_LINE = re.compile(r"bench frames=(\d+) ticks=(\d+) seconds=([\d.]+) fps=([\d.]+) tps=([\d.]+)")
STATUS_LINE = re.compile(r"bridge ready .* tick=([\d.]+) wave=(\d+)")

DEFAULT_SPEEDS = [1, 2, 4, 8, 16, 32, 64, "max"]
MAP = "Ancient_Caldera"


def read_tick(server: ServerProcess) -> float:
    line = server.command("bridge-status", r"bridge ready")
    match = STATUS_LINE.search(line)
    if not match:
        raise RuntimeError(f"unparseable status line: {line}")
    return float(match.group(1))


def measure(server: ServerProcess, speed: int | str, window: int) -> dict[str, float]:
    server.command(f"bridge-speed {speed}", r"speed set")

    tick_before = read_tick(server)
    wall_before = time.monotonic()
    line = server.command(f"bridge-bench {window}", r"bench frames=", timeout=window * 6 + 60)
    wall_elapsed = time.monotonic() - wall_before
    tick_after = read_tick(server)

    match = BENCH_LINE.search(line)
    if not match:
        raise RuntimeError(f"unparseable bench line: {line}")

    tps = float(match.group(5))
    game_seconds = (tick_after - tick_before) / 60.0
    return {
        "tps": tps,
        "ratio_ticks": tps / 60.0,
        "ratio_game_time": game_seconds / wall_elapsed if wall_elapsed else 0.0,
    }


def sweep(server_dir: Path, speeds: list[int | str], window: int) -> list[tuple[str, dict[str, float]]]:
    results = []
    with ServerProcess(server_dir) as server:
        server.command(f"host {MAP} survival", r"Opened a server", timeout=90)
        for speed in speeds:
            results.append((str(speed), measure(server, speed, window)))
            print(f"  speed={speed:<4} {results[-1][1]}")
    return results


def one_instance_max(server_dir: Path, window: int, port: int | None = None) -> float:
    """Run a single uncapped instance and return its tps. Used for the scaling test."""
    with ServerProcess(server_dir, port=port) as server:
        server.command(f"host {MAP} survival", r"Opened a server", timeout=120)
        return measure(server, "max", window)["tps"]


def scaling(base_dir: Path, jar: Path, counts: list[int], window: int) -> list[tuple[int, float]]:
    """Measure aggregate throughput with several servers running at once.

    Each instance is a separate JVM because the Mindustry engine is not thread safe,
    so this is the only parallelism available and the only one worth measuring.
    """
    results = []
    for count in counts:
        jobs = []
        for i in range(count):
            d = setup_server(base_dir / f"inst{count}_{i}")
            install_plugin(d, jar)
            jobs.append((d, 6600 + i))

        start = time.monotonic()
        with ThreadPoolExecutor(max_workers=count) as pool:
            each = list(pool.map(lambda j: one_instance_max(j[0], window, j[1]), jobs))
        total = sum(each)
        results.append((count, total))
        print(
            f"  instances={count:<3} total_tps={total:>10.0f} "
            f"per_instance={total / count:>9.0f} wall={time.monotonic() - start:.0f}s"
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", type=int, default=5, help="seconds per measurement")
    parser.add_argument("--root", type=Path, default=Path("mindustry-bench"))
    parser.add_argument("--instances", type=int, nargs="*", default=[1, 4, 12, 24])
    parser.add_argument("--skip-scaling", action="store_true")
    args = parser.parse_args()

    jar = next((Path("bridge") / "build" / "libs").glob("*.jar"))
    args.root.mkdir(parents=True, exist_ok=True)

    main_dir = setup_server(args.root / "main")
    install_plugin(main_dir, jar)

    print("Speed sweep, single instance:")
    speed_results = sweep(main_dir, DEFAULT_SPEEDS, args.window)

    scale_results: list[tuple[int, float]] = []
    if not args.skip_scaling:
        print("\nScaling across processes:")
        scale_results = scaling(args.root, jar, args.instances, args.window)

    print("\n--- markdown ---\n")
    print("| Speed setting | Ticks per second | Ratio (ticks) | Ratio (game time) |")
    print("|---|---|---|---|")
    for speed, r in speed_results:
        print(
            f"| {speed} | {r['tps']:,.0f} | {r['ratio_ticks']:.1f}x | {r['ratio_game_time']:.1f}x |"
        )

    if scale_results:
        print("\n| Instances | Aggregate TPS | Per instance | Efficiency |")
        print("|---|---|---|---|")
        single = scale_results[0][1] if scale_results else 1.0
        for count, total in scale_results:
            per = total / count
            print(f"| {count} | {total:,.0f} | {per:,.0f} | {per / single * 100:.0f}% |")


if __name__ == "__main__":
    main()
