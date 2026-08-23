"""The throughput benchmark reports a number that means what it says."""

from __future__ import annotations

import re

from tools.mindustry_server import ServerProcess

BENCH_LINE = re.compile(
    r"bench frames=(\d+) ticks=(\d+) seconds=([\d.]+) fps=([\d.]+) tps=([\d.]+)"
)


def run_bench(server: ServerProcess, seconds: int = 5) -> dict[str, float]:
    line = server.command(
        f"bridge-bench {seconds}", r"bench frames=", timeout=seconds * 6 + 60
    )
    match = BENCH_LINE.search(line)
    assert match, f"unparseable benchmark line: {line}"
    return {
        "frames": float(match.group(1)),
        "ticks": float(match.group(2)),
        "seconds": float(match.group(3)),
        "fps": float(match.group(4)),
        "tps": float(match.group(5)),
    }


def test_default_speed_is_about_sixty_ticks_per_second(hosting_server: ServerProcess) -> None:
    """Mindustry targets 60 TPS. A result far from that means the counter is wrong,
    not that the machine is slow, and would make every later measurement worthless."""
    result = run_bench(hosting_server)
    assert 50.0 <= result["tps"] <= 70.0, f"unexpected default tick rate: {result}"


def test_benchmark_window_is_respected(hosting_server: ServerProcess) -> None:
    result = run_bench(hosting_server, seconds=3)
    assert 2.5 <= result["seconds"] <= 4.5, f"window drifted: {result}"


def test_ticks_track_frames_while_a_match_runs(hosting_server: ServerProcess) -> None:
    """A large gap means time is going somewhere other than simulating the world."""
    result = run_bench(hosting_server)
    assert result["ticks"] > 0, "no game ticks were counted during a live match"
    assert result["ticks"] >= result["frames"] * 0.8, f"ticks lag frames badly: {result}"
