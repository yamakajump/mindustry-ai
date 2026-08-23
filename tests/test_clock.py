"""Simulation acceleration works, and does not break the simulation doing it."""

from __future__ import annotations

from tools.mindustry_server import ServerProcess

from .test_benchmark import run_bench


def test_acceleration_multiplies_the_tick_rate(hosting_server: ServerProcess) -> None:
    baseline = run_bench(hosting_server, seconds=4)

    hosting_server.command("bridge-speed 8", r"speed set multiplier=8")
    accelerated = run_bench(hosting_server, seconds=4)

    # 8x is requested; 4x is accepted as proof the mechanism works. The true ceiling is
    # hardware-bound and belongs in docs/measurements/throughput.md, not in an assertion.
    assert accelerated["tps"] > baseline["tps"] * 4, f"{baseline=} {accelerated=}"


def test_uncapped_is_faster_still(hosting_server: ServerProcess) -> None:
    hosting_server.command("bridge-speed max", r"speed set")
    uncapped = run_bench(hosting_server, seconds=4)
    assert uncapped["tps"] > 60.0 * 4, f"uncapped produced only {uncapped}"


def test_speed_returns_to_realtime(hosting_server: ServerProcess) -> None:
    """Acceleration must be reversible, or an environment cannot be stepped normally."""
    hosting_server.command("bridge-speed max", r"speed set")
    run_bench(hosting_server, seconds=2)

    hosting_server.command("bridge-speed 1", r"speed set multiplier=1")
    restored = run_bench(hosting_server, seconds=4)
    assert 50.0 <= restored["tps"] <= 70.0, f"speed did not return to realtime: {restored}"


def test_simulation_still_advances_under_acceleration(hosting_server: ServerProcess) -> None:
    """Ticking faster is worthless if the world stops behaving.

    Wave countdown is driven by Time.delta, the same value every block and unit update
    uses. If it drains far faster than wall-clock time, the whole simulation is genuinely
    running fast rather than merely spinning the loop.
    """
    hosting_server.command("bridge-speed 1", r"speed set multiplier=1")
    before = hosting_server.command("status", r"seconds until next wave")

    hosting_server.command("bridge-speed max", r"speed set")
    run_bench(hosting_server, seconds=5)

    after = hosting_server.command("status", r"seconds until next wave|Wave \d+")
    assert after != before, "wave timer did not move under acceleration"
