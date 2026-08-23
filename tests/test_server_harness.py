"""The Python harness can provision, start and drive a real headless server.

These are integration tests against an actual Mindustry process. Mindustry state
only exists inside a running engine, so mocking it here would test nothing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.mindustry_server import ServerProcess
from tools.setup_server import MIN_JAR_BYTES, setup_server


@pytest.fixture(scope="session")
def server_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return setup_server(tmp_path_factory.mktemp("mindustry"))


def test_setup_downloads_pinned_server_jar(server_dir: Path) -> None:
    jar = server_dir / "server-release.jar"
    assert jar.exists()
    assert jar.stat().st_size >= MIN_JAR_BYTES


def test_setup_is_idempotent(server_dir: Path) -> None:
    jar = server_dir / "server-release.jar"
    before = jar.stat().st_mtime_ns
    setup_server(server_dir)
    assert jar.stat().st_mtime_ns == before, "an existing jar was re-downloaded"


def test_server_runs_the_pinned_version(server_dir: Path) -> None:
    """Guards the pin itself: a bumped engine invalidates every measurement."""
    with ServerProcess(server_dir) as server:
        server.send("version")
        # Matches "Version: Mindustry 8-release official / build 159.7" and not
        # the "Java Version:" line printed right after it.
        line = server.wait_for(r"Version:\s+Mindustry")
        assert "159.7" in line, f"unexpected engine version: {line}"


def test_server_hosts_a_map(server_dir: Path) -> None:
    with ServerProcess(server_dir) as server:
        server.send("host Ancient_Caldera survival")
        assert server.wait_for(r"Opened a server", timeout=90)
