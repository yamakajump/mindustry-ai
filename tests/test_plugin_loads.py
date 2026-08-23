"""The bridge plugin is loaded by the server and answers its commands."""

from __future__ import annotations

from pathlib import Path

from gamma.server import ServerProcess


def test_plugin_is_listed_by_the_server(server_with_plugin: Path) -> None:
    with ServerProcess(server_with_plugin) as server:
        server.send("mods")
        assert server.wait_for(r"mindustry-ai")


def test_bridge_status_answers(server_with_plugin: Path) -> None:
    with ServerProcess(server_with_plugin) as server:
        server.send("bridge-status")
        line = server.wait_for(r"bridge ready")
        assert "version=0.1.0" in line, line


def test_clock_installs_cleanly(server_with_plugin: Path) -> None:
    """A degraded clock means acceleration is unavailable, which blocks everything."""
    with ServerProcess(server_with_plugin) as server:
        server.send("bridge-status")
        line = server.wait_for(r"bridge ready")
        assert "clock=ok" in line, f"clock failed to install: {line}"
