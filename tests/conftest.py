"""Shared fixtures: a provisioned server with the freshly built plugin installed."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from tools.mindustry_server import ServerProcess, install_plugin
from tools.setup_server import setup_server

REPO_ROOT = Path(__file__).resolve().parent.parent
BRIDGE = REPO_ROOT / "bridge"


@pytest.fixture(scope="session")
def bridge_jar() -> Path:
    """Build the plugin once per session and return the jar."""
    gradlew = BRIDGE / ("gradlew.bat" if os.name == "nt" else "gradlew")
    subprocess.run([str(gradlew), "jar", "--no-daemon"], cwd=BRIDGE, check=True)

    jars = list((BRIDGE / "build" / "libs").glob("*.jar"))
    if not jars:
        raise RuntimeError("gradle produced no jar")
    return jars[0]


@pytest.fixture(scope="session")
def server_with_plugin(tmp_path_factory: pytest.TempPathFactory, bridge_jar: Path) -> Path:
    server_dir = setup_server(tmp_path_factory.mktemp("mindustry-plugin"))
    install_plugin(server_dir, bridge_jar)
    return server_dir


@pytest.fixture
def hosting_server(server_with_plugin: Path):
    """A server with a match already running, which is what makes ticks happen."""
    with ServerProcess(server_with_plugin) as server:
        server.send("host Ancient_Caldera survival")
        server.wait_for(r"Opened a server", timeout=90)
        yield server
