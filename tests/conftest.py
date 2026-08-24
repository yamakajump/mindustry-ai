"""Shared fixtures: a provisioned server with the freshly built plugin installed."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from gamma.server import ServerProcess, install_plugin
from gamma.server_setup import setup_server
from gamma import tasks
from gamma.bridge import Bridge
from gamma.env import MindustryEnv

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


# Base ports, offset per test module. Two modules holding module-scoped servers can
# overlap during teardown, and sharing one port makes the newcomer fail to listen while
# its client silently connects to the outgoing server. Distinct from the defaults too,
# so a stray process from an earlier run cannot quietly satisfy a connection.
BRIDGE_PORT_BASE = 7810
GAME_PORT_BASE = 6810

# Deliberately explicit rather than hashed: a port collision is painful to diagnose, and
# a reader should be able to see which module owns which port.
_MODULE_PORT_OFFSETS = {
    "test_bridge_protocol": 0,
    "test_observations": 1,
    "test_actions": 2,
}

BRIDGE_PORT = BRIDGE_PORT_BASE
"""Default for tests that do not care. Real ports come from the fixture."""


def _ports_for(module_name: str) -> tuple[int, int]:
    offset = _MODULE_PORT_OFFSETS.get(module_name.rsplit(".", 1)[-1], 9)
    return BRIDGE_PORT_BASE + offset, GAME_PORT_BASE + offset


@pytest.fixture(scope="module")
def bridge_ports(request) -> tuple[int, int]:
    return _ports_for(request.module.__name__)


@pytest.fixture(scope="module")
def bridge_server(tmp_path_factory: pytest.TempPathFactory, bridge_jar: Path, bridge_ports):
    """A server whose agent socket is listening, shared across one module."""
    bridge_port, game_port = bridge_ports
    server_dir = setup_server(tmp_path_factory.mktemp("mindustry-bridge"))
    install_plugin(server_dir, bridge_jar)
    with ServerProcess(
        server_dir,
        jvm_args=[f"-Dmindustryai.port={bridge_port}"],
        port=game_port,
    ) as server:
        server.wait_for_bridge(bridge_port, timeout=60)
        # Run at the speed the environment will actually use. At 1x a 300 tick step
        # costs five real seconds, which made the CI run take minutes per test.
        server.command("bridge-speed max", r"speed set")
        yield server


@pytest.fixture
def bridge(bridge_server, bridge_ports) -> Bridge:
    """A fresh connection per test, so one test's state cannot leak into the next."""
    with Bridge(port=bridge_ports[0]) as client:
        yield client


ENV_BRIDGE_PORT = 7860
ENV_GAME_PORT = 6860


@pytest.fixture(scope="session")
def env(tmp_path_factory: pytest.TempPathFactory, bridge_jar: Path):
    """One environment for the whole session: starting a server per test would dominate."""
    environment = MindustryEnv(
        tasks.T1_COPPER,
        server_dir=str(tmp_path_factory.mktemp("mindustry-env")),
        bridge_port=ENV_BRIDGE_PORT,
        game_port=ENV_GAME_PORT,
        jar=str(bridge_jar),
    )
    try:
        yield environment
    finally:
        environment.close()


