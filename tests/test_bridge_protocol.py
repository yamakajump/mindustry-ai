"""The agent socket speaks the protocol, and stepping is exact and synchronous."""

from __future__ import annotations

import json
import socket
import struct
import time

import pytest

from gamma.bridge import PROTOCOL_VERSION, Bridge, BridgeError


def test_handshake_reports_versions(bridge: Bridge) -> None:
    hello = bridge.request({"cmd": "hello"})
    assert hello["protocol"] == PROTOCOL_VERSION
    assert hello["mindustry"].startswith("159")
    assert hello["clock"] == "ok"


def test_reset_starts_a_match(bridge: Bridge) -> None:
    obs = bridge.reset("Ancient_Caldera", "survival")
    assert obs["playing"] is True
    assert obs["wave"] == 1
    assert obs["has_core"] is True
    assert obs["map_width"] > 0
    assert obs["items"]["copper"] > 0, "a survival start grants the core some copper"


def test_world_is_frozen_between_decisions(bridge: Bridge) -> None:
    """The core guarantee. If the world moves while the agent thinks, the state an
    action was chosen for is stale by the time it lands, and nothing is reproducible."""
    bridge.reset("Ancient_Caldera", "survival")
    before = bridge.observe()["tick"]
    time.sleep(1.5)
    after = bridge.observe()["tick"]
    assert after == before, f"world advanced {after - before} ticks while idle"


@pytest.mark.parametrize("repeat", [1, 15, 100])
def test_step_advances_exactly_the_requested_ticks(bridge: Bridge, repeat: int) -> None:
    bridge.reset("Ancient_Caldera", "survival")
    before = bridge.observe()["tick"]
    bridge.step(repeat=repeat)
    after = bridge.observe()["tick"]
    assert after - before == pytest.approx(repeat, abs=0.01)


def test_steps_accumulate(bridge: Bridge) -> None:
    bridge.reset("Ancient_Caldera", "survival")
    start = bridge.observe()["tick"]
    for _ in range(5):
        bridge.step(repeat=60)
    assert bridge.observe()["tick"] - start == pytest.approx(300, abs=0.01)


def test_wave_timer_drains_as_the_world_runs(bridge: Bridge) -> None:
    bridge.reset("Ancient_Caldera", "survival")
    before = bridge.observe()["wave_time"]
    bridge.step(repeat=120)
    after = bridge.observe()["wave_time"]
    assert after < before, "wave countdown did not move, so the world did not really run"


# Failure handling ---------------------------------------------------------------


def test_unknown_command_is_reported(bridge: Bridge) -> None:
    with pytest.raises(BridgeError, match="unknown command"):
        bridge.request({"cmd": "nonsense"})


def test_missing_command_is_reported(bridge: Bridge) -> None:
    with pytest.raises(BridgeError, match="missing"):
        bridge.request({"repeat": 1})


def test_malformed_json_does_not_kill_the_connection(bridge: Bridge) -> None:
    """A bad frame must be survivable: an agent that crashes the bridge on one typo
    makes every training run fragile."""
    bridge._send(b"{not json at all")
    _, payload = bridge._receive()
    assert json.loads(payload)["ok"] is False

    assert bridge.request({"cmd": "hello"})["ok"] is True


def test_disconnecting_mid_step_does_not_poison_the_next_client(bridge_server, bridge_ports) -> None:
    """Regression test.

    A step spans many ticks, so an agent can vanish while one is still running. If the
    bridge delivers that late reply anyway, it is handed to whoever connects next: their
    handshake returns an observation, every later request answers the previous one, and
    the session dies on a timeout with nothing in the logs explaining why. Found on CI,
    where the machine was slow enough to make the race reliable.
    """
    victim = Bridge(port=bridge_ports[0], timeout=30.0)
    victim.connect()
    victim.reset("Ancient_Caldera", "survival")

    # Ask for a long step, then leave without reading the answer.
    victim._send(json.dumps({"cmd": "step", "repeat": 600}).encode("utf-8"))
    assert victim._sock is not None
    victim._sock.close()
    victim._sock = None

    with Bridge(port=bridge_ports[0], timeout=30.0) as successor:
        hello = successor.request({"cmd": "hello"})
        assert "protocol" in hello, f"got a stale reply meant for the previous agent: {hello}"

        # And the connection is genuinely usable, not merely one message ahead.
        observed = successor.observe()
        assert "tick" in observed


def test_oversized_frame_is_refused(bridge_server, bridge_ports) -> None:
    """Guards against a length prefix being trusted enough to allocate from.

    Uses its own short-timeout connection rather than the shared fixture. The bridge
    answers by dropping the connection, and how quickly that surfaces to the client
    depends on the platform: Windows delivers a reset immediately, Linux can leave the
    reader waiting until it times out. A dedicated socket keeps that cost bounded and
    stops a deliberately broken connection from leaking into later tests.
    """
    with Bridge(port=bridge_ports[0], timeout=10.0) as probe:
        assert probe._sock is not None
        probe._sock.sendall(struct.pack(">BI", 0, 2**30 + 1))
        with pytest.raises((ConnectionError, OSError, TimeoutError, socket.timeout)):
            probe._receive()

    # The server must still take new connections after refusing that frame.
    with Bridge(port=bridge_ports[0], timeout=30.0) as after:
        assert after.request({"cmd": "hello"})["ok"] is True
