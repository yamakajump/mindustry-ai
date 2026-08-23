"""Client for the bridge socket exposed by the Mindustry plugin.

Speaks the framing described in `bridge/src/mindustryai/net/Protocol.java`: one type
byte, a four byte big-endian length, then the payload.

The connection is synchronous by design. Every request receives exactly one reply, and
the world does not advance between them, so an observation always describes the state the
next action will be applied to.
"""

from __future__ import annotations

import json
import socket
import struct
import time
from typing import Any

import numpy as np

TYPE_JSON = 0
TYPE_BINARY = 1
PROTOCOL_VERSION = 1

_HEADER = struct.Struct(">BI")


class BridgeError(RuntimeError):
    """The bridge reported that a command failed."""


class Bridge:
    """A connection to one Mindustry instance."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7654,
        timeout: float = 60.0,
        tensor: bool = False,
    ) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        # Spatial tensors are large, so they are opt-in and negotiated at handshake.
        self.tensor = tensor
        self.channels: list[str] = []
        self._sock: socket.socket | None = None

    # Connection ----------------------------------------------------------------

    def connect(self, retries: int = 30, delay: float = 1.0) -> dict[str, Any]:
        """Connect and perform the handshake, retrying while the server boots.

        Returns the server's hello reply, which carries the protocol revision and the
        engine version. Both are checked, because a mismatch produces failures far more
        confusing than a clear error here.
        """
        last: OSError | None = None
        for _ in range(retries):
            try:
                self._sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
                self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                break
            except OSError as e:
                last = e
                time.sleep(delay)
        else:
            raise ConnectionError(f"no bridge on {self.host}:{self.port}") from last

        hello = self.request({"cmd": "hello", "tensor": self.tensor})
        self.channels = hello.get("channels", [])
        if hello.get("protocol") != PROTOCOL_VERSION:
            raise BridgeError(
                f"protocol mismatch: client speaks {PROTOCOL_VERSION}, "
                f"bridge speaks {hello.get('protocol')}"
            )
        if hello.get("clock") != "ok":
            raise BridgeError("bridge clock is degraded, acceleration is unavailable")
        return hello

    def close(self) -> None:
        """Say goodbye if possible, then drop the socket regardless.

        The goodbye uses a short timeout of its own: on an already broken connection the
        normal one would stall teardown for minutes, and a close that hangs is worse than
        a close that skips the courtesy.
        """
        if self._sock is None:
            return
        try:
            self._sock.settimeout(2.0)
            self.request({"cmd": "close"})
        except Exception:
            pass
        finally:
            try:
                self._sock.close()
            finally:
                self._sock = None

    def __enter__(self) -> Bridge:
        self.connect()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # Framing -------------------------------------------------------------------

    def _send(self, payload: bytes, kind: int = TYPE_JSON) -> None:
        if self._sock is None:
            raise ConnectionError("not connected")
        self._sock.sendall(_HEADER.pack(kind, len(payload)) + payload)

    def _recv_exactly(self, count: int) -> bytes:
        if self._sock is None:
            raise ConnectionError("not connected")
        chunks = []
        remaining = count
        while remaining:
            chunk = self._sock.recv(remaining)
            if not chunk:
                raise ConnectionError("bridge closed the connection")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _receive(self) -> tuple[int, bytes]:
        kind, length = _HEADER.unpack(self._recv_exactly(_HEADER.size))
        return kind, self._recv_exactly(length)

    # Commands ------------------------------------------------------------------

    def request(self, message: dict[str, Any]) -> dict[str, Any]:
        """Send one command and return its reply, raising on a reported failure.

        When the reply announces a tensor, the binary frame that follows is read and
        exposed as `reply["spatial"]`, a numpy array shaped (channels, height, width).
        """
        self._send(json.dumps(message).encode("utf-8"))
        kind, payload = self._receive()
        if kind != TYPE_JSON:
            raise BridgeError(f"expected a JSON frame, got type {kind}")

        reply = json.loads(payload.decode("utf-8"))
        if not reply.get("ok", False):
            raise BridgeError(reply.get("error", "unknown error"))

        spec = reply.get("tensor")
        if isinstance(spec, dict):
            # The layout can change between maps, so trust what this frame declares
            # rather than what the handshake said.
            self.channels = spec.get("channels", self.channels)
            reply["spatial"] = self._read_tensor(spec)
        return reply

    def _read_tensor(self, spec: dict[str, Any]) -> np.ndarray:
        kind, payload = self._receive()
        if kind != TYPE_BINARY:
            raise BridgeError(f"expected a binary frame, got type {kind}")

        shape = tuple(spec["shape"])
        expected = int(np.prod(shape))
        if len(payload) != expected:
            raise BridgeError(
                f"tensor frame is {len(payload)} bytes, shape {shape} needs {expected}"
            )

        # frombuffer avoids a copy, but the buffer is read-only and reused, so reshape
        # into a fresh array the caller can safely hold on to.
        return np.frombuffer(payload, dtype=np.uint8).reshape(shape)

    def reset(self, map_name: str | None = None, mode: str = "survival") -> dict[str, Any]:
        """Load a map and start a match. Returns the initial observation."""
        message: dict[str, Any] = {"cmd": "reset", "mode": mode}
        if map_name is not None:
            message["map"] = map_name
        return self.request(message)

    def step(self, repeat: int = 15) -> dict[str, Any]:
        """Advance the world by `repeat` ticks and return the resulting observation."""
        return self.request({"cmd": "step", "repeat": repeat})

    def observe(self) -> dict[str, Any]:
        """Read current state without advancing the world."""
        return self.request({"cmd": "observe"})
