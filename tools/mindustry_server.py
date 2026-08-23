"""Drive a Mindustry headless server as a subprocess.

The server reads commands from stdin and logs to stdout. Output is drained on a
background thread: letting the pipe buffer fill would block the server itself,
which looks exactly like a hang in the game and wastes an afternoon.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

# Mindustry colourises its log when it believes a capable terminal is attached, which
# happens on Linux but not under the dumb terminal Windows gives it. Stripping the
# escapes here keeps patterns identical across platforms; matching them in every
# caller instead would be a permanent source of tests that pass on one OS only.
ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

# Printed once the server has finished loading and is accepting commands.
READY_PATTERN = r"Server loaded\.|Opened a server|Loaded \d+ mod"


class ServerProcess:
    """A running headless server, usable as a context manager."""

    def __init__(
        self,
        server_dir: Path,
        java: str = "java",
        jvm_args: list[str] | None = None,
        start_timeout: float = 120.0,
        port: int | None = None,
    ) -> None:
        self.server_dir = Path(server_dir)
        self.java = java
        self.jvm_args = jvm_args or []
        self.start_timeout = start_timeout
        # Every instance binds a listen port even when nothing connects to it, so
        # parallel instances need distinct ports or all but the first fail to host.
        self.port = port
        self._proc: subprocess.Popen[str] | None = None
        self._lines: list[str] = []
        self._lock = threading.Lock()
        self._reader: threading.Thread | None = None

    def __enter__(self) -> ServerProcess:
        self._proc = subprocess.Popen(
            [self.java, *self.jvm_args, "-jar", "server-release.jar"],
            cwd=self.server_dir,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        self._reader = threading.Thread(target=self._pump, daemon=True)
        self._reader.start()
        self.wait_for(READY_PATTERN, timeout=self.start_timeout)
        if self.port is not None:
            self.command(f"config port {self.port}", r"port.*set to|Port|port", timeout=20)
        return self

    def _pump(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        for line in self._proc.stdout:
            clean = ANSI_ESCAPE.sub("", line).rstrip("\r\n")
            with self._lock:
                self._lines.append(clean)

    def send(self, command: str) -> None:
        """Write one command to the server console."""
        if self._proc is None or self._proc.stdin is None:
            raise RuntimeError("server is not running")
        self._proc.stdin.write(command + "\n")
        self._proc.stdin.flush()

    def lines(self) -> list[str]:
        with self._lock:
            return list(self._lines)

    def mark(self) -> int:
        """Current output position, for use as `since` in a later `wait_for`."""
        with self._lock:
            return len(self._lines)

    def command(self, command: str, pattern: str, timeout: float = 30.0) -> str:
        """Send a command and return the first line it produces that matches.

        Prefer this over `send` followed by `wait_for`. It records the output position
        before writing, so a command issued twice in one session cannot match the reply
        to the earlier one, and it cannot miss a reply that arrives immediately either.
        """
        since = self.mark()
        self.send(command)
        return self.wait_for(pattern, timeout=timeout, since=since)

    def wait_for(self, pattern: str, timeout: float = 30.0, since: int = 0) -> str:
        """Block until an output line matches, and return that line.

        Searches the whole history by default, which is what start-up detection needs.
        For anything issued more than once in a session, use `command` instead: a bare
        `wait_for` will happily return a stale line from an earlier invocation.
        """
        deadline = time.monotonic() + timeout
        compiled = re.compile(pattern)
        seen = since
        while time.monotonic() < deadline:
            current = self.lines()
            for line in current[seen:]:
                if compiled.search(line):
                    return line
            seen = len(current)
            if self._proc is not None and self._proc.poll() is not None:
                raise RuntimeError(
                    f"server exited with code {self._proc.returncode}\n" + self._tail()
                )
            time.sleep(0.05)
        raise TimeoutError(f"no line matched {pattern!r} within {timeout}s\n" + self._tail())

    def _tail(self, count: int = 40) -> str:
        return "\n".join(self.lines()[-count:])

    def __exit__(self, *exc_info: object) -> None:
        if self._proc is None:
            return
        try:
            if self._proc.poll() is None:
                self.send("exit")
                self._proc.wait(timeout=20)
        except Exception:
            pass
        finally:
            if self._proc.poll() is None:
                self._proc.kill()
                self._proc.wait(timeout=10)


def install_plugin(server_dir: Path, jar: Path) -> Path:
    """Copy a built plugin jar into the server mod directory."""
    mods = Path(server_dir) / "config" / "mods"
    mods.mkdir(parents=True, exist_ok=True)
    target = mods / Path(jar).name
    shutil.copy2(jar, target)
    return target
