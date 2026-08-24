"""Kill Mindustry servers left behind by a previous run.

A training run starts one headless server per environment. If the Python process is killed
rather than shut down (Ctrl+C at the wrong moment, a timeout, a crash), those servers
survive: they keep their ports, so the next run reports "no environment started" and
exits, and they quietly hold on to a gigabyte of memory each.

Only servers belonging to this project are touched, identified by `server-release.jar` on
their command line. A player's own Mindustry client is never matched.
"""

from __future__ import annotations

import subprocess
import sys
import time

_POWERSHELL_LIST = (
    "Get-CimInstance Win32_Process -Filter \"Name='java.exe'\" | "
    "Where-Object {{ $_.CommandLine -like '*server-release*' }} | "
    "Select-Object -ExpandProperty ProcessId"
)


def find_servers() -> list[int]:
    """Process ids of headless Mindustry servers started by this project."""
    if sys.platform == "win32":
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", _POWERSHELL_LIST.format()],
            capture_output=True, text=True, timeout=30,
        )
    else:
        result = subprocess.run(
            ["pgrep", "-f", "server-release.jar"], capture_output=True, text=True, timeout=30
        )
    return [int(line) for line in result.stdout.split() if line.strip().isdigit()]


def kill_servers(verbose: bool = True) -> int:
    """Terminate every leftover server. Returns how many were killed."""
    pids = find_servers()
    if not pids:
        return 0

    if verbose:
        print(f"cleaning up {len(pids)} leftover Mindustry server(s)")

    for pid in pids:
        try:
            if sys.platform == "win32":
                subprocess.run(["taskkill", "/F", "/PID", str(pid)],
                               capture_output=True, timeout=15)
            else:
                subprocess.run(["kill", "-9", str(pid)], capture_output=True, timeout=15)
        except Exception:
            pass

    # Ports are released a moment after the process dies, and binding too early fails.
    time.sleep(2)
    return len(pids)


if __name__ == "__main__":
    killed = kill_servers()
    print(f"killed {killed} server(s)" if killed else "nothing to clean up")
