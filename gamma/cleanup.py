"""Kill Mindustry servers left behind by a previous run.

A training run starts one headless server per environment. If the Python process is killed
rather than shut down (Ctrl+C at the wrong moment, a timeout, a crash), those servers
survive: they keep their ports, so the next run reports "no environment started" and
exits, and they quietly hold on to a gigabyte of memory each.

Only servers belonging to this project are touched, identified by `server-release.jar` on
their command line. A player's own Mindustry client is never matched.

And only servers that nobody is talking to. This used to kill every Mindustry server on
the machine, which is fine for the case it was written for and catastrophic for the case
that actually happens: a probe finishing, calling this to tidy up after itself, and taking
a training run down with it. That happened twice in one session, and from the outside it
looked like a mysterious bleed of environments dying one after another on
ConnectionResetError, which cost far more to diagnose than the tidy-up ever saved.

A live server has a client on its bridge port. A leftover does not. That is the whole
distinction, it is observable, and it is the one the word "leftover" always meant.
"""

from __future__ import annotations

import re
import subprocess
import sys
import time

_POWERSHELL_LIST = (
    "Get-CimInstance Win32_Process -Filter \"Name='java.exe'\" | "
    "Where-Object { $_.CommandLine -like '*server-release*' } | "
    "ForEach-Object { \"$($_.ProcessId) $($_.CommandLine)\" }"
)

_POWERSHELL_BUSY_PORTS = (
    "Get-NetTCPConnection -State Established -ErrorAction SilentlyContinue | "
    "Select-Object -ExpandProperty LocalPort -Unique"
)

_PORT = re.compile(r"-Dmindustryai\.port=(\d+)")


def find_servers() -> list[tuple[int, int | None]]:
    """Headless Mindustry servers started by this project, as (pid, bridge port)."""
    if sys.platform == "win32":
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", _POWERSHELL_LIST],
            capture_output=True, text=True, timeout=30,
        )
        found = []
        for line in result.stdout.splitlines():
            head, _, rest = line.strip().partition(" ")
            if head.isdigit():
                port = _PORT.search(rest)
                found.append((int(head), int(port.group(1)) if port else None))
        return found

    result = subprocess.run(
        ["pgrep", "-af", "server-release.jar"], capture_output=True, text=True, timeout=30
    )
    found = []
    for line in result.stdout.splitlines():
        head, _, rest = line.strip().partition(" ")
        if head.isdigit():
            port = _PORT.search(rest)
            found.append((int(head), int(port.group(1)) if port else None))
    return found


def busy_ports() -> set[int]:
    """Ports with a connection on them right now, so a server holding one is in use."""
    try:
        if sys.platform == "win32":
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command", _POWERSHELL_BUSY_PORTS],
                capture_output=True, text=True, timeout=30,
            )
        else:
            result = subprocess.run(["ss", "-Htan", "state", "established"],
                                    capture_output=True, text=True, timeout=30)
        return {int(tok) for tok in re.findall(r"(\d{4,5})", result.stdout)}
    except Exception:
        # Unable to tell who is in use, so nothing is safe to kill. Erring the other way
        # is what took a training run down.
        return set()


def kill_servers(verbose: bool = True, force: bool = False) -> int:
    """Terminate leftover servers. Returns how many were killed.

    Leaves alone any server with a client on its bridge port, unless `force`. That is what
    separates a leftover from a running environment, and getting it wrong is expensive:
    the caller is usually a script tidying up after itself, and the victim is usually a
    training run that has been going for hours.
    """
    servers = find_servers()
    if not servers:
        return 0

    if not force:
        in_use = busy_ports()
        spared = [(pid, port) for pid, port in servers if port in in_use]
        servers = [(pid, port) for pid, port in servers if port not in in_use]
        if spared and verbose:
            print(f"leaving {len(spared)} server(s) alone, something is connected to them")

    pids = [pid for pid, _ in servers]
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
    killed = kill_servers(force="--force" in sys.argv)
    print(f"killed {killed} server(s)" if killed else "nothing to clean up")
