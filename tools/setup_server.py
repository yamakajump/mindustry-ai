"""Provision a pinned Mindustry headless server.

The version is pinned deliberately. A silent engine bump would invalidate every
measurement recorded in this repository, so `latest` is never used.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

MINDUSTRY_VERSION = "v159.7"
SERVER_JAR_URL = (
    "https://github.com/Anuken/Mindustry/releases/download/{version}/server-release.jar"
)

# The v159.7 server jar is roughly 18.3 MB. Anything much smaller is a truncated
# download or an error page saved under the wrong name.
MIN_JAR_BYTES = 15_000_000


def setup_server(root: Path, version: str = MINDUSTRY_VERSION) -> Path:
    """Lay out a server directory and return it.

    Idempotent: an already valid jar is left untouched, so tests can share one
    provisioned directory across a session without re-downloading.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "config" / "mods").mkdir(parents=True, exist_ok=True)

    jar = root / "server-release.jar"
    if jar.exists() and jar.stat().st_size >= MIN_JAR_BYTES:
        return root

    url = SERVER_JAR_URL.format(version=version)
    partial = jar.with_suffix(".part")
    urllib.request.urlretrieve(url, partial)

    size = partial.stat().st_size
    if size < MIN_JAR_BYTES:
        partial.unlink()
        raise RuntimeError(f"downloaded jar is only {size} bytes, expected >= {MIN_JAR_BYTES}")

    partial.replace(jar)
    return root


if __name__ == "__main__":
    import sys

    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("mindustry-server")
    print(setup_server(target))
