"""Build the replay mod and install it into whichever Mindustry this machine has.

The data directory is not where you would guess. The Steam build keeps mods, maps and
saves inside its own install folder under `saves/`, while the standalone build uses the
platform's usual application data path. Copying to the wrong one produces "no mods found"
and no other clue, so the location is detected rather than assumed.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

MOD_DIR = Path("replay-mod")
MOD_JAR = MOD_DIR / "build" / "libs" / "mindustry-ai-replay.jar"


def candidate_data_dirs() -> list[Path]:
    """Places Mindustry might keep its data, most likely first."""
    home = Path.home()
    candidates: list[Path] = []

    if os.name == "nt":
        program_files = [
            Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")),
            Path(os.environ.get("ProgramFiles", r"C:\Program Files")),
        ]
        for base in program_files:
            candidates.append(base / "Steam/steamapps/common/Mindustry/saves")
        # Steam libraries are often on another drive.
        for drive in "DEFGH":
            candidates.append(Path(f"{drive}:/SteamLibrary/steamapps/common/Mindustry/saves"))
        appdata = os.environ.get("APPDATA")
        if appdata:
            candidates.append(Path(appdata) / "Mindustry")
    elif sys.platform == "darwin":
        candidates.append(home / "Library/Application Support/Mindustry")
        candidates.append(home / "Library/Application Support/Steam/steamapps/common/Mindustry/saves")
    else:
        candidates.append(home / ".local/share/Mindustry")
        candidates.append(home / ".steam/steam/steamapps/common/Mindustry/saves")

    return candidates


def find_data_dir(override: Path | None) -> Path:
    if override is not None:
        return override

    # A real data directory already holds the game's own folders. An empty directory that
    # merely exists proves nothing, and picking one is how the mod ends up invisible.
    for candidate in candidate_data_dirs():
        if candidate.is_dir() and any((candidate / name).is_dir() for name in ("mods", "saves", "maps")):
            return candidate

    raise SystemExit(
        "Could not find a Mindustry data directory. Pass one with --data-dir.\n"
        "In game it is: Mods -> Open Folder, then go up one level."
    )


def build() -> Path:
    gradlew = MOD_DIR / ("gradlew.bat" if os.name == "nt" else "gradlew")
    subprocess.run([str(gradlew), "jar", "--no-daemon", "-q"], cwd=MOD_DIR, check=True)
    if not MOD_JAR.exists():
        raise SystemExit(f"gradle produced no jar at {MOD_JAR}")
    return MOD_JAR


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--replays", type=Path, default=Path("replays/showcase"))
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()

    jar = MOD_JAR if args.skip_build else build()
    data = find_data_dir(args.data_dir)
    print(f"Mindustry data directory: {data}")

    mods = data / "mods"
    mods.mkdir(parents=True, exist_ok=True)
    shutil.copy2(jar, mods / jar.name)
    print(f"installed {jar.name} -> {mods}")

    replays = data / "replays"
    replays.mkdir(parents=True, exist_ok=True)
    copied = 0
    for path in sorted(args.replays.glob("*.jsonl.gz")):
        shutil.copy2(path, replays / path.name)
        copied += 1
    print(f"copied {copied} replays -> {replays}")

    print()
    print("Restart Mindustry, then use the 'AI replay' button in the main menu.")


if __name__ == "__main__":
    main()
