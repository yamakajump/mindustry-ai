"""Extract the sprites the viewer needs from Mindustry's own asset jar.

The viewer draws the game with the game's art rather than coloured squares, which is the
difference between a debugging aid and something worth showing someone.

Mindustry is GPL-3.0 and its assets live in the same repository under the same licence,
so redistributing the subset used here is allowed. Only the sprites actually referenced
by a replay are extracted: the full set is over three thousand files.

Output is one PNG atlas plus an index JSON, both written to `viewer/assets/`.
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.request
import zipfile
from pathlib import Path

ASSETS_URL = "https://github.com/Anuken/Mindustry/releases/download/{version}/assets.jar"
DEFAULT_VERSION = "v159.7"

#: Sprite size in Mindustry is 32 pixels per world tile.
TILE = 32

#: Sprites are named by block, but the file can live anywhere under blocks/. These are
#: the directories worth searching, ordered so the common cases resolve first.
SEARCH_DIRS = (
    "environment", "distribution", "drills", "storage", "walls", "turrets",
    "production", "power", "defense", "liquid", "units", "crafting", "logic",
)


def sprite_candidates(name: str, variants: int) -> list[str]:
    """File names Mindustry might store this block under, best first.

    Numbered variants are tried for every block, not only for those the palette declares.
    Static walls and props ship as `sand-wall1.png` and the like while reporting no
    variant count, because that field only exists on floors. Trusting it lost six sprites.
    """
    names = [f"{name}{i}" for i in range(1, max(variants, 4) + 1)]
    names.append(name)
    if name == "conveyor":
        # Conveyors are animated: variant-frame. The first frame is enough for a still map.
        names.insert(0, "conveyor-0-0")
    return names


def build(jar: Path, wanted: dict[str, int], out_dir: Path) -> dict:
    """Pack every found sprite into one atlas and return its index."""
    from PIL import Image

    with zipfile.ZipFile(jar) as archive:
        entries = archive.namelist()
        by_stem: dict[str, str] = {}
        for entry in entries:
            if entry.endswith(".png") and "/sprites/" in entry:
                by_stem.setdefault(Path(entry).stem, entry)

        found: dict[str, Image.Image] = {}
        missing: list[str] = []
        for name, variants in sorted(wanted.items()):
            for candidate in sprite_candidates(name, variants):
                path = by_stem.get(candidate)
                if path:
                    with archive.open(path) as handle:
                        found[name] = Image.open(handle).convert("RGBA")
                    break
            else:
                missing.append(name)

    if not found:
        raise SystemExit("no sprites matched, is the jar correct?")

    columns = 16
    rows = (len(found) + columns - 1) // columns
    cell = max(max(img.width for img in found.values()), max(img.height for img in found.values()))
    atlas = Image.new("RGBA", (columns * cell, rows * cell), (0, 0, 0, 0))

    index: dict[str, dict] = {}
    for position, (name, image) in enumerate(sorted(found.items())):
        x = (position % columns) * cell
        y = (position // columns) * cell
        atlas.paste(image, (x, y))
        index[name] = {
            "x": x, "y": y, "w": image.width, "h": image.height,
            "tiles": max(1, round(image.width / TILE)),
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    atlas.save(out_dir / "atlas.png", optimize=True)
    (out_dir / "atlas.json").write_text(
        json.dumps({"cell": cell, "tile": TILE, "sprites": index}, indent=1), encoding="utf-8"
    )

    return {"found": len(found), "missing": missing, "atlas": out_dir / "atlas.png"}


def wanted_from_replays(replays: list[Path]) -> dict[str, int]:
    """Collect every block name a replay references, with its variant count."""
    import gzip

    wanted: dict[str, int] = {}
    for path in replays:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            header = json.loads(handle.readline())
        for entry in header.get("palette", {}).values():
            name = entry["name"]
            if name != "air":
                wanted[name] = max(wanted.get(name, 0), entry.get("variants", 0))

        # Blocks the agent can place are not on the map at load time, so the palette
        # never mentions them. Without this the viewer has no sprite for the very
        # things the agent builds, which is all anyone wants to look at.
        for name in header.get("blocks", []):
            wanted.setdefault(name, 0)
    return wanted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument("--jar", type=Path, default=Path("mindustry-bench/assets.jar"))
    parser.add_argument("--out", type=Path, default=Path("viewer/assets"))
    parser.add_argument("--replays", type=Path, default=Path("replays/showcase"))
    args = parser.parse_args()

    if not args.jar.exists():
        args.jar.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {args.version} assets...")
        urllib.request.urlretrieve(ASSETS_URL.format(version=args.version), args.jar)

    replays = sorted(args.replays.glob("*.jsonl.gz"))
    if not replays:
        raise SystemExit(f"no replays in {args.replays}, record one first")

    wanted = wanted_from_replays(replays)
    print(f"{len(wanted)} sprites referenced by {len(replays)} replays")

    result = build(args.jar, wanted, args.out)
    print(f"packed {result['found']} sprites into {result['atlas']}")
    if result["missing"]:
        print(f"not found ({len(result['missing'])}): {', '.join(sorted(result['missing'])[:12])}")


if __name__ == "__main__":
    main()
