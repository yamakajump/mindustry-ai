"""Extract the sprites the viewers need from Mindustry's own asset jar.

The viewers draw the game with the game's art rather than coloured squares, which is the
difference between a debugging aid and something worth showing someone.

Mindustry is GPL-3.0 and its assets live in the same repository under the same licence, so
redistributing them is allowed. They are not committed here all the same: this writes
about three and a half thousand files, and a repository is a poor place to keep a build
artifact that regenerates in twenty seconds.

**The whole catalogue is extracted, not the subset in use.** An index covering only what
the current curriculum touches has to be rebuilt the day the agent unlocks a new block,
which is the day nobody wants to be regenerating assets. Sprites are written one per file
and fetched by a viewer the first time it needs one, so covering everything costs disk and
nothing else: a match draws a few dozen of them.

Output is `viewer/assets/sprites/` plus an `index.json` describing how they group.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import urllib.request
import zipfile
from pathlib import Path

ASSETS_URL = "https://github.com/Anuken/Mindustry/releases/download/{version}/assets.jar"
DEFAULT_VERSION = "v159.7"

#: Sprite size in Mindustry is 32 pixels per world tile.
TILE = 32

#: Sprite folders that hold something a viewer draws.
#:
#: `generated` carries the composed unit icons, which is what a unit is actually drawn
#: from; the parts under `units` are assembled into them. `rubble` is where the damage
#: cracks live, which is not where anyone would look for them.
DRAWN_FOLDERS = ("blocks", "units", "items", "liquids", "rubble", "generated")

#: Suffixes that only ever appear in menus. Packing them would add a third to the output
#: for sprites no viewer draws.
UI_ONLY = re.compile(r"-(preview|icon|ui)$")

#: The composed icon the game generates for a block, used only as a fallback.
#:
#: Some blocks have no region of their own. A salvo is drawn entirely from parts, so
#: `salvo.png` does not exist and a viewer looking for it finds nothing. The generated icon
#: is the whole machine in one image, which is exactly what a top-down viewer wants when it
#: is not going to assemble the parts itself.
COMPOSED = re.compile(r"^block-(?P<name>.+)-full$")

#: Teams whose painted plates are worth keeping. The others exist but never play here.
KEEP_TEAMS = ("sharded", "crux")

#: `conveyor-3-2` is shape three, frame two. `cracks-1-5` is size one, stage five. Same
#: shape, and both are sheets of frames rather than independent sprites.
SHEET = re.compile(r"^(?P<group>.+-\d+)-(?P<frame>\d+)$")

#: `sand-floor2` is the second variant of `sand-floor`. The engine picks one per tile from
#: the tile position, which is what stops a floor tiling into a visible weave.
VARIANT = re.compile(r"^(?P<base>.*[^\d])(?P<index>\d+)$")


def is_drawn(path: str) -> bool:
    """Whether a sprite in the jar is something a viewer will ever put on screen."""
    if not path.endswith(".png") or "/sprites/" not in path:
        return False

    folder = path.split("/sprites/")[1].split("/")[0]
    if folder not in DRAWN_FOLDERS:
        return False

    stem = Path(path).stem
    if UI_ONLY.search(stem):
        return False

    if stem.endswith("-full") and not COMPOSED.match(stem):
        return False

    team = re.search(r"-team-([a-z]+)$", stem)
    return not (team and team.group(1) not in KEEP_TEAMS)


#: `block-1` to `block-4` are the plain plates a turret without a base of its own stands
#: on, one per block size. They are not variants of a `block-` sprite, and grouping them as
#: such makes a one-tile turret pick a four-tile plate.
BASE_PLATE = re.compile(r"^block-\d+$")


def group_of(stem: str) -> tuple[str, int]:
    """The logical sprite a file belongs to, and its position within it.

    Returns the group name and the frame or variant index, zero-based for sheets and
    one-based for variants, matching how the engine numbers each.
    """
    if BASE_PLATE.match(stem):
        return stem, 0

    sheet = SHEET.match(stem)
    if sheet:
        return sheet.group("group"), int(sheet.group("frame"))

    variant = VARIANT.match(stem)
    if variant and variant.group("base"):
        return variant.group("base"), int(variant.group("index"))

    return stem, 0


def extract(jar: Path, out_dir: Path) -> dict:
    """Write every drawn sprite as its own file and describe how they group."""
    from PIL import Image

    sprites = out_dir / "sprites"
    if sprites.exists():
        shutil.rmtree(sprites)
    sprites.mkdir(parents=True)

    index: dict[str, dict] = {}
    written = 0

    with zipfile.ZipFile(jar) as archive:
        for path in sorted(archive.namelist()):
            if not is_drawn(path):
                continue

            stem = Path(path).stem
            composed = COMPOSED.match(stem)
            if composed and composed.group("name") in index:
                # The block has a region of its own, which the engine draws and then adds
                # its moving parts to. The composed icon would freeze those parts.
                continue

            with archive.open(path) as handle:
                image = Image.open(handle).convert("RGBA")
                image.save(sprites / f"{stem}.png", optimize=True)
            written += 1

            group, position = (composed.group("name"), 0) if composed else group_of(stem)
            entry = index.setdefault(group, {
                "w": image.width,
                "h": image.height,
                # How many tiles across the sprite covers, so a viewer never paints a
                # three-by-three core nine times over.
                "tiles": max(1, round(image.width / TILE)),
                "frames": [],
            })
            entry["frames"].append([position, stem])

            # A sheet is one image cut into a grid: three by three for the fades a floor
            # blends onto its neighbours, two by two for the block a rock field is tiled
            # from. The viewer needs the grid to index into it.
            if stem.endswith("-edge") or stem.endswith("-large"):
                entry["grid"] = max(1, round(image.width / TILE))

    for entry in index.values():
        # Ordered, and reduced to plain names: the position mattered only for sorting.
        entry["frames"] = [name for _, name in sorted(entry["frames"])]

    (out_dir / "index.json").write_text(
        json.dumps({"tile": TILE, "sprites": index}, separators=(",", ":")),
        encoding="utf-8",
    )

    return {"files": written, "groups": len(index), "dir": sprites}


def ensure_assets(
    out: Path = Path("viewer/assets"),
    jar: Path = Path("mindustry-bench/assets.jar"),
    version: str = DEFAULT_VERSION,
) -> None:
    """Download the asset jar and extract the sprites, unless they are already there.

    Called from the training entry point, because a fresh clone that starts a run and gets
    a dashboard of grey squares has no way of guessing what it is missing.
    """
    if (out / "index.json").exists():
        return

    if not jar.exists():
        jar.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {version} assets (first run only)...", flush=True)
        urllib.request.urlretrieve(ASSETS_URL.format(version=version), jar)

    print("extracting sprites (first run only)...", flush=True)
    result = extract(jar, out)
    print(f"{result['files']} sprites in {result['groups']} groups", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument("--jar", type=Path, default=Path("mindustry-bench/assets.jar"))
    parser.add_argument("--out", type=Path, default=Path("viewer/assets"))
    args = parser.parse_args()

    if not args.jar.exists():
        args.jar.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {args.version} assets...")
        urllib.request.urlretrieve(ASSETS_URL.format(version=args.version), args.jar)

    result = extract(args.jar, args.out)
    print(f"wrote {result['files']} sprites in {result['groups']} groups to {result['dir']}")


if __name__ == "__main__":
    main()
