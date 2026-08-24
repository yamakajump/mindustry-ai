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


#: Sprites the live dashboard needs that no replay can mention.
#:
#: A replay records what the agent built, so the palette covers the map and the block
#: catalogue and nothing else. The live view also draws every unit on the field, and units
#: are entities rather than tiles: they never appear in a palette, and without them the
#: only thing moving on screen is invisible.
#:
#: Restricted to what actually shows up in the early Serpulo curriculum. The full roster is
#: over eighty units and packing them all would quadruple the atlas to draw four of them.
LIVE_SPRITES = (
    # The player avatars, which is what the agent inhabits.
    "alpha", "beta", "gamma",
    # Ground attackers of the first waves, then what follows if it survives long enough.
    "dagger", "mace", "crawler", "atrax", "nova", "pulsar",
    # Air, which arrives early on most maps and ignores walls.
    "flare", "horizon", "mono", "poly",
)


def sprite_candidates(name: str, variants: int, exact: bool = False) -> list[str]:
    """Every file Mindustry might store this block under, in draw order.

    All of them, not the first that matches. Mindustry picks a variant per tile from a
    seed derived from its position, so a floor with three variants covers the ground
    without a visible weave. Packing one and repeating it is what makes an otherwise
    correct render look like wallpaper.

    Static walls and props ship as `sand-wall1.png` and the like while reporting no
    variant count, because that field only exists on floors. Trusting it lost six sprites.
    """
    if exact:
        # Units have no numbered variants, and a block elsewhere in the jar happening to
        # be called `dagger1` would otherwise be packed in place of the unit.
        return [name]

    names = [f"{name}{i}" for i in range(1, max(variants, 4) + 1)]
    names.append(name)
    if name == "conveyor":
        # Conveyors are variant-frame. Frame zero of each of the four shapes is what a
        # still tile needs; the shapes differ by how neighbours connect.
        names.insert(0, "conveyor-0-0")
    return names


#: Overlays Mindustry draws on top of a block, in the order it draws them.
#:
#: A drill is not one sprite. The base is drawn, then the rotating bit, then a translucent
#: top. Drawing only the base gives a hole where the machine should be.
BLOCK_OVERLAYS = {
    "mechanical-drill": ("mechanical-drill-top",),
}


def edge_name(name: str) -> str:
    """The transition sheet a floor blends onto its neighbours with."""
    return f"{name}-edge"


def large_name(name: str) -> str:
    """The 2x2 sheet a static wall uses to cover a field without repeating itself."""
    return f"{name}-large"


def build(jar: Path, wanted: dict[str, int], out_dir: Path) -> dict:
    """Pack every sprite, every variant and every edge sheet into one atlas."""
    from PIL import Image

    with zipfile.ZipFile(jar) as archive:
        by_stem: dict[str, str] = {}
        for entry in archive.namelist():
            if entry.endswith(".png") and "/sprites/" in entry:
                by_stem.setdefault(Path(entry).stem, entry)

        def load(stem: str) -> "Image.Image | None":
            path = by_stem.get(stem)
            if path is None:
                return None
            with archive.open(path) as handle:
                return Image.open(handle).convert("RGBA")

        found: dict[str, list[Image.Image]] = {}
        missing: list[str] = []
        for name, variants in sorted(wanted.items()):
            exact = name in LIVE_SPRITES
            images = [image for candidate in sprite_candidates(name, variants, exact)
                      if (image := load(candidate)) is not None]
            if images:
                # Variants of one block are the same size, so a stray match of another
                # block that happens to share the prefix is dropped rather than drawn at
                # the wrong scale.
                found[name] = [i for i in images if i.size == images[0].size]
            else:
                missing.append(name)

            for overlay in BLOCK_OVERLAYS.get(name, ()):
                image = load(overlay)
                if image is not None:
                    found[overlay] = [image]

            for extra in (edge_name(name), large_name(name)):
                sheet = load(extra)
                if sheet is not None:
                    found[extra] = [sheet]

    if not found:
        raise SystemExit("no sprites matched, is the jar correct?")

    flat = [(name, image) for name, images in sorted(found.items()) for image in images]
    columns = 16
    rows = (len(flat) + columns - 1) // columns
    cell = max(max(i.width, i.height) for _, i in flat)
    atlas = Image.new("RGBA", (columns * cell, rows * cell), (0, 0, 0, 0))

    index: dict[str, dict] = {}
    for position, (name, image) in enumerate(flat):
        x = (position % columns) * cell
        y = (position // columns) * cell
        atlas.paste(image, (x, y))
        frame = {"x": x, "y": y, "w": image.width, "h": image.height}
        entry = index.get(name)
        if entry is None:
            # `x`/`y`/`w`/`h` stay at the top level so a viewer that only knows about one
            # sprite per block keeps working; `frames` is what a viewer that picks per
            # tile reads.
            index[name] = {**frame, "tiles": max(1, round(image.width / TILE)),
                           "frames": [frame]}
        else:
            entry["frames"].append(frame)

    for name, entry in index.items():
        if name.endswith("-edge") or name.endswith("-large"):
            # A sheet rather than a sprite: 3x3 for the directions a floor blends onto its
            # neighbours, 2x2 for the block a wall field is tiled from.
            entry["grid"] = max(1, round(entry["w"] / TILE))

    out_dir.mkdir(parents=True, exist_ok=True)
    atlas.save(out_dir / "atlas.png", optimize=True)
    (out_dir / "atlas.json").write_text(
        json.dumps({"cell": cell, "tile": TILE, "sprites": index}, indent=1), encoding="utf-8"
    )

    return {"found": len(flat), "names": len(index), "missing": missing,
            "atlas": out_dir / "atlas.png"}


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


def wanted_for_live(blocks: tuple[str, ...]) -> dict[str, int]:
    """Everything the live dashboard draws, replays or not."""
    wanted = {name: 0 for name in LIVE_SPRITES}
    for name in blocks:
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

    from gamma.env import DEFAULT_BLOCKS

    wanted = wanted_for_live(DEFAULT_BLOCKS)
    replays = sorted(args.replays.glob("*.jsonl.gz"))
    for name, variants in wanted_from_replays(replays).items():
        wanted[name] = max(wanted.get(name, 0), variants)
    print(f"{len(wanted)} sprites wanted, {len(replays)} replays scanned")

    result = build(args.jar, wanted, args.out)
    print(f"packed {result['found']} images over {result['names']} names into {result['atlas']}")
    if result["missing"]:
        print(f"not found ({len(result['missing'])}): {', '.join(sorted(result['missing'])[:12])}")


if __name__ == "__main__":
    main()
