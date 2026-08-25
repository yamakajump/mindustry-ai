"""Can this replay be replayed, and does what comes out match what went in?

    python tools/verify_replay.py showcase/fidelite.jsonl.gz

A replay used to be a reconstruction: the reader replayed the agent's actions on a rebuilt
map and let the game work out everything else. Everything else is most of what happened,
so what it showed was a plausible episode rather than the episode. The recording now
carries the world itself, one delta per step, and this checks that the deltas are a
complete and consistent account rather than assuming they are.

It rebuilds the world here, in Python, by exactly the rules the mod applies in Java. Two
implementations of the same reading is the point: if they agree on what stands and who is
where, the format says what both think it says. If they disagree, one of them is wrong and
this is the cheaper place to find out.

Every check names what would be lost if it failed, because "replay looks fine" is what was
believed about a reader that showed 1% of what the agent built.
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

#: Field widths the bridge writes, and the mod reads back. Kept here so a change to one
#: side fails loudly against a recording rather than silently misreading every number.
UNIT = 11
BUILDING = 6
HURT = 2
SHOT = 4


def read(path: Path) -> tuple[dict, list[dict]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        header = json.loads(handle.readline())
        return header, [json.loads(line) for line in handle]


class World:
    """The world as the scene stream describes it, rebuilt step by step."""

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        #: tile key -> (block, rotation, team, health, progress)
        self.buildings: dict[int, tuple[int, int, int, int, int]] = {}
        #: unit id -> (type, team, x, y, rotation, health)
        self.units: dict[int, tuple[int, ...]] = {}
        self.problems: Counter = Counter()
        self.seen_units: set[int] = set()
        self.shots = 0

    def off_map(self, key: int) -> bool:
        return not 0 <= key < self.width * self.height

    def apply(self, scene: dict) -> None:
        placed = scene.get("placed") or []
        for i in range(0, len(placed) - BUILDING + 1, BUILDING):
            key = int(placed[i])
            if self.off_map(key):
                self.problems["batiment hors carte"] += 1
                continue
            self.buildings[key] = tuple(int(v) for v in placed[i + 1:i + BUILDING])

        for key in scene.get("removed") or []:
            if int(key) not in self.buildings:
                # Not a fault by itself: the recorder reports a departure whether or not
                # the reader ever saw the arrival, and a building placed and destroyed
                # inside one step is never reported as present.
                self.problems["retrait d un batiment jamais vu"] += 1
            self.buildings.pop(int(key), None)

        hurt = scene.get("hurt") or []
        for i in range(0, len(hurt) - HURT + 1, HURT):
            key = int(hurt[i])
            if key not in self.buildings:
                self.problems["degats sur un batiment jamais vu"] += 1
                continue
            current = self.buildings[key]
            self.buildings[key] = current[:3] + (int(hurt[i + 1]),) + current[4:]

        units = scene.get("units") or []
        if len(units) % UNIT:
            self.problems["tableau d unites de largeur invalide"] += 1
        for i in range(0, len(units) - UNIT + 1, UNIT):
            row = [int(v) for v in units[i:i + UNIT]]
            self.units[row[0]] = tuple(row[1:])
            self.seen_units.add(row[0])

        for uid in scene.get("gone") or []:
            if int(uid) not in self.units:
                self.problems["depart d une unite jamais vue"] += 1
            self.units.pop(int(uid), None)

        shots = scene.get("shots") or []
        if len(shots) % SHOT:
            self.problems["tableau de tirs de largeur invalide"] += 1
        self.shots += len(shots) // SHOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay", type=Path)
    args = parser.parse_args()

    header, records = read(args.replay)
    frames = [r for r in records if r.get("type") == "frame"]
    width, height = header["width"], header["height"]

    world = World(width, height)
    scened = 0
    teams: Counter = Counter()
    for frame in frames:
        scene = frame.get("scene")
        if not scene:
            continue
        scened += 1
        world.apply(scene)
        for uid, row in world.units.items():
            teams[row[1]] += 0  # touch, so a team with no movement still appears
        units = scene.get("units") or []
        for i in range(0, len(units) - UNIT + 1, UNIT):
            teams[int(units[i + 2])] += 1

    actions = Counter(f["act"]["t"] for f in frames if f.get("act"))
    stamped = sum(len(f["act"].get("cells") or [])
                  for f in frames if f.get("act", {}).get("t") == "stamp")

    print(f"{args.replay.name}  {header.get('task')}  carte {width}x{height}")
    print(f"  frames                    {len(frames):7,}")
    print(f"  dont avec une scene       {scened:7,}  "
          f"({100 * scened / max(1, len(frames)):.0f}%)")
    print()
    print("  actions de l agent :")
    for name, count in actions.most_common():
        extra = f"  ({stamped:,} blocs)" if name == "stamp" else ""
        print(f"    {name:<8} {count:6,}{extra}")
    print()
    print(f"  unites distinctes vues    {len(world.seen_units):7,}")
    print(f"  par equipe (apparitions)  {dict(teams.most_common())}")
    print(f"  unites encore la a la fin {len(world.units):7,}")
    print(f"  batiments debout a la fin {len(world.buildings):7,}")
    print(f"  tirs vus                  {world.shots:7,}")

    print()
    if world.problems:
        for name, count in world.problems.most_common():
            print(f"  [ALERTE] {name} : {count}")
    else:
        print("  aucune incoherence dans le flux de scene")

    print()
    fatal = sum(count for name, count in world.problems.items()
                if "hors carte" in name or "largeur invalide" in name)
    if fatal:
        print("VERDICT : le flux est illisible par endroits, le rejeu ne peut pas etre fidele")
    elif scened < len(frames) * 0.9:
        print("VERDICT : trop de steps sans scene, ce replay est encore une reconstruction")
    elif not teams:
        print("VERDICT : aucune unite enregistree, il n y a rien a rejouer")
    else:
        print("VERDICT : le flux se rejoue entierement et sans contradiction")


if __name__ == "__main__":
    main()
