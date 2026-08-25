"""What did the agent ask to build, and could it ever have formed a chain?

CAVEAT, and it matters: this reads the `place` actions recorded in a replay and
reconstructs a layout from them. That is what the agent REQUESTED, not what stood in the
world. An embodied agent flies to a site and builds it, so requests are refused,
interrupted, or demolished later, and one measured episode issued 559 placements against
439 demolitions. Everything below is an upper bound on what existed, with no notion of
two blocks existing at the same moment.

Do not use it to conclude that nothing was ever connected. It was read that way once and
the conclusion was wrong: an episode it scored at zero chains was being paid for ore
arriving through a transport block on 87% of its steps. For where the points came from,
read the per-frame `terms` the recorder writes, which is measured rather than inferred.

Original question, still worth asking of the requests themselves:

    python tools/analyse_replays.py

The training run reports that some episodes automated production. That number says ore
reached the core through a transport block; it does not say the agent built a line to get
it there. A drill placed touching the core delivers straight into it, because
`Drill.offload` pushes to any adjacent building and the core counts as one. So a run can
report automation without a single conveyor ever having been useful.

This reads the archived episodes and separates the two. For every drill the agent placed,
it looks for a conveyor placed next to it pointing away from it, and follows the line to
see where it ends. What comes out is the number that matters: how often, in hundreds of
episodes, a deliberate chain was ever completed.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import Counter
from pathlib import Path

#: Mindustry rotations: 0 right, 1 up, 2 left, 3 down.
STEP = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}

DRILLS = ("mechanical-drill", "pneumatic-drill", "laser-drill", "airblast-drill")
CARRIERS = ("conveyor", "titanium-conveyor", "plastanium-conveyor", "armored-conveyor",
            "junction", "router", "distributor", "bridge-conveyor", "phase-conveyor",
            "overflow-gate", "underflow-gate", "sorter", "inverted-sorter")

SCORED = re.compile(r"^ep(\d+)-(neg|pos)(\d+)\.jsonl\.gz$")


def read(path: Path) -> tuple[dict, list[dict]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle]
    return records[0], [r for r in records if r.get("type") == "frame"]


def touches(x: int, y: int, size: int, cx: int, cy: int, core: int) -> bool:
    """Whether a block at (x,y) of the given size is adjacent to the core.

    Mindustry anchors a block on its origin tile and grows up and right, and a core is
    three tiles across for a shard. Touching means the two footprints are one tile apart
    or less on both axes.
    """
    half = core // 2
    return (abs(x - cx) <= half + size and abs(y - cy) <= half + size)


def follow(start: tuple[int, int], carriers: dict, limit: int = 400) -> tuple[int, int] | None:
    """Walk a line from its first tile to wherever it stops, honouring rotations.

    This is the strict reading and it is a lower bound. A router has no meaningful
    rotation and sends items every way at once, and a junction passes them straight
    through in the direction they arrived, so a walk that treats either as a one-way
    conveyor stops early. The walk therefore ends at the first of those it meets, and the
    caller decides whether where it stopped is good enough.
    """
    x, y = start
    seen = set()
    for _ in range(limit):
        if (x, y) in seen or (x, y) not in carriers:
            return (x, y)
        seen.add((x, y))
        block, rotation = carriers[(x, y)]
        if block in ("router", "distributor", "overflow-gate", "underflow-gate", "sorter",
                     "inverted-sorter", "junction"):
            return (x, y)
        dx, dy = STEP.get(rotation, (0, 0))
        if (dx, dy) == (0, 0):
            return (x, y)
        x, y = x + dx, y + dy
    return (x, y)


def reaches(start: tuple[int, int], carriers: dict, cx: int, cy: int) -> bool:
    """Whether any unbroken run of carriers connects the start to the core.

    Rotation is ignored on purpose. This is the generous reading and it is an upper
    bound: it answers "were the tiles even laid next to each other", which is a necessary
    condition for a chain and nowhere near a sufficient one. Bracketing the truth between
    this and the strict walk beats trusting either alone.
    """
    stack = [start]
    seen = set()
    while stack:
        tile = stack.pop()
        if tile in seen or tile not in carriers:
            continue
        seen.add(tile)
        if touches(tile[0], tile[1], 1, cx, cy, 3):
            return True
        x, y = tile
        stack.extend([(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)])
    return False


def study(path: Path) -> dict:
    header, frames = read(path)
    core = header.get("core") or [-1, -1]
    cx, cy = int(core[0]), int(core[1])

    drills: list[tuple[int, int, str]] = []
    carriers: dict[tuple[int, int], int] = {}
    placed = Counter()

    for frame in frames:
        action = frame.get("act")
        if action is None or action.get("t") != "place":
            continue
        block, x, y = action["b"], int(action["x"]), int(action["y"])
        placed[block] += 1
        if block in DRILLS:
            drills.append((x, y, block))
        elif block in CARRIERS:
            carriers[(x, y)] = (block, int(action.get("r", 0)))

    # A drill touching the core needs nothing else: it offloads straight into it.
    adjacent = sum(1 for x, y, block in drills
                   if touches(x, y, 2 if block == "mechanical-drill" else 2, cx, cy, 3))

    # Two readings of "the agent built a chain", one strict and one generous.
    chains = 0
    touching = 0
    for x, y, _ in drills:
        neighbours = [(x + dx, y + dy) for dx, dy in STEP.values()
                      if (x + dx, y + dy) in carriers]
        if not neighbours:
            continue
        if any(reaches(n, carriers, cx, cy) for n in neighbours):
            touching += 1
        for neighbour in neighbours:
            end = follow(neighbour, carriers)
            if end is not None and touches(end[0], end[1], 1, cx, cy, 3):
                chains += 1
                break

    return {
        "file": path,
        "reward": (lambda m: int(m.group(3)) / 100 * (-1 if m.group(2) == "neg" else 1))(
            SCORED.match(path.name)) if SCORED.match(path.name) else 0.0,
        "steps": len(frames),
        "drills": len(drills),
        "carriers": len(carriers),
        "adjacent": adjacent,
        "chains": chains,
        "touching": touching,
        "placed": placed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, nargs="?", default=Path("replays/live"))
    args = parser.parse_args()

    episodes = sorted(args.root.rglob("ep*-*.jsonl.gz"))
    if not episodes:
        raise SystemExit(f"no archived episode under {args.root}")

    # Episodes are pruned by the recorder while training runs, so a path listed a moment
    # ago can be gone by the time it is opened. Reading a live archive has to tolerate
    # that; the alternative is an analysis that only works once training has stopped.
    studied = []
    vanished = 0
    for path in episodes:
        try:
            studied.append(study(path))
        except (FileNotFoundError, EOFError, gzip.BadGzipFile):
            vanished += 1
    if vanished:
        print(f"{vanished} episode(s) pruned or half-written while reading, skipped")
    if not studied:
        raise SystemExit(f"no readable episode under {args.root}")
    blocks: Counter = Counter()
    for episode in studied:
        blocks.update(episode["placed"])

    with_chain = [e for e in studied if e["chains"]]
    with_touch = [e for e in studied if e["touching"]]
    with_adjacent = [e for e in studied if e["adjacent"]]

    print(f"{len(studied)} archived episodes under {args.root}")
    print()
    print(f"  placed a drill at all           {sum(1 for e in studied if e['drills']):4d}"
          f"  ({sum(e['drills'] for e in studied):,} drills)")
    print(f"  a drill touching the core       {len(with_adjacent):4d}"
          f"   <- delivers with no conveyor at all")
    print(f"  carriers merely reaching it     {len(with_touch):4d}"
          f"   <- tiles laid end to end, rotations ignored")
    print(f"  a conveyor line drill -> core   {len(with_chain):4d}"
          f"   <- an actual chain, rotations correct")
    print()

    top = [f"{name} {count:,}" for name, count in blocks.most_common(8)]
    print("  most placed blocks: " + ", ".join(top))
    print()

    best = sorted(studied, key=lambda e: -e["reward"])[:5]
    print("  the five best-scoring episodes:")
    for episode in best:
        print(f"    {episode['reward']:+8.1f}  {episode['drills']:3d} drills, "
              f"{episode['carriers']:4d} carriers, {episode['adjacent']} touching the core, "
              f"{episode['touching']} reaching, {episode['chains']} chains")
    print()

    if not with_chain:
        print("VERDICT: not one chain among the placements the agent REQUESTED.")
        print()
        print("         Read that literally, because this tool reads intentions and not")
        print("         the world. It reconstructs a layout from the `place` actions in")
        print("         the replay, and an embodied agent has to fly to a site and build")
        print("         it, so a request can be refused, interrupted, or demolished later:")
        print("         one episode here issued 559 placements and 439 demolitions. The")
        print("         layout below is therefore an upper bound on what ever existed and")
        print("         says nothing about what existed at the same time.")
        print()
        print("         It also cannot explain delivery. An episode with zero chains by")
        print("         this measure was paid for ore arriving through a transport block on")
        print("         87% of its steps, and hand mining was ruled out against a live")
        print("         server: mined by hand, core stock went 200 -> 503 while the")
        print("         delivery counter stayed at zero. Something was connected that this")
        print("         reconstruction does not see.")
        print()
        print("         For where an episode's points actually came from, read the `terms`")
        print("         field the recorder now writes on every frame. That is measured, not")
        print("         reconstructed.")
    else:
        share = len(with_chain) / len(studied)
        print(f"VERDICT: {len(with_chain)} episodes built a real chain, {share:.1%} of them.")


if __name__ == "__main__":
    main()
