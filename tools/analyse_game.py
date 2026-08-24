"""What the agent is actually being asked to discover.

    python tools/analyse_game.py

A drill on ore does nothing. A drill with no conveyor to the core does nothing. A smelter
without power, or without both of its inputs arriving, does nothing. Mindustry pays for
*connected chains*, and every intermediate state on the way to one is worth exactly zero.

This reads the game's own content definitions and reports the shape of that problem: what
each factory consumes and produces, how deep the chains run, and what a policy placing
blocks at random is up against. It is a measurement, not an opinion, and it is the thing
that decides whether flat reinforcement learning can find any of this on its own.
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

#: Where the extracted game sources live. The jar ships them alongside the classes.
SOURCES = Path(r"C:\Users\coren\AppData\Local\Temp\claude\mdsrc")

BLOCK = re.compile(r"^\s{8}(\w+) = new (\w+)\(\"([\w-]+)\"\)\{\{", re.MULTILINE)
OUTPUT_ITEM = re.compile(r"outputItem = new ItemStack\(Items\.(\w+), (\d+)\)")
OUTPUT_ITEMS = re.compile(r"outputItems = with\(([^)]*)\)")
CONSUME_ITEM = re.compile(r"consumeItem\(Items\.(\w+)(?:, (\d+))?\)")
CONSUME_ITEMS = re.compile(r"consumeItems\(with\(([^)]*)\)\)")
CONSUME_POWER = re.compile(r"consumePower\(([\d.]+)f?\)")
CONSUME_LIQUID = re.compile(r"consumeLiquid\(Liquids\.(\w+)")
REQUIREMENTS = re.compile(r"requirements\(Category\.(\w+), (?:.*?)with\(([^)]*)\)")
STACK = re.compile(r"Items\.(\w+),\s*(\d+)")


def blocks(text: str) -> list[dict]:
    """Every block definition, with the body that follows it."""
    found = []
    matches = list(BLOCK.finditer(text))
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        found.append({
            "field": match.group(1),
            "kind": match.group(2),
            "name": match.group(3),
            "body": text[match.end():end],
        })
    return found


def parse(block: dict) -> dict:
    body = block["body"]

    requirements = {}
    category = ""
    if (match := REQUIREMENTS.search(body)) is not None:
        category = match.group(1)
        requirements = {item: int(amount) for item, amount in STACK.findall(match.group(2))}

    consumes = {}
    for item, amount in CONSUME_ITEM.findall(body):
        consumes[item] = int(amount or 1)
    if (match := CONSUME_ITEMS.search(body)) is not None:
        for item, amount in STACK.findall(match.group(1)):
            consumes[item] = int(amount)

    produces = {}
    if (match := OUTPUT_ITEM.search(body)) is not None:
        produces[match.group(1)] = int(match.group(2))
    if (match := OUTPUT_ITEMS.search(body)) is not None:
        for item, amount in STACK.findall(match.group(1)):
            produces[item] = int(amount)

    return {
        **block,
        "category": category,
        "requirements": requirements,
        "consumes": consumes,
        "produces": produces,
        "power": float(match.group(1)) if (match := CONSUME_POWER.search(body)) else 0.0,
        "liquid": match.group(1) if (match := CONSUME_LIQUID.search(body)) else "",
    }


def depth_of(item: str, recipes: dict[str, list[dict]], seen: frozenset = frozenset()) -> int:
    """How many crafting steps separate an item from what a drill can dig up.

    Raw ores are depth zero. Anything else is one more than the deepest of its cheapest
    recipe's inputs, which is the number of separate working production lines that have
    to exist at once before a single unit of it appears.
    """
    if item not in recipes or item in seen:
        return 0
    best = None
    for recipe in recipes[item]:
        inputs = recipe["consumes"]
        cost = 1 + max((depth_of(i, recipes, seen | {item}) for i in inputs), default=0)
        best = cost if best is None else min(best, cost)
    return best or 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=SOURCES)
    args = parser.parse_args()

    content = args.sources / "mindustry" / "content" / "Blocks.java"
    if not content.is_file():
        raise SystemExit(f"no game sources at {content}. Extract them from the Mindustry jar.")

    parsed = [parse(block) for block in blocks(content.read_text(encoding="utf-8"))]

    recipes: dict[str, list[dict]] = defaultdict(list)
    for block in parsed:
        for item in block["produces"]:
            if block["consumes"] or block["power"]:
                recipes[item].append(block)

    factories = [b for b in parsed if b["produces"] and b["consumes"]]
    drills = [b for b in parsed if b["kind"] in ("Drill", "BurstDrill") or "drill" in b["name"]]

    print("=" * 74)
    print("  WHAT THE AGENT HAS TO DISCOVER")
    print("=" * 74)
    print()
    print(f"{len(parsed)} blocks, {len(factories)} of them factories that turn items into "
          f"other items,")
    print(f"{len(drills)} drills, {len(recipes)} items that only exist if something makes them.")
    print()

    print("-" * 74)
    print("  PRODUCTION CHAINS, BY HOW DEEP THEY RUN")
    print("-" * 74)
    print()
    depths: dict[int, list[str]] = defaultdict(list)
    for item in recipes:
        depths[depth_of(item, recipes)].append(item)

    for depth in sorted(depths):
        print(f"depth {depth}: {', '.join(sorted(depths[depth]))}")
    print()
    print("Depth is the number of separate production lines that must all be running at")
    print("the same time before one unit of the item exists. Nothing partial pays.")
    print()

    print("-" * 74)
    print("  THE FIRST FEW STEPS, IN FULL")
    print("-" * 74)
    print()
    early = ("graphite-press", "silicon-smelter", "kiln", "combustion-generator",
             "pneumatic-drill", "mechanical-drill", "duo", "conveyor")
    for name in early:
        block = next((b for b in parsed if b["name"] == name), None)
        if block is None:
            continue
        costs = ", ".join(f"{amount} {item}" for item, amount in block["requirements"].items())
        needs = ", ".join(f"{amount} {item}" for item, amount in block["consumes"].items())
        makes = ", ".join(f"{amount} {item}" for item, amount in block["produces"].items())
        power = f", {block['power']} power" if block["power"] else ""
        liquid = f", {block['liquid']}" if block["liquid"] else ""
        print(f"  {name:22s} costs {costs}")
        if needs or makes:
            print(f"  {'':22s} eats  {needs or 'nothing'}{power}{liquid}")
            print(f"  {'':22s} makes {makes or 'nothing'}")
        print()

    print("-" * 74)
    print("  WHAT RANDOM PLACEMENT IS UP AGAINST")
    print("-" * 74)
    print()
    window = 48
    tiles = window * window
    print(f"The agent picks a tile in a {window}x{window} window, so {tiles:,} choices.")
    print()
    print("A drill only produces on ore. On a generated sector maybe 3% of tiles carry ore,")
    print(f"so roughly {round(tiles * 0.03)} of those {tiles:,} tiles are worth a drill:")
    print("  one placement in 33 lands on ore at all.")
    print()
    print("That drill then has to reach the core. A conveyor line is a contiguous path of")
    print("correctly rotated tiles, and each conveyor has 4 rotations:")
    for length in (3, 5, 10):
        chance = (1 / tiles) ** 0 * (1 / 4) ** length
        print(f"  a {length:2d}-tile line, if every tile is placed in the right square: "
              f"1 in {1 / chance:,.0f} on rotations alone")
    print()
    print("And that is the *first* rung. A graphite press needs a working coal line and a")
    print("working copper line at once, before it produces anything at all.")
    print()
    print("=" * 74)


if __name__ == "__main__":
    main()
