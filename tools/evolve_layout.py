"""Let the agent invent its own production layout, and judge it by what it delivers.

    python tools/evolve_layout.py

Every candidate is stamped into a real Mindustry world and run for a few seconds, and its
score is the ore that actually reached the core. Nothing tells it what a conveyor is for,
which way a drill faces, or that the two go together. The rules of the game are the whole
of the fitness function, so whatever survives is the agent's own design rather than a
blueprint copied from someone who already knew the answer.

This exists because the policy cannot find these structures by choosing tiles one at a
time. Measured over 177 archived episodes of a real training run: 4,481 drills placed,
5,719 conveyors placed, and exactly one episode in which the tiles ever met end to end.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from gamma.bridge import Bridge
from gamma.evolve import PALETTE, Layout, Population, fitness
from gamma.server import ServerProcess, install_plugin
from gamma.server_setup import setup_server

BRIDGE_PORT = 7995
GAME_PORT = 6595


def best_patch(spatial: np.ndarray, channels: list[str], core: tuple[int, int],
               width: int, height: int, ore: str = "ore_copper") -> tuple[int, int]:
    """Where to put the test rectangle: touching the core, over as much ore as possible.

    It has to touch the core, because a layout is judged on what reaches the core and
    nothing inside the rectangle can deliver anywhere outside it. Among the placements
    that do, the one covering the most ore is the one where a working design exists at
    all.
    """
    plane = spatial[channels.index(ore)]
    rows, columns = plane.shape
    cx, cy = core

    # Every position the rectangle can take while still covering the core, scored on the
    # ore it captures. Exhaustive because it is cheap and because guessing an anchor is
    # how the first run of this measured a rectangle with one tile of copper in it and
    # then reported, correctly and uselessly, that nothing delivered anything.
    best, where = -1, (cx, cy)
    for y in range(max(0, cy - height + 2), min(rows - height, cy - 1) + 1):
        for x in range(max(0, cx - width + 2), min(columns - width, cx - 1) + 1):
            count = int((plane[y:y + height, x:x + width] > 0).sum())
            if count > best:
                best, where = count, (x, y)
    return where


def clear(bridge: Bridge, origin: tuple[int, int], width: int, height: int,
          core: tuple[int, int]) -> None:
    """Empty the test rectangle, sparing the core.

    Demolishing the core would end the run and take the measurement with it, so the three
    tiles around its centre are left alone. Everything else goes, refusals included: a
    tile that was already empty refuses and that is the expected answer, not a fault.
    """
    cx, cy = core
    for y in range(origin[1], origin[1] + height):
        for x in range(origin[0], origin[0] + width):
            if abs(x - cx) <= 2 and abs(y - cy) <= 2:
                continue
            bridge.act({"type": "break", "x": x, "y": y})


def stamp(bridge: Bridge, layout: Layout, origin: tuple[int, int],
          core: tuple[int, int]) -> int:
    """Place a layout, and report how many of its blocks the engine accepted.

    Refusals are ordinary. A drill needs two clear tiles by two, a cell may sit on the
    core, and a candidate drawn at random asks for plenty that cannot exist. The count of
    what stood is what the cost term is charged on, so a design that asks for the
    impossible is not billed for it.
    """
    placed = 0
    cx, cy = core
    for x, y, block, rotation in layout.cells():
        tx, ty = origin[0] + x, origin[1] + y
        if abs(tx - cx) <= 2 and abs(ty - cy) <= 2:
            continue
        outcome = bridge.act({
            "type": "place", "block": block, "x": tx, "y": ty, "rotation": rotation,
        })["action"]
        placed += bool(outcome.get("applied"))
    return placed


def delivered(observation: dict) -> int:
    return sum(int(amount) for amount in observation.get("produced", {}).values())


def evaluate(bridge: Bridge, layout: Layout, origin: tuple[int, int],
             core: tuple[int, int], ticks: int) -> None:
    """Run one candidate and record what it delivered.

    The counter is the engine's own, and it counts only what arrives through a transport
    block. A drill dropped against the core delivers without any line at all and is
    counted here exactly as the game counts it, which is correct: if that is the best
    design, the search should be allowed to find it.
    """
    clear(bridge, origin, layout.width, layout.height, core)
    placed = stamp(bridge, layout, origin, core)

    before = delivered(bridge.observe())
    after = delivered(bridge.step(repeat=ticks))

    layout.delivered = max(0, after - before)
    layout.cost = placed


def render(layout: Layout) -> str:
    """The layout as text, so a good one can be read rather than guessed at."""
    glyphs = {"air": ".", "conveyor": "><^v", "mechanical-drill": "D",
              "junction": "+", "router": "o"}
    rows = []
    for y in range(layout.height - 1, -1, -1):
        row = ""
        for x in range(layout.width):
            index = y * layout.width + x
            name = PALETTE[layout.blocks[index]]
            mark = glyphs.get(name, "?")
            row += mark[layout.rotations[index]] if len(mark) == 4 else mark
        rows.append(row)
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", default="Ancient_Caldera")
    parser.add_argument("--width", type=int, default=13,
                        help="wide enough to span from the core to the nearest ore; on "
                             "Ancient_Caldera that is five tiles away and there are only "
                             "fifteen within ten")
    parser.add_argument("--height", type=int, default=13)
    parser.add_argument("--population", type=int, default=40)
    parser.add_argument("--generations", type=int, default=25)
    parser.add_argument("--ticks", type=int, default=900,
                        help="game ticks each candidate is given to deliver something")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("docs/measurements/best-layout.json"))
    parser.add_argument("--jar", type=Path, default=None)
    args = parser.parse_args()

    jar = args.jar or next((Path("bridge") / "build" / "libs").glob("*.jar"))
    server_dir = setup_server("mindustry-evolve")
    install_plugin(server_dir, jar)

    with ServerProcess(server_dir, jvm_args=[f"-Dmindustryai.port={BRIDGE_PORT}"],
                       port=GAME_PORT) as server:
        server.wait_for(rf"listening on 127\.0\.0\.1:{BRIDGE_PORT}", timeout=120)

        with Bridge(port=BRIDGE_PORT, tensor=True, timeout=120.0) as bridge:
            # Sandbox, so a candidate is never refused for being unaffordable. What is
            # being searched for is a shape that works, and making the search pay for
            # copper would only teach it to be small.
            observation = bridge.reset(args.map, "sandbox")
            server.command("bridge-speed max", r"speed set")

            core = (int(observation["core_x"]), int(observation["core_y"]))
            origin = best_patch(observation["spatial"], bridge.channels, core,
                                args.width, args.height)
            ore = int((observation["spatial"][bridge.channels.index("ore_copper")]
                       [origin[1]:origin[1] + args.height,
                        origin[0]:origin[0] + args.width] > 0).sum())

            print(f"map {args.map}, core at {core}")
            print(f"test rectangle {args.width}x{args.height} at {origin}, "
                  f"{ore} tiles of copper inside")
            print(f"{args.population} layouts per generation, {args.generations} "
                  f"generations, {args.ticks} ticks each")
            print()
            if ore == 0:
                raise SystemExit("no copper in the rectangle: nothing here can deliver "
                                 "anything, and the search would be measuring noise")

            population = Population(args.width, args.height, size=args.population,
                                    rng=__import__("random").Random(args.seed))
            population.seed()

            started = time.time()
            history = []

            for generation in range(1, args.generations + 1):
                for layout in population.members:
                    if layout.delivered is None:
                        evaluate(bridge, layout, origin, core, args.ticks)

                best = population.best()
                scores = [fitness(layout) for layout in population.members]
                working = sum(1 for layout in population.members if (layout.delivered or 0) > 0)
                history.append({
                    "generation": generation,
                    "best_delivered": best.delivered if best else 0,
                    "best_cost": best.cost if best else 0,
                    "working": working,
                    "mean_fitness": round(sum(scores) / len(scores), 2),
                })
                print(f"generation {generation:3d}  best {best.delivered if best else 0:4d} ore "
                      f"with {best.used() if best else 0:3d} blocks  "
                      f"{working:3d}/{len(population.members)} deliver anything  "
                      f"mean {history[-1]['mean_fitness']:8.2f}", flush=True)

                if generation < args.generations:
                    population.advance()

            best = population.best()
            print()
            if best is None or not best.delivered:
                print("nothing delivered anything. Either the rectangle has no workable")
                print("design in it, or the budget of ticks is too short to see one.")
            else:
                print(f"best layout: {best.delivered} ore in {args.ticks} ticks, "
                      f"{best.used()} blocks")
                print()
                print(render(best))
                print()
                print("  . empty   D drill   > < ^ v conveyor   + junction   o router")

            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps({
                "map": args.map, "rectangle": [args.width, args.height],
                "origin": list(origin), "core": list(core), "ore_tiles": ore,
                "ticks": args.ticks, "seconds": round(time.time() - started, 1),
                "history": history,
                "best": None if best is None else {
                    "delivered": best.delivered, "blocks": best.used(),
                    "layout": render(best),
                    "cells": [[x, y, block, rotation] for x, y, block, rotation in best.cells()],
                },
            }, indent=2), encoding="utf-8")
            print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
