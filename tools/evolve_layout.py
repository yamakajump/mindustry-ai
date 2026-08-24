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
from gamma.evolve import (
    PALETTE,
    DesignPopulation,
    Layout,
    Population,
    fitness,
)
from gamma.server import ServerProcess, install_plugin
from gamma.server_setup import setup_server

BRIDGE_PORT = 7995
GAME_PORT = 6595


def reachable_ore(spatial: np.ndarray, channels: list[str], core: tuple[int, int],
                  ore: str = "ore_copper", keep_out: int = 0) -> np.ndarray:
    """The ore plane, with everything too close to the core erased.

    A drill touching the core hands its ore straight over: `Drill.offload` pushes to any
    adjacent building and a core is one. So a bench with ore against the core has a
    trivial best answer, and the search finds it. Measured: given such a bench the winning
    design was eight drills and **not one conveyor**, which is correct, optimal, and says
    nothing whatever about whether a line can be discovered.

    Blanking the ore within `keep_out` tiles of the core makes a line the only way to
    deliver anything, which is the question actually worth asking.
    """
    plane = spatial[channels.index(ore)].copy()
    if keep_out <= 0:
        return plane

    cx, cy = core
    rows, columns = plane.shape
    ys, xs = np.ogrid[:rows, :columns]
    plane[np.maximum(np.abs(xs - cx), np.abs(ys - cy)) <= keep_out] = 0
    return plane


def best_patch(spatial: np.ndarray, channels: list[str], core: tuple[int, int],
               width: int, height: int, ore: str = "ore_copper",
               keep_out: int = 0) -> tuple[int, int]:
    """Where to put the test rectangle: covering the core, over as much ore as possible.

    It has to cover the core, because a layout is judged on what reaches the core and
    nothing inside the rectangle can deliver anywhere outside it. Among the placements
    that do, the one covering the most ore is the one where a working design exists.

    Exhaustive because it is cheap, and because guessing an anchor is how the first run of
    this measured a rectangle with one tile of copper in it and then reported, correctly
    and uselessly, that nothing delivered anything.
    """
    plane = reachable_ore(spatial, channels, core, ore, keep_out)
    rows, columns = plane.shape
    cx, cy = core

    best, where = -1, (cx, cy)
    for y in range(max(0, cy - height + 2), min(rows - height, cy - 1) + 1):
        for x in range(max(0, cx - width + 2), min(columns - width, cx - 1) + 1):
            count = int((plane[y:y + height, x:x + width] > 0).sum())
            if count > best:
                best, where = count, (x, y)
    return where


def core_footprint(spatial: np.ndarray, channels: list[str],
                   core: tuple[int, int], reach: int = 4) -> set[tuple[int, int]]:
    """Exactly the tiles the core stands on, read off the map rather than assumed.

    Guessing this is how the bench spent a whole afternoon being unwinnable. Sparing
    everything within two tiles of the core's centre protects a five-by-five square, and a
    core shard is three across, so the ring immediately around it was never cleared and
    never built on. The last tile of every line, the one that has to touch the core, was
    silently skipped, and no design could deliver anything however right it was.
    """
    plane = spatial[channels.index("block_ally")]
    cx, cy = core
    rows, columns = plane.shape
    return {
        (x, y)
        for y in range(max(0, cy - reach), min(rows, cy + reach + 1))
        for x in range(max(0, cx - reach), min(columns, cx + reach + 1))
        if plane[y, x] > 0
    }


def clear(bridge: Bridge, origin: tuple[int, int], width: int, height: int,
          spared: set[tuple[int, int]]) -> None:
    """Empty the test rectangle, sparing the core.

    Demolishing the core would end the run and take the measurement with it. Everything
    else goes, refusals included: a tile that was already empty refuses, and that is the
    expected answer rather than a fault.
    """
    for y in range(origin[1], origin[1] + height):
        for x in range(origin[0], origin[0] + width):
            if (x, y) in spared:
                continue
            bridge.act({"type": "break", "x": x, "y": y})


def stamp(bridge: Bridge, layout, origin: tuple[int, int],
          spared: set[tuple[int, int]]) -> int:
    """Place a layout, and report how many of its blocks the engine accepted.

    Refusals are ordinary. A drill needs two clear tiles by two, a cell may sit on the
    core, and a candidate drawn at random asks for plenty that cannot exist. The count of
    what stood is what the cost term is charged on, so a design that asks for the
    impossible is not billed for it.
    """
    placed = 0
    for x, y, block, rotation in layout.cells():
        tx, ty = origin[0] + x, origin[1] + y
        if (tx, ty) in spared:
            continue
        outcome = bridge.act({
            "type": "place", "block": block, "x": tx, "y": ty, "rotation": rotation,
        })["action"]
        placed += bool(outcome.get("applied"))
    return placed


def delivered(observation: dict, item: str | None = None) -> int:
    """Ore that reached the core through a transport block, of one kind or of any.

    One kind, by default, and that is not fussiness. Counting everything gave the bench a
    trivial answer the search duly found: sand covers 3,848 tiles of this map against
    copper's 1,339, and plenty of it sits against the core. The winning design was eight
    drills on sand and **not one conveyor**, which is a real Mindustry strategy, an honest
    optimum, and no answer at all to the question of whether a line can be discovered.
    """
    produced = observation.get("produced", {})
    if item is None:
        return sum(int(amount) for amount in produced.values())
    return int(produced.get(item, 0))


def evaluate(bridge: Bridge, layout, origin: tuple[int, int],
             spared: set[tuple[int, int]], ticks: int, item: str | None = None) -> None:
    """Run one candidate and record what it delivered.

    The counter is the engine's own, and it counts only what arrives through a transport
    block. A drill dropped against the core delivers without any line at all and is
    counted here exactly as the game counts it, which is correct: if that is the best
    design, the search should be allowed to find it.
    """
    clear(bridge, origin, layout.width, layout.height, spared)
    placed = stamp(bridge, layout, origin, spared)

    before = delivered(bridge.observe(), item)
    after = delivered(bridge.step(repeat=ticks), item)

    # Ore that reached the core, plus a fraction of the ore still sitting in the design.
    # Without the second term every incomplete line scores exactly zero, so a line one
    # tile short is worth the same as an empty rectangle and the search has nothing to
    # climb. Measured before it existed: twenty-five generations, both genomes, zero
    # delivered, and the only pressure left was to build less, so the population shrank
    # to four blocks and stayed there.
    region = bridge.region(origin[0], origin[1], layout.width, layout.height)
    stuck = region["held"].get(item, 0) if item else sum(region["held"].values())

    layout.delivered = max(0, after - before)
    layout.stuck = int(stuck)
    layout.cost = placed


def render(candidate) -> str:
    """The layout as text, so a good one can be read rather than guessed at."""
    layout = candidate.to_layout() if hasattr(candidate, "to_layout") else candidate
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
    parser.add_argument("--genome", default="parts", choices=("cells", "parts"),
                        help="cells writes a layout one square at a time; parts writes it "
                             "as drills and lines, so a line is one gene and is never "
                             "wrong")
    parser.add_argument("--item", default="copper",
                        help="the only ore that counts; pass an empty string to score "
                             "anything that arrives")
    parser.add_argument("--keep-out", type=int, default=3,
                        help="tiles around the core whose ore does not count; a drill "
                             "touching the core delivers with no line at all, so without "
                             "this the bench has a trivial answer and the search finds it")
    parser.add_argument("--world-seed", type=int, default=20260824,
                        help="pins the ore, which the engine otherwise re-randomises on "
                             "every load, so two runs can be compared at all")
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
            observation = bridge.reset(args.map, "sandbox", seed=args.world_seed)
            server.command("bridge-speed max", r"speed set")

            core = (int(observation["core_x"]), int(observation["core_y"]))
            origin = best_patch(observation["spatial"], bridge.channels, core,
                                args.width, args.height, keep_out=args.keep_out)
            usable = reachable_ore(observation["spatial"], bridge.channels, core,
                                   keep_out=args.keep_out)
            ore = int((usable[origin[1]:origin[1] + args.height,
                              origin[0]:origin[0] + args.width] > 0).sum())
            spared = core_footprint(observation["spatial"], bridge.channels, core)

            print(f"map {args.map}, core at {core}, standing on {len(spared)} tiles")
            print(f"test rectangle {args.width}x{args.height} at {origin}, "
                  f"{ore} tiles of copper more than {args.keep_out} from the core")
            print(f"{args.population} layouts per generation, {args.generations} "
                  f"generations, {args.ticks} ticks each, genome '{args.genome}', "
                  f"scoring {args.item or 'anything'}")
            print()
            if ore == 0:
                raise SystemExit("no copper in the rectangle: nothing here can deliver "
                                 "anything, and the search would be measuring noise")

            kind = Population if args.genome == "cells" else DesignPopulation
            population = kind(args.width, args.height, size=args.population,
                              rng=__import__("random").Random(args.seed))
            population.seed()

            started = time.time()
            history = []

            for generation in range(1, args.generations + 1):
                for layout in population.members:
                    if layout.delivered is None:
                        evaluate(bridge, layout, origin, spared, args.ticks,
                                 args.item or None)

                best = population.best()
                scores = [fitness(layout) for layout in population.members]
                working = sum(1 for layout in population.members if (layout.delivered or 0) > 0)
                # Reported next to the score on purpose. A generation whose score is high
                # and whose delivery is not is a generation that found a way to be paid
                # for something other than the objective, and that has happened here.
                hoarded = max((getattr(m, "stuck", 0) for m in population.members), default=0)
                history.append({
                    "generation": generation,
                    "best_delivered": best.delivered if best else 0,
                    "best_cost": best.cost if best else 0,
                    "working": working,
                    "mean_fitness": round(sum(scores) / len(scores), 2),
                    "most_hoarded": hoarded,
                })
                print(f"generation {generation:3d}  best {best.delivered if best else 0:4d} ore "
                      f"with {best.used() if best else 0:3d} blocks  "
                      f"{working:3d}/{len(population.members)} deliver anything  "
                      f"mean {history[-1]['mean_fitness']:8.2f}  "
                      f"stuck {hoarded:5d}", flush=True)

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
                "map": args.map, "genome": args.genome, "world_seed": args.world_seed,
                "item": args.item,
                "rectangle": [args.width, args.height],
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
