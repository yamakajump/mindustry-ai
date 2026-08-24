"""Is what the search invented better than what a person wrote by hand?

    python tools/compare_designs.py docs/measurements/cost-0.20.json

Alpha is the scripted baseline: find the nearest copper, put a drill on it, run a conveyor
back to the core, repeat. A person wrote that routine, and the whole curriculum is measured
against it.

This plays it against a design nobody wrote, on the same world, from the same start, for
the same number of steps. The score is copper the engine says arrived at the core through
a transport block, so hand mining cannot flatter either of them.

It needs no training at all, which is the point. If the vocabulary the search discovered
cannot beat a routine written in an afternoon, there is nothing worth training on top of
it, and finding that out costs four minutes rather than four hours.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from gamma import tasks
from gamma.alpha import AlphaPolicy
from gamma.env import MindustryEnv
from gamma.adapt import lay, route, split
from gamma.library import Design, from_evolution, save
from gamma.policies import MaskedRandomPolicy

BRIDGE_PORT = 7890
GAME_PORT = 6890


class BlindStampPolicy:
    """Stamp the design at tiles chosen at random, never looking at the ore.

    This is the guardrail for the whole approach, and it has to be run rather than argued
    about. If a policy that picks its spots blindly does as well as one that learned where
    to pick them, then the structure is doing all the work, the learning is decoration, and
    what has been built is a scripted bot with a chooser bolted on.

    It gets the same number of stamps and the same structure. The only thing it lacks is
    any idea of where to put them.
    """

    def __init__(self, env, design: Design, copies: int = 3, seed: int = 0,
                 reach: int = 24) -> None:
        self.env = env
        self.design = design
        self.copies = copies
        self.reach = reach
        self.rng = np.random.default_rng(seed)
        self.queue: list[dict] = []
        self.placed = 0

    def reset(self) -> None:
        self.queue = []
        self.placed = 0

    def act(self, observation, info) -> np.ndarray:
        raw = info.get("raw", {})
        if not self.queue and self.placed == 0:
            core = (int(raw.get("core_x", -1)), int(raw.get("core_y", -1)))
            if core[0] < 0:
                return np.zeros(5, dtype=np.int64)

            cells: list[tuple[int, int, str, int]] = []
            shape = split(self.design)
            for _ in range(self.copies):
                ax = int(core[0] + self.rng.integers(-self.reach, self.reach + 1))
                ay = int(core[1] + self.rng.integers(-self.reach, self.reach + 1))
                cells += [(ax + p.dx, ay + p.dy, p.block, p.rotation)
                          for p in shape.producers]
                cells += route((ax, ay), core)

            cells.sort(key=lambda cell: 0 if "drill" in cell[2] else 1)
            self.queue = [
                {"type": "place", "block": block, "x": x, "y": y, "rotation": rotation}
                for x, y, block, rotation in cells
            ]

        if not self.queue:
            return np.zeros(5, dtype=np.int64)

        action = self.queue.pop(0)
        self.placed += 1
        if action["block"] not in self.env.blocks:
            return np.zeros(5, dtype=np.int64)
        return np.array([
            self.env.action_types.index("place"),
            self.env.blocks.index(action["block"]),
            action["x"], action["y"], action["rotation"],
        ], dtype=np.int64)


class StampPolicy:
    """Place a design once, then stand back and let it run.

    Deliberately does nothing else. The question is whether the structure delivers, not
    whether a policy wrapped around it can be clever, and anything more here would make
    the comparison about the wrapper.

    Two ways of placing it, because the difference between them is the whole finding.
    Anchored on the core, a design is a set of fixed offsets: it delivered 264 items on
    each of three unseen worlds and zero copper on any of them, because the drills landed
    on whatever lay at those offsets and that was sand. Anchored on ore, the shape the
    search found is dropped where there is something to mine and the line to the core is
    recomputed for the distance it actually has to cover.
    """

    def __init__(self, env, design: Design, adaptive: bool = True,
                 item: str = "copper", copies: int = 1) -> None:
        self.env = env
        self.design = design
        self.adaptive = adaptive
        self.item = item
        self.copies = copies
        self.queue: list[dict] = []
        self.placed = 0

    def reset(self) -> None:
        self.queue = []
        self.placed = 0

    def _cells(self, raw) -> list[tuple[int, int, str, int]]:
        core = (int(raw.get("core_x", -1)), int(raw.get("core_y", -1)))
        if core[0] < 0:
            return []
        if not self.adaptive:
            return self.design.at(core)

        channel = f"ore_{self.item}"
        channels = self.env._bridge.channels if self.env._bridge else []
        if "spatial" not in raw or channel not in channels:
            return self.design.at(core)
        return lay(self.design, raw["spatial"][channels.index(channel)], core,
                   copies=self.copies)

    def act(self, observation, info) -> np.ndarray:
        raw = info.get("raw", {})
        if not self.queue and self.placed == 0:
            cells = self._cells(raw)
            if not cells:
                return np.zeros(5, dtype=np.int64)
            # Drills before anything else: a conveyor on a tile a drill wanted makes the
            # drill impossible and leaves a line fed by nothing.
            cells.sort(key=lambda cell: 0 if "drill" in cell[2] else 1)
            self.queue = [
                {"type": "place", "block": block, "x": x, "y": y, "rotation": rotation}
                for x, y, block, rotation in cells
            ]

        if not self.queue:
            return np.zeros(5, dtype=np.int64)

        action = self.queue.pop(0)
        self.placed += 1
        if action["block"] not in self.env.blocks:
            return np.zeros(5, dtype=np.int64)
        return np.array([
            self.env.action_types.index("place"),
            self.env.blocks.index(action["block"]),
            action["x"], action["y"], action["rotation"],
        ], dtype=np.int64)


def play(env, policy, steps: int) -> dict:
    """One episode, reporting what the engine says arrived through a transport block."""
    observation, info = env.reset()
    if hasattr(policy, "reset"):
        policy.reset()

    applied = refused = 0
    raw = info["raw"]
    for _ in range(steps):
        observation, _, terminated, truncated, info = env.step(policy.act(observation, info))
        raw = info["raw"]
        outcome = info.get("action") or {}
        if outcome:
            applied += bool(outcome.get("applied"))
            refused += not outcome.get("applied")
        if terminated or truncated:
            break

    produced = raw.get("produced", {})
    return {
        "delivered": int(produced.get("copper", 0)),
        "all_delivered": sum(int(a) for a in produced.values()),
        "banked": int(raw.get("items", {}).get("copper", 0)),
        "built": int(raw.get("built", 0)),
        "applied": applied,
        "refused": refused,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("design", type=Path,
                        help="a report written by tools/evolve_layout.py")
    parser.add_argument("--map", default="Ancient_Caldera")
    parser.add_argument("--world-seed", type=int, default=None,
                        help="defaults to the world the design was found on, which is the "
                             "only one it can honestly claim anything about")
    parser.add_argument("--steps", type=int, default=450)
    parser.add_argument("--out", type=Path,
                        default=Path("docs/measurements/designs-vs-alpha.json"))
    parser.add_argument("--library", type=Path, default=Path("docs/designs.json"))
    parser.add_argument("--jar", type=Path, default=None)
    args = parser.parse_args()

    design = from_evolution(args.design)
    seed = args.world_seed if args.world_seed is not None else design.world_seed
    if seed is None:
        raise SystemExit("no world seed: the comparison would run on two different maps")

    jar = str(args.jar or next((Path("bridge") / "build" / "libs").glob("*.jar")))
    task = replace(tasks.T1_COPPER, map_name=args.map, world_seed=seed,
                   max_steps=args.steps)

    env = MindustryEnv(task, server_dir="mindustry-compare", bridge_port=BRIDGE_PORT,
                       game_port=GAME_PORT, jar=jar, embodied=False, speed="max")

    print(f"design  : {design.name}, {len(design)} blocks, "
          f"{design.delivered} {design.item or 'ore'} on the bench")
    print(f"world   : {args.map} pinned to seed {seed}")
    print(f"episode : {args.steps} steps, direct placement, cost paid as usual")
    print()

    try:
        contenders = [
            ("alpha", AlphaPolicy(env)),
            ("discovered", StampPolicy(env, design, adaptive=True)),
            ("discovered x3", StampPolicy(env, design, adaptive=True, copies=3)),
            ("discovered x6", StampPolicy(env, design, adaptive=True, copies=6)),
            ("blind x3", BlindStampPolicy(env, design, copies=3)),
            ("blind x6", BlindStampPolicy(env, design, copies=6)),
            ("random", MaskedRandomPolicy(env.action_space, seed=0, env=env)),
        ]
        results = {}
        for name, policy in contenders:
            outcome = play(env, policy, args.steps)
            results[name] = outcome
            print(f"{name:>11}  copper delivered {outcome['delivered']:6d}  "
                  f"any ore {outcome['all_delivered']:6d}  "
                  f"banked {outcome['banked']:5d}  built {outcome['built']:4d}  "
                  f"applied {outcome['applied']:4d}/{outcome['applied'] + outcome['refused']}",
                  flush=True)
    finally:
        env.close()

    print()
    alpha, discovered = results["alpha"]["delivered"], results["discovered"]["delivered"]
    if discovered > alpha:
        print(f"The design nobody wrote beats the routine somebody did, "
              f"{discovered} to {alpha}.")
        save([design], args.library)
        print(f"kept in {args.library}")
    elif discovered == alpha == 0:
        print("Neither delivered anything. The world, the step budget or the placement is "
              "wrong, and the comparison says nothing until that is fixed.")
    else:
        print(f"Alpha still wins, {alpha} to {discovered}. The vocabulary is not worth "
              f"training on yet.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "design": design.name, "map": args.map, "world_seed": seed,
        "steps": args.steps, "blocks": len(design), "results": results,
    }, indent=2), encoding="utf-8")
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
