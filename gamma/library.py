"""Designs the search discovered, stored so they can be played.

A layout found by the forge is a rectangle of blocks that delivered ore on one world. To
be worth anything it has to leave the bench: be written down, put back on a map, and beaten
against the routine a person wrote by hand. That is what this holds.

Positions are kept **relative to the core**, not to the rectangle they were evolved in. A
design is a claim about where things go with respect to the base, and the rectangle was
only ever scaffolding. Stored the other way it could not be placed on a world whose core
sits somewhere else, which is every world.

Nothing here is a human blueprint and nothing here ever will be. Copying a schematic from
somebody who already solved the game would automate a player rather than beat one, and
would prove nothing about the agent. Every design in a library comes out of the search.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Placement:
    """One block of a design, offset from the core."""

    dx: int
    dy: int
    block: str
    rotation: int


@dataclass
class Design:
    """A structure the search found, and what it was measured to do.

    The provenance is kept with it on purpose. A design is only ever evidence about the
    world it was evolved on, and a library that has forgotten which world, which ore and
    how much is a library of unsourced claims.
    """

    name: str
    placements: tuple[Placement, ...]
    #: What it delivered on the bench, and of what.
    delivered: int = 0
    item: str = ""
    #: The world it was found on, so the claim can be rerun.
    world_seed: int | None = None
    ticks: int = 0
    notes: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.placements)

    def at(self, core: tuple[int, int]) -> list[tuple[int, int, str, int]]:
        """The design in map coordinates, given where the core is."""
        cx, cy = core
        return [(cx + p.dx, cy + p.dy, p.block, p.rotation) for p in self.placements]

    def actions(self, core: tuple[int, int]) -> list[dict[str, Any]]:
        """The design as bridge actions, drills before anything else.

        Order matters and it is not cosmetic. A drill needs two clear tiles by two, so a
        conveyor laid first on a tile the drill wanted makes the drill impossible and
        leaves a line fed by nothing.
        """
        cells = self.at(core)
        cells.sort(key=lambda cell: 0 if "drill" in cell[2] else 1)
        return [
            {"type": "place", "block": block, "x": x, "y": y, "rotation": rotation}
            for x, y, block, rotation in cells
        ]


def from_evolution(path: Path, name: str | None = None) -> Design:
    """Read what the forge wrote, and re-anchor it on the core.

    The search reports cells inside its test rectangle, along with where that rectangle sat
    and where the core was. Subtracting one from the other is the whole conversion, and it
    is done here rather than there so the search stays free to move its rectangle around
    without every stored design becoming wrong.

    The search itself lives in [mindustry-forge](https://github.com/yamakajump/mindustry-forge). It is a
    different job with a different loop: it asks what the best machine for a job is, over
    minutes, and this project asks where and when to build one, over hours. What crosses
    between them is this file and nothing else.
    """
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    best = report.get("best")
    if not best:
        raise ValueError(f"{path} holds no design: the run delivered nothing")

    ox, oy = report["origin"]
    cx, cy = report["core"]

    placements = tuple(
        Placement(ox + int(x) - cx, oy + int(y) - cy, str(block), int(rotation))
        for x, y, block, rotation in best["cells"]
    )

    return Design(
        name=name or Path(path).stem,
        placements=placements,
        delivered=int(best["delivered"]),
        item=report.get("item", ""),
        world_seed=report.get("world_seed"),
        ticks=int(report.get("ticks", 0)),
        notes={"map": report.get("map"), "genome": report.get("genome"),
               "blocks": int(best["blocks"])},
    )


def save(designs: list[Design], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([
        {
            "name": d.name, "delivered": d.delivered, "item": d.item,
            "world_seed": d.world_seed, "ticks": d.ticks, "notes": d.notes,
            "placements": [[p.dx, p.dy, p.block, p.rotation] for p in d.placements],
        }
        for d in designs
    ], indent=2), encoding="utf-8")


def load(path: Path) -> list[Design]:
    return [
        Design(
            name=entry["name"],
            placements=tuple(Placement(*p) for p in entry["placements"]),
            delivered=entry.get("delivered", 0),
            item=entry.get("item", ""),
            world_seed=entry.get("world_seed"),
            ticks=entry.get("ticks", 0),
            notes=entry.get("notes", {}),
        )
        for entry in json.loads(Path(path).read_text(encoding="utf-8"))
    ]
