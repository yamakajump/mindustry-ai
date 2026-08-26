"""One line per episode, kept forever, so any question can be asked afterwards.

Asked one morning what the overnight progression had been, across ten restarts and 4.7
million steps, nothing anywhere could answer. The dashboard held four hundred points in
memory and lost them on exit. The replay archive prunes to the best five episodes per
environment, so reading it by age compares survivors against a fresh sample and
manufactures a collapse every time. The checkpoints hold weights, and weights do not say
where they came from.

Every diagnosis of the past two days therefore had to be re-derived from whatever episodes
happened to survive on disk at that moment, and twice the answer came out backwards
because the archive had moved underneath. That is the cost this file exists to remove.

A line is a few hundred bytes and an episode is thousands of steps, so a run of a hundred
thousand episodes costs tens of megabytes: cheap enough to never delete, which is the
whole point. What is written is deliberately raw. Rates and ratios are derived at reading
time, because a summary computed at writing time is a question decided in advance, and the
questions worth asking are not known yet.
"""

from __future__ import annotations

import json
import os
import threading
from collections import Counter
from pathlib import Path
from typing import Any

#: Written under the run's output directory, beside the checkpoints it explains.
DEFAULT_NAME = "journal.jsonl"

_lock = threading.Lock()


class Journal:
    """Append-only record of what happened, safe to share across environment threads.

    Twenty-four environments finish episodes on twenty-four threads. Each line is written
    in a single `write` call under a lock, so lines never interleave: a half-written line
    is one lost episode where a torn one would corrupt every reading that follows it.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        #: Stamped on every line by the trainer, so an episode can be placed on the curve.
        #: Held here rather than threaded through each worker: it is one integer that the
        #: trainer writes and the recorders read, and the alternative was a copy per
        #: environment that could drift.
        self.generation = 0

    def append(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, separators=(",", ":"), default=str) + "\n"
        try:
            with _lock, self.path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                # Flushed rather than left to the operating system. A run killed to apply a
                # fix is the normal way this ends, and the episodes it had just finished
                # are exactly the ones that say whether the fix was needed.
                handle.flush()
                os.fsync(handle.fileno())
        except OSError:
            # Losing a line must never bring a training run down.
            pass

    def read(self) -> list[dict[str, Any]]:
        """Every episode ever recorded here, skipping anything torn by a kill."""
        out: list[dict[str, Any]] = []
        if not self.path.exists():
            return out
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return out


class EpisodeTally:
    """Everything worth knowing about one episode, accumulated as it plays.

    Counted from the frames the recorder already builds, so nothing here costs a second
    pass over the game. The fields are the ones that answered a real question over the past
    two days, and each earned its place by a diagnosis that could not be made without it:

    - `actions` and `refused`, because a third of the agent's life turned out to be spent
      on actions the world would not accept, and no number said so
    - `reasons`, because "no route" hid three unrelated causes behind one word
    - `blocks`, because the agent laid nine turrets an episode and they defended nothing
    - `self_broken`, because a demolition loop was visible on screen while the counter
      built to price it read zero
    - `terms`, because a rising score twice turned out to be an exploit rather than skill
    """

    def __init__(self) -> None:
        self.actions: Counter = Counter()
        self.refused: Counter = Counter()
        self.reasons: Counter = Counter()
        self.blocks: Counter = Counter()
        self.terms: Counter = Counter()
        self.steps = 0
        self.reward = 0.0
        self.wave = 0
        self.core_lost = False
        self.self_broken = 0
        self.turret_spread: list[float] = []
        self._built: set[tuple[int, int]] = set()

    def note(self, frame: dict[str, Any], core: tuple[int, int] | None = None) -> None:
        self.steps += 1
        self.reward += float(frame.get("reward", 0.0))
        self.wave = max(self.wave, int(frame.get("wave", 0)))

        for name, value in (frame.get("terms") or {}).items():
            self.terms[name] += float(value)
            if name == "core_lost":
                self.core_lost = True

        act = frame.get("act") or {}
        kind = str(act.get("t", ""))
        if not kind:
            return

        if frame.get("refused"):
            self.refused[kind] += 1
            self.reasons[f"{kind}:{act.get('reason', '')}"] += 1
            return

        self.actions[kind] += 1

        cells = list(act.get("cells") or [])
        if kind == "place" and act.get("b") is not None:
            cells.append([act.get("x"), act.get("y"), act["b"]])

        for cell in cells:
            if len(cell) < 2:
                continue
            where = (int(cell[0]), int(cell[1]))
            if kind == "break":
                self.self_broken += where in self._built
                self._built.discard(where)
                continue
            self._built.add(where)
            if len(cell) > 2:
                block = str(cell[2])
                self.blocks[block] += 1
                if core is not None and ("turret" in block or block in _TURRETS):
                    self.turret_spread.append(
                        abs(where[0] - core[0]) + abs(where[1] - core[1]))

        if kind == "break":
            where = (int(act.get("x", -1)), int(act.get("y", -1)))
            self.self_broken += where in self._built
            self._built.discard(where)

    def summary(self, **extra: Any) -> dict[str, Any]:
        spread = self.turret_spread
        return {
            "steps": self.steps,
            "reward": round(self.reward, 3),
            "wave": self.wave,
            "core_lost": self.core_lost,
            "actions": dict(self.actions),
            "refused": dict(self.refused),
            "reasons": {k: v for k, v in self.reasons.items() if v},
            "blocks": dict(self.blocks),
            "terms": {k: round(v, 3) for k, v in self.terms.items()},
            "self_broken": self.self_broken,
            "turrets": len(spread),
            "turret_distance": round(sum(spread) / len(spread), 1) if spread else None,
            **extra,
        }


#: Blocks that shoot, by name, for the ones whose name does not say so.
_TURRETS = frozenset({"duo", "scatter", "scorch", "hail", "wave", "lancer", "arc",
                      "swarmer", "salvo", "fuse", "ripple", "cyclone", "foreshadow",
                      "spectre", "meltdown", "segment", "parallax", "tsunami"})
