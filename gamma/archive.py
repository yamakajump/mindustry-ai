"""Keeping the replays worth keeping.

A training run plays thousands of episodes. Recording all of them fills a disk and buries
the interesting ones; recording none leaves the run with nothing to show but a curve.

So episodes are recorded continuously and pruned on a rule that answers the two questions
someone actually asks of a run: *what does its best attempt look like*, and *what is it
doing right now*. The best few are kept forever, the latest few are kept until something
newer arrives, and everything else is deleted as soon as it loses on both counts.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: episode number, reward with the sign spelled out so a filename never grows a minus that
#: a shell would read as an option.
_NAME = re.compile(r"^ep(\d+)-(neg|pos)(\d+)\.jsonl\.gz$")


@dataclass(frozen=True)
class Episode:
    """One recorded episode on disk."""

    number: int
    reward: float
    solved: bool
    path: Path

    def as_dict(self) -> dict[str, Any]:
        return {
            "episode": self.number,
            "reward": round(self.reward, 3),
            "solved": self.solved,
            "file": self.path.name,
            "size": self.path.stat().st_size if self.path.exists() else 0,
        }


class ReplayArchive:
    """Names, ranks and prunes the replays of one match.

    Thread safe: episodes finish on an environment thread while the dashboard reads the
    listing on an HTTP thread.
    """

    def __init__(self, directory: Path | str, keep_best: int = 5, keep_recent: int = 3) -> None:
        self.directory = Path(directory)
        self.keep_best = keep_best
        self.keep_recent = keep_recent
        self._episodes: list[Episode] = []
        self._solved: set[int] = set()
        self._lock = threading.Lock()
        self.directory.mkdir(parents=True, exist_ok=True)
        for stale in self.directory.glob("*.jsonl.gz"):
            # A previous run's replays refer to a previous run's policy. Keeping them
            # would put another agent's best episode at the top of this run's board.
            stale.unlink(missing_ok=True)

    # Naming ----------------------------------------------------------------------

    def pending(self, number: int) -> Path:
        """Where the episode being recorded should be written.

        Its reward is unknown until it ends, so it is written under a neutral name and
        renamed on the way out.
        """
        return self.directory / f"ep{number:06d}.pending.jsonl.gz"

    def commit(self, number: int, reward: float, solved: bool) -> Episode | None:
        """Give a finished recording its final name, then prune what it displaced."""
        source = self.pending(number)
        if not source.exists():
            return None

        sign = "pos" if reward >= 0 else "neg"
        target = self.directory / f"ep{number:06d}-{sign}{abs(round(reward * 100)):06d}.jsonl.gz"
        try:
            source.replace(target)
        except OSError:
            source.unlink(missing_ok=True)
            return None

        episode = Episode(number=number, reward=reward, solved=solved, path=target)
        with self._lock:
            self._episodes.append(episode)
            if solved:
                self._solved.add(number)
            self._prune()
        return episode

    def discard(self, number: int) -> None:
        """Drop a recording that will never be committed, so it does not linger."""
        self.pending(number).unlink(missing_ok=True)

    # Selection -------------------------------------------------------------------

    def _prune(self) -> None:
        """Keep the best, the most recent, and every solved episode. Delete the rest.

        Solved episodes survive regardless of reward: they are the proof the task is
        achievable at all, and on a hard task there may only ever be a handful.
        """
        best = sorted(self._episodes, key=lambda e: -e.reward)[:self.keep_best]
        recent = sorted(self._episodes, key=lambda e: -e.number)[:self.keep_recent]
        keep = {id(e) for e in best} | {id(e) for e in recent}
        keep |= {id(e) for e in self._episodes if e.solved}

        survivors = []
        for episode in self._episodes:
            if id(episode) in keep:
                survivors.append(episode)
            else:
                episode.path.unlink(missing_ok=True)
        self._episodes = survivors

    def listing(self) -> list[dict[str, Any]]:
        """Everything kept, best first, for a viewer to offer."""
        with self._lock:
            ordered = sorted(self._episodes, key=lambda e: (-e.reward, -e.number))
            return [episode.as_dict() for episode in ordered]

    def resolve(self, name: str) -> Path | None:
        """The file behind a listed name, or None if it is not one of ours.

        Names come from a URL, so they are checked against the archive rather than
        trusted: joining an arbitrary string to a directory is how a viewer ends up
        serving the private key next to it.
        """
        if not _NAME.match(name):
            return None
        with self._lock:
            for episode in self._episodes:
                if episode.path.name == name:
                    return episode.path
        return None
