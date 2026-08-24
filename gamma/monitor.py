"""Live view of a training run: every match at once, scores as they move.

Training is many environments stepping in parallel for hours. A log line per episode tells
you almost nothing about what is happening inside them, and a reward curve tells you it
went up without ever telling you why.

This collects the state of every running match and serves it over HTTP, so a browser can
show them side by side: what each agent is doing, what it holds, how far it is from the
objective, and which runs are pulling ahead.

Deliberately no dependencies. A dashboard that needs a web framework installed is a
dashboard that is not running when you want it.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

VIEWER_DIR = Path(__file__).resolve().parent.parent / "viewer"


@dataclass
class MatchState:
    """What one running environment looks like right now."""

    index: int
    policy: str = ""
    task: str = ""
    objective: str = ""

    step: int = 0
    max_steps: int = 0
    tick: float = 0.0
    wave: int = 0

    reward: float = 0.0
    best_reward: float = 0.0
    items: dict[str, int] = field(default_factory=dict)
    progress: float = 0.0

    unit: dict[str, Any] = field(default_factory=dict)
    action: str = ""
    refused: int = 0
    applied: int = 0

    episode: int = 0
    solved: int = 0
    finished: int = 0
    alive: bool = True

    #: Tile coordinates of what the agent has built, for the mini map.
    built: list[list[int]] = field(default_factory=list)
    core: list[int] = field(default_factory=lambda: [-1, -1])
    size: list[int] = field(default_factory=lambda: [0, 0])

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index, "policy": self.policy, "task": self.task,
            "objective": self.objective, "step": self.step, "max_steps": self.max_steps,
            "tick": round(self.tick), "wave": self.wave,
            "reward": round(self.reward, 3), "best_reward": round(self.best_reward, 3),
            "items": self.items, "progress": round(self.progress, 4),
            "unit": self.unit, "action": self.action,
            "refused": self.refused, "applied": self.applied,
            "episode": self.episode, "solved": self.solved,
            "finished": self.finished, "alive": self.alive,
            "built": self.built, "core": self.core, "size": self.size,
        }


class TrainingMonitor:
    """Collects match states and serves them. Thread safe: environments run in threads."""

    def __init__(self, title: str = "training") -> None:
        self.title = title
        self.started = time.time()
        self._matches: dict[int, MatchState] = {}
        self._lock = threading.Lock()
        self._server: ThreadingHTTPServer | None = None
        self._history: list[dict[str, Any]] = []

    # Collection ------------------------------------------------------------------

    def match(self, index: int) -> MatchState:
        with self._lock:
            if index not in self._matches:
                self._matches[index] = MatchState(index=index)
            return self._matches[index]

    def record_episode(self, index: int, reward: float, solved: bool) -> None:
        """Close out an episode, keeping just enough history for a trend line."""
        with self._lock:
            state = self._matches.get(index)
            if state is None:
                return
            state.episode += 1
            state.finished += 1
            state.solved += int(solved)
            state.best_reward = max(state.best_reward, reward)
            self._history.append({
                "at": round(time.time() - self.started, 1),
                "match": index,
                "policy": state.policy,
                "reward": round(reward, 3),
                "solved": bool(solved),
            })
            # Bounded on purpose: this is a live view, not a datastore.
            del self._history[:-500]

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            matches = [m.as_dict() for m in sorted(self._matches.values(), key=lambda m: m.index)]
            history = list(self._history)

        episodes = sum(m["finished"] for m in matches)
        solved = sum(m["solved"] for m in matches)
        elapsed = max(1e-6, time.time() - self.started)
        steps = sum(m["step"] for m in matches)

        leaderboard = sorted(
            ({"policy": m["policy"], "match": m["index"],
              "best": m["best_reward"], "solved": m["solved"], "episodes": m["finished"]}
             for m in matches),
            key=lambda row: (-row["best"], -row["solved"]),
        )

        return {
            "title": self.title,
            "elapsed": round(elapsed, 1),
            "matches": matches,
            "history": history,
            "leaderboard": leaderboard,
            "totals": {
                "matches": len(matches),
                "episodes": episodes,
                "solved": solved,
                "solve_rate": round(solved / episodes, 3) if episodes else 0.0,
                "steps_per_second": round(steps / elapsed, 1),
            },
        }

    # Serving ---------------------------------------------------------------------

    def serve(self, port: int = 8800) -> str:
        """Start the HTTP server in the background and return its URL."""
        monitor = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                if self.path.startswith("/state"):
                    body = json.dumps(monitor.snapshot()).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return

                name = "dashboard.html" if self.path in ("/", "") else self.path.lstrip("/")
                target = (VIEWER_DIR / name).resolve()
                if not target.is_file() or VIEWER_DIR.resolve() not in target.parents:
                    self.send_error(404)
                    return

                body = target.read_bytes()
                kind = "text/html" if target.suffix == ".html" else "application/octet-stream"
                self.send_response(200)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args: object) -> None:
                # Silent: request logs would drown the training output.
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        thread.start()
        return f"http://127.0.0.1:{port}/"

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server = None
