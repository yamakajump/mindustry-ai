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
import socket
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

VIEWER_DIR = Path(__file__).resolve().parent.parent / "viewer"


def _listening(port: int, host: str = "127.0.0.1") -> bool:
    """Whether something already answers on this port."""
    try:
        with socket.create_connection((host, port), timeout=0.3):
            return True
    except OSError:
        return False


class SceneBuffer:
    """The animated world of one match, kept current and served incrementally.

    The bridge hands out deltas against its own last frame, which suits a single reader
    and nobody else: a browser that polls twice a second would only ever receive the two
    frames it happened to land on, and would miss everything in between. So the deltas are
    folded here into an authoritative world, and every change is stamped with a version.
    A viewer then asks for what changed since the version it holds and gets exactly that,
    whatever rate it polls at, and a viewer arriving late gets the whole world instead.

    Units are always sent in full. There are hundreds of them at worst, they all move every
    frame anyway, and tracking arrivals and departures separately would cost more code than
    the bytes it saves.
    """

    #: Removals kept for replay to late pollers. Beyond this a viewer is resynced from
    #: scratch, which is correct and rare, rather than shown a building that no longer
    #: exists.
    MAX_REMOVED = 2000

    def __init__(self) -> None:
        self.version = 0
        self.units: list[float] = []
        self.shots: list[float] = []
        self.buildings: dict[int, list[int]] = {}
        self.changed_at: dict[int, int] = {}
        self.removed: list[tuple[int, int]] = []
        #: Versions below this no longer have their removals retained, so a viewer that
        #: far behind has to be resynced rather than patched.
        self.dropped_before = 0
        self.blocks: dict[str, Any] = {}
        self.types: dict[str, Any] = {}
        self.item_names: dict[str, str] = {}
        #: Conveyors and what rides on them. Replaced wholesale each frame: a belt is
        #: nothing but movement, so there is no delta to take.
        self.belts: list[float] = []
        #: Where each turret points and how hard it just kicked, whole each frame.
        self.turrets: list[float] = []
        #: Item handovers not yet handed to a viewer. Queued rather than overwritten: at
        #: ten frames a second a viewer polling at five would miss half of them.
        self.deposits: list[list[float]] = []
        #: Which unit the agent inhabits, so a viewer never has to guess.
        self.agent = -1
        self.tick = 0.0
        self.wave = 0
        self.wave_time = 0.0
        self.enemies = 0
        self.width = 0
        self.height = 0
        self._lock = threading.Lock()

    def clear(self) -> None:
        """Drop the world, keeping the version so viewers resync rather than rewind."""
        with self._lock:
            self.version += 1
            self.units = []
            self.shots = []
            self.belts = []
            self.turrets = []
            self.deposits.clear()
            self.buildings.clear()
            self.changed_at.clear()
            self.removed.clear()
            self.dropped_before = self.version
            self.blocks.clear()
            self.types.clear()
            self.item_names.clear()

    def apply(self, frame: dict[str, Any]) -> None:
        """Fold one bridge frame into the world."""
        if not frame.get("playing"):
            return

        with self._lock:
            self.version += 1
            version = self.version

            self.tick = float(frame.get("tick", 0.0))
            self.wave = int(frame.get("wave", 0))
            self.wave_time = float(frame.get("wave_time", 0.0))
            self.enemies = int(frame.get("enemies", 0))
            self.width = int(frame.get("width", self.width))
            self.height = int(frame.get("height", self.height))

            self.agent = int(frame.get("agent", -1))
            self.units = list(frame.get("units", []))
            self.shots = list(frame.get("shots", []))
            self.belts = list(frame.get("belts", []))
            self.turrets = list(frame.get("turrets", []))
            if frame.get("deposit"):
                # Stamped with the version, so a viewer plays each handover exactly once
                # however often it polls.
                self.deposits.append([version, *(float(v) for v in frame["deposit"])])
                del self.deposits[:-16]
            self.blocks.update(frame.get("blocks") or {})
            self.types.update(frame.get("types") or {})
            self.item_names.update(frame.get("items") or {})

            placed = frame.get("placed", [])
            for i in range(0, len(placed) - 5, 6):
                tile = int(placed[i])
                self.buildings[tile] = [int(v) for v in placed[i + 1:i + 6]]
                self.changed_at[tile] = version

            hurt = frame.get("hurt", [])
            for i in range(0, len(hurt) - 1, 2):
                tile = int(hurt[i])
                building = self.buildings.get(tile)
                if building is not None:
                    building[3] = int(hurt[i + 1])
                    self.changed_at[tile] = version

            for tile in frame.get("removed", []):
                tile = int(tile)
                self.buildings.pop(tile, None)
                self.changed_at.pop(tile, None)
                self.removed.append((version, tile))

            if len(self.removed) > self.MAX_REMOVED:
                dropped = self.removed[:-self.MAX_REMOVED]
                del self.removed[:-self.MAX_REMOVED]
                self.dropped_before = max(self.dropped_before, dropped[-1][0])

    def since(self, version: int) -> dict[str, Any]:
        """What a viewer holding `version` needs to catch up.

        A version of zero, or one older than the retained removals, is answered with the
        whole world. Anything else is answered with the difference.
        """
        with self._lock:
            # Behind the retention window, or ahead of us because the match restarted:
            # either way the only honest answer is the whole world.
            full = version <= 0 or version < self.dropped_before or version > self.version

            if full:
                placed: list[int] = []
                for tile, building in self.buildings.items():
                    placed.append(tile)
                    placed.extend(building)
                removed: list[int] = []
                blocks = dict(self.blocks)
                types = dict(self.types)
            else:
                placed = []
                for tile, changed in self.changed_at.items():
                    if changed > version:
                        placed.append(tile)
                        placed.extend(self.buildings[tile])
                removed = [tile for at, tile in self.removed if at > version]
                # Descriptions are cumulative and small, so they ride along every time
                # rather than being tracked. A viewer missing one draws a blank square.
                blocks = dict(self.blocks)
                types = dict(self.types)

            return {
                "version": self.version,
                "full": full,
                "agent": self.agent,
                "tick": round(self.tick),
                "wave": self.wave,
                "wave_time": round(self.wave_time),
                "enemies": self.enemies,
                "width": self.width,
                "height": self.height,
                "units": self.units,
                "shots": self.shots,
                "belts": self.belts,
                "turrets": self.turrets,
                "items": self.item_names,
                # A viewer arriving late replays nothing: these are moments, and playing
                # a minute of banked handovers at once would be a firework display.
                "deposits": [] if full else [d[1:] for d in self.deposits if d[0] > version],
                "placed": placed,
                "removed": removed,
                "blocks": blocks,
                "types": types,
            }


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

    #: Ore a machine delivered to the core, as opposed to ore the agent carried in. The
    #: single number that says whether this match has an economy or a busy pair of hands.
    produced: int = 0

    #: Rungs of the milestone ladder this episode has reached, by name. Worth far more on
    #: screen than a reward: a match that never reaches `automation` has not started
    #: playing, whatever it scores.
    reached: list[str] = field(default_factory=list)

    unit: dict[str, Any] = field(default_factory=dict)
    action: str = ""
    refused: int = 0
    applied: int = 0

    episode: int = 0
    solved: int = 0
    finished: int = 0
    alive: bool = True

    #: Every step this match has ever taken, across episodes.
    #:
    #: Distinct from `step`, which is the position inside the current episode and drops
    #: back to zero every time one ends. Summing `step` across matches and dividing by the
    #: elapsed time is what the throughput figure used to do, and it reported 41 steps/s
    #: for a run the trainer measured at 283: it was averaging how far into their episodes
    #: the matches happened to be.
    total_steps: int = 0

    #: What the agent has built, as [x, y, block name], so the map can use real sprites.
    built: list[Any] = field(default_factory=list)
    core: list[int] = field(default_factory=lambda: [-1, -1])
    size: list[int] = field(default_factory=lambda: [0, 0])

    #: The map itself, sent once per episode: base64 planes plus the block palette. Large
    #: enough that it is served on its own endpoint rather than in every poll.
    terrain: dict[str, Any] | None = None
    terrain_version: int = 0

    #: Size of the map the terrain describes. The scene carries its own, and a viewer that
    #: draws entities against a terrain of a different size puts them nowhere.
    terrain_size: list[int] = field(default_factory=lambda: [0, 0])

    #: Port a real Mindustry client can join to watch this match. The dashboard draws a
    #: map; this is the only way to see the actual game, animations and all.
    game_port: int = 0

    #: True when this match runs at a speed a human can follow.
    watchable: bool = False

    #: Units, buildings and shots as they move. Served on its own endpoint: it changes
    #: several times a second while the rest of this object changes once per step.
    scene: SceneBuffer = field(default_factory=SceneBuffer)

    #: Recorded episodes kept for this match, best first.
    replays: list[Any] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index, "policy": self.policy, "task": self.task,
            "objective": self.objective, "step": self.step, "max_steps": self.max_steps,
            "tick": round(self.tick), "wave": self.wave,
            "reward": round(self.reward, 3), "best_reward": round(self.best_reward, 3),
            "items": self.items, "progress": round(self.progress, 4),
            "produced": self.produced, "reached": self.reached,
            "unit": self.unit, "action": self.action,
            "refused": self.refused, "applied": self.applied,
            "episode": self.episode, "solved": self.solved,
            "total_steps": self.total_steps,
            "finished": self.finished, "alive": self.alive,
            "built": self.built, "core": self.core, "size": self.size,
            "terrain_version": self.terrain_version,
            "terrain_size": self.terrain_size,
            "game_port": self.game_port, "watchable": self.watchable,
            "replays": self.replays,
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
        #: One entry per training update, so progress can be read as a curve rather than
        #: guessed from whichever episodes happen to be on screen.
        self._generations: list[dict[str, Any]] = []
        self._pending: list[float] = []
        #: Replay archives by match, so a recorded episode can be downloaded by name
        #: without the server ever joining a path from a URL.
        self._archives: dict[int, Any] = {}

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
                "wave": state.wave,
            })
            self._pending.append(reward)
            # Bounded on purpose: this is a live view, not a datastore.
            del self._history[:-500]

    def record_generation(self, update: int, stats: dict[str, float]) -> dict[str, Any]:
        """Close a training update, summarising the episodes that fed it.

        This is what makes progress legible. Individual episodes are noisy enough that a
        good run and a lucky one look the same; the mean across an update does not.
        """
        with self._lock:
            rewards = list(self._pending)
            self._pending.clear()
            best_wave = max((m.wave for m in self._matches.values()), default=0)

            generation = {
                "update": update,
                "at": round(time.time() - self.started, 1),
                "episodes": len(rewards),
                "mean_reward": round(sum(rewards) / len(rewards), 3) if rewards else None,
                "best_reward": round(max(rewards), 3) if rewards else None,
                "best_wave": best_wave,
                "entropy": round(stats.get("entropy", 0.0), 3),
                "value_loss": round(stats.get("value_loss", 0.0), 4),
                #: Set once the weights behind this generation are written to disk, so a
                #: point on the curve can be pointed at a file rather than admired.
                "checkpoint": None,
                "best": False,
            }
            self._generations.append(generation)
            del self._generations[:-400]
            return generation

    def register_replays(self, index: int, archive: Any) -> None:
        with self._lock:
            self._archives[index] = archive

    def replay_path(self, index: int, name: str):
        with self._lock:
            archive = self._archives.get(index)
        return archive.resolve(name) if archive else None

    def scene_of(self, index: int, since: int) -> dict[str, Any] | None:
        with self._lock:
            state = self._matches.get(index)
        return state.scene.since(since) if state else None

    def annotate_generation(self, update: int, **fields: Any) -> None:
        """Attach what was only known after the fact, such as the file that was saved."""
        with self._lock:
            for generation in reversed(self._generations):
                if generation["update"] == update:
                    generation.update(fields)
                    return

    def terrain_of(self, index: int) -> dict[str, Any] | None:
        with self._lock:
            state = self._matches.get(index)
            return state.terrain if state else None

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            matches = [m.as_dict() for m in sorted(self._matches.values(), key=lambda m: m.index)]
            history = list(self._history)
            generations = list(self._generations)

        episodes = sum(m["finished"] for m in matches)
        solved = sum(m["solved"] for m in matches)
        elapsed = max(1e-6, time.time() - self.started)
        steps = sum(m["total_steps"] for m in matches)

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
            "generations": generations,
            "leaderboard": leaderboard,
            "totals": {
                "matches": len(matches),
                "episodes": episodes,
                "solved": solved,
                "solve_rate": round(solved / episodes, 3) if episodes else 0.0,
                "steps_per_second": round(steps / elapsed, 1),
                "updates": len(generations),
                "best_wave": max((m["wave"] for m in matches), default=0),
                "best_generation": next(
                    (g["update"] for g in reversed(generations) if g["best"]), None),
                "best_reward": round(max((m["best_reward"] for m in matches), default=0.0), 2),
            },
        }

    # Serving ---------------------------------------------------------------------

    def serve(self, port: int = 8800, strict: bool = False) -> str:
        """Start the HTTP server in the background and return its URL.

        With `strict`, a port already in use is an error rather than something to step
        around. Stepping around it is what let two training runs start at once, each
        clearing the other's Mindustry servers and reloading worlds underneath it, while
        the dashboard on the first port described a run that was no longer the one doing
        the work.
        """
        monitor = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                if self.path.startswith("/terrain"):
                    # Served separately: a 256x256 typed map is tens of kilobytes, and
                    # putting it in every half-second poll would swamp the connection.
                    try:
                        index = int(self.path.rsplit("/", 1)[-1])
                    except ValueError:
                        self.send_error(400)
                        return
                    terrain = monitor.terrain_of(index)
                    if terrain is None:
                        self.send_error(404)
                        return
                    self._json(terrain)
                    return

                if self.path.startswith("/replay/"):
                    parts = self.path[len("/replay/"):].split("/")
                    if len(parts) != 2 or not parts[0].isdigit():
                        self.send_error(400)
                        return
                    target = monitor.replay_path(int(parts[0]), parts[1])
                    if target is None or not target.is_file():
                        self.send_error(404)
                        return
                    body = target.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/gzip")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return

                if self.path.startswith("/scene/"):
                    # `since` lets a viewer receive only what changed while it was away,
                    # at whatever rate it polls, instead of the frame the bridge happened
                    # to produce last.
                    target, _, query = self.path[len("/scene/"):].partition("?")
                    try:
                        index = int(target)
                    except ValueError:
                        self.send_error(400)
                        return
                    since = 0
                    for pair in query.split("&"):
                        key, _, value = pair.partition("=")
                        if key == "since" and value.isdigit():
                            since = int(value)
                    scene = monitor.scene_of(index, since)
                    if scene is None:
                        self.send_error(404)
                        return
                    self._json(scene)
                    return

                if self.path.startswith("/state"):
                    self._json(monitor.snapshot())
                    return

                name = "dashboard.html" if self.path in ("/", "") else self.path.lstrip("/")
                target = (VIEWER_DIR / name).resolve()
                if not target.is_file() or VIEWER_DIR.resolve() not in target.parents:
                    self.send_error(404)
                    return

                body = target.read_bytes()
                kind = {
                    ".html": "text/html", ".json": "application/json", ".png": "image/png",
                }.get(target.suffix, "application/octet-stream")
                self.send_response(200)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _json(self, payload: dict[str, Any]) -> None:
                body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args: object) -> None:
                # Silent: request logs would drown the training output.
                pass

        # A dashboard from a previous run may still hold the port. Failing loudly here
        # beats binding nothing and leaving the caller pointed at stale data from a run
        # that ended minutes ago.
        if strict:
            # Asked by connecting rather than by binding. Python's HTTP server sets
            # SO_REUSEADDR, and on Windows that lets a second process take a port another
            # is already listening on: the bind succeeds, the lock silently is not one,
            # and two training runs proceed to clear each other's servers.
            if _listening(port):
                raise OSError(
                    f"port {port} is already answering, which almost always means another "
                    f"training run is going. Stop it first, or pass --port to run beside "
                    f"it on its own ports."
                )
            self._server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        else:
            for candidate in range(port, port + 20):
                try:
                    self._server = ThreadingHTTPServer(("127.0.0.1", candidate), Handler)
                    break
                except OSError:
                    continue
            else:
                raise OSError(f"no free port between {port} and {port + 19}")

        thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        thread.start()
        return f"http://127.0.0.1:{self._server.server_address[1]}/"

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server = None
