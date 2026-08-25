"""Train Beta, the first agent that actually learns.

    python tools/train_beta.py

That is the whole command. It cleans up servers a previous run left behind, builds the
bridge if it is missing, starts the environments, opens the dashboard, and trains until
told to stop. Everything else is a flag with a sensible default.

Runs several Mindustry instances in parallel, collects rollouts, and trains a masked
factored-action PPO policy on them. The dashboard on http://127.0.0.1:8800 shows every
match live and animated while it happens, keeps the versions worth comparing, and offers
the recorded replays.

Environments run in threads because each one is a separate process the thread only waits
on. The policy itself runs on the GPU in the main thread.
"""

from __future__ import annotations

import argparse
import queue
import subprocess
import sys
import threading
import time
import webbrowser
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from gamma import tasks
from gamma.archive import ReplayArchive
from gamma.cleanup import kill_servers
from gamma.env import MindustryEnv
from gamma.library import load as load_designs
from gamma.monitor import TrainingMonitor
from gamma.replay import ReplayRecorder, _encode_bytes
from tools.extract_sprites import ensure_assets
from gamma.net import PolicyNet
from gamma.ppo import PPO, PPOConfig, RolloutBuffer
from gamma.window import DEFAULT_SIZE, LocalWindow

#: How many finished episodes a "best so far" claim has to rest on.
#:
#: Thirty is not a statistical ritual; it is roughly ten generations on this task, which
#: is enough that one lucky automated episode moves the average by a few points rather
#: than by a hundred.
MIN_EPISODES_FOR_BEST = 30

BRIDGE_PORT_BASE = 7920
GAME_PORT_BASE = 6920


def idle_speed(workers, speed: str | None) -> None:
    """Slow every server down, or put each back to the speed it was started at.

    Called from the main thread while every worker is parked on its request queue, which
    is the only moment where touching another thread's server is safe. A failure here is
    logged rather than raised: giving up a pause is not worth losing a run over.
    """
    for worker in workers:
        env = worker.env
        if env is None:
            continue
        target = speed if speed is not None else (
            str(worker.args.watch_speed) if worker.showcase else "max"
        )
        try:
            env.set_speed(target)
        except Exception as error:  # noqa: BLE001
            print(f"env {worker.index}: could not set speed to {target}: {error!r}",
                  flush=True)


def showcase_policy(net: PolicyNet, window: int, device: str):
    """Play the current policy, for the one match a person is watching.

    Sampled rather than greedy, and read straight off the live network: the showcase is
    meant to show what the agent is doing right now, weights and all, not a frozen
    snapshot of it and not an idealised version of it. Its experience is never stored, so
    it cannot influence the run it is displaying.
    """

    def act(observation, info):
        with torch.no_grad():
            mask = info.get("action_mask", {})
            action, _, _, _ = net.act(
                torch.as_tensor(observation["spatial"][None], device=device),
                torch.as_tensor(observation["global"][None], device=device),
                {
                    "type": torch.as_tensor(mask["type"][None], device=device),
                    "block": torch.as_tensor(mask["block"][None], device=device),
                    "position": torch.as_tensor(mask["position"].reshape(1, -1), device=device),
                },
            )
            return net.to_env_action(action)[0]

    return act


class EnvWorker:
    """One environment on its own thread, driven by a request/response queue pair."""

    def __init__(self, index: int, args, monitor: TrainingMonitor,
                 showcase: bool = False) -> None:
        self.index = index
        self.args = args
        self.monitor = monitor
        #: True for the one match played at a speed a person can follow.
        #:
        #: It is deliberately **not** part of the training batch. Rollout collection is
        #: lockstep: the trainer hands an action to every environment and waits for all of
        #: them before the next step, so the slowest one sets the pace for the rest. A
        #: match at 2x takes a quarter of a second per step and every other environment
        #: spends that quarter second idle. Measured with sixteen environments: 46.3
        #: steps/s in lockstep behind one watchable match, against a simulation that is
        #: otherwise uncapped. The showcase runs itself instead, on the same policy,
        #: and its experience is thrown away.
        self.showcase = showcase
        #: Set by the trainer once the network exists. Until then the showcase idles.
        self.act = None
        self.stop = threading.Event()
        self.env = None
        #: Wall-clock spacing between animation frames. At training speed a step takes a
        #: couple of milliseconds, so fetching a frame per step would produce hundreds a
        #: second for a browser that can draw ten. The simulation is what should run flat
        #: out, not the sightseeing.
        self.scene_interval = 1.0 / max(1, args.scene_fps)
        self.scene_at = 0.0
        self.archive: ReplayArchive | None = None
        self.recorder: ReplayRecorder | None = None
        #: Which episode the recorder is currently writing. Tracked here rather than read
        #: back from the monitor, whose counter has already moved on by the time an
        #: episode is archived, which orphaned every recording by one.
        self.episode = 0
        self.requests: queue.Queue = queue.Queue(1)
        self.results: queue.Queue = queue.Queue(1)
        self.ready = threading.Event()
        self.failed: Exception | None = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def _build(self):
        task = tasks.get(self.args.task)
        if self.args.max_steps:
            task = replace(task, max_steps=self.args.max_steps)
        if self.args.worlds:
            task = replace(task, worlds=self.args.worlds)

        jar = str(ensure_jar())

        # Structures the search found, handed to the policy as single actions. A line from
        # a drill to the core pays nothing until it is complete, and a policy choosing
        # tiles one at a time never completes one: 5,719 conveyors placed across 177
        # archived episodes and one line that ever met end to end. With a structure as one
        # action the policy stops spelling and starts deciding which patch and how many.
        designs = ()
        if self.args.designs and Path(self.args.designs).is_file():
            designs = tuple(load_designs(Path(self.args.designs)))

        # The showcase runs at a speed a person can follow. Training speed is uncapped,
        # which is unwatchable: a ten minute match goes by in a moment. It is a separate
        # environment rather than one of the training ones precisely because a slow one in
        # the batch would hold every other environment at its own pace.
        watched = self.showcase
        env = MindustryEnv(
            task,
            server_dir=str(Path(self.args.root) / f"env{self.index}"),
            bridge_port=BRIDGE_PORT_BASE + self.index,
            game_port=GAME_PORT_BASE + self.index,
            jar=jar,
            embodied=self.args.embodied,
            speed=str(self.args.watch_speed) if watched else "max",
            designs=designs,
            window=self.args.window,
        )

        # Every match records, not just the one being watched.
        #
        # Recording only the watched match records the wrong episodes. The watched match
        # is the one a person happens to be looking at, and the episodes worth keeping are
        # the ones that went well, wherever they happened. Measured on a real run: the best
        # episode scored 392 with 2,739 ore automated, and it was never written to disk,
        # while the two files that were kept scored -38 and -29.
        #
        # The archive keeps the best five, the latest three and every solved episode per
        # match, so twenty-five of them cost single-digit megabytes rather than growing
        # without bound.
        inner: Any = env
        if self.args.record:
            self.archive = ReplayArchive(Path(self.args.replays) / f"match{self.index}")
            self.recorder = ReplayRecorder(env, self.archive.pending(0))
            self.monitor.register_replays(self.index, self.archive)
            inner = self.recorder

        # Kept so the run can hand the machine back while it is paused. The wrapper is
        # what the trainer steps; the speed lives on the environment underneath it.
        self.env = env
        return LocalWindow(inner, size=self.args.window, channels=self.args.channels), task

    def _archive_episode(self, state, reward: float, solved: bool) -> None:
        """Seal the recording, rank it, and point the next one at a fresh file.

        Nothing in here is allowed to take the environment down with it. A replay is a
        side artefact: it is read afterwards, by people and by tools, and losing one costs
        an episode of hindsight and nothing else. An environment is hours of training.

        Windows makes this concrete. Renaming or deleting a file another process has open
        raises PermissionError, and the archive both renames and prunes, so anything that
        reads the archive while a run is going can collide with it. Something did, and the
        environment died of it, and at the time a dead environment ended the entire run:
        608,256 steps of a five million step budget, gone, with exit code zero.
        """
        if self.archive is None or self.recorder is None:
            return

        try:
            self.recorder.finish()
            if self.archive.commit(self.episode, reward, solved) is None:
                self.archive.discard(self.episode)
        except OSError as error:
            print(f"env {self.index}: could not archive episode {self.episode}, "
                  f"carrying on: {error!r}", flush=True)

        self.episode += 1
        try:
            self.recorder.path = self.archive.pending(self.episode)
            state.replays = self.archive.listing()
        except OSError as error:
            print(f"env {self.index}: archive unreadable, recording paused: {error!r}",
                  flush=True)

    def _capture_scene(self, env, state, force: bool = False) -> None:
        """Fetch what moved and fold it into the match, on a wall-clock schedule."""
        now = time.monotonic()
        if not force and now - self.scene_at < self.scene_interval:
            return
        self.scene_at = now
        try:
            state.scene.apply(env._bridge.scene())
        except Exception:
            # A dropped frame is a stutter in a dashboard, not a reason to lose the
            # environment it came from. The next one resynchronises anyway.
            pass

    def _capture_terrain(self, env, state) -> None:
        """Fetch the typed map once per episode so the dashboard can draw the real thing.

        Costs one extra request per episode, against tens of kilobytes that would
        otherwise have to ride along with every half-second poll.
        """
        try:
            typed = env._bridge.map()
        except Exception as error:
            # Swallowed, this leaves the viewer holding the map of an earlier episode while
            # the scene describes the current one, and it draws the agent and everything it
            # built outside a world that is not the one it is in.
            print(f"env {self.index}: could not fetch the map, the view will lag: {error!r}",
                  flush=True)
            return

        planes = typed["spatial"]
        tiles = typed["width"] * typed["height"]
        state.terrain = {
            "width": typed["width"],
            "height": typed["height"],
            "palette": typed["palette"],
            "floor": _encode_bytes(planes[0:tiles * 2]),
            "overlay": _encode_bytes(planes[tiles * 2:tiles * 4]),
            "block": _encode_bytes(planes[tiles * 4:tiles * 6]),
        }
        state.terrain_version += 1
        state.terrain_size = [typed["width"], typed["height"]]

    def _run(self) -> None:
        env = None
        try:
            env, task = self._build()
            state = self.monitor.match(self.index)
            state.policy = "beta"
            state.task = task.name
            state.objective = task.description
            state.max_steps = task.max_steps
            state.game_port = GAME_PORT_BASE + self.index
            state.watchable = self.showcase

            observation, info = env.reset()
            self._capture_terrain(env, state)
            self._capture_scene(env, state, force=True)
            episode_reward = 0.0
            episode_steps = 0
            self.results.put((observation, info))
            self.ready.set()

            while not self.stop.is_set():
                if self.showcase:
                    if self.act is None:
                        # The policy is built from the first observation, which this
                        # worker has only just produced. A short wait beats a busy loop.
                        time.sleep(0.2)
                        continue
                    action = self.act(observation, info)
                else:
                    action = self.requests.get()
                    if action is None:
                        break

                observation, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                episode_steps += 1

                raw = info["raw"]
                state.step = episode_steps
                state.total_steps += 1
                state.tick = float(raw.get("tick", 0))
                state.wave = int(raw.get("wave", 0))
                state.reward = episode_reward
                state.items = dict(raw.get("items", {}))
                state.produced = sum(int(a) for a in raw.get("produced", {}).values())
                # Read off the current observation rather than accumulated: every
                # milestone counter is cumulative within an episode, so the latest one
                # already holds everything reached so far.
                state.reached = [
                    stone.name for stone in tasks.MILESTONES
                    if stone.read(raw) >= stone.threshold
                ]
                state.unit = dict(raw.get("unit", {}))
                state.action = env.action_types[int(action[0])]
                state.core = [int(raw.get("core_x", -1)), int(raw.get("core_y", -1))]
                state.size = [int(raw.get("map_width", 0)), int(raw.get("map_height", 0))]
                outcome = info.get("action") or {}
                if outcome:
                    state.applied += bool(outcome.get("applied"))
                    state.refused += not outcome.get("applied")

                self._capture_scene(env, state)

                done = terminated or truncated
                if done:
                    solved = task.succeeded(raw)
                    self.monitor.record_episode(
                        self.index, episode_reward, solved, state.reached
                    )
                    self._archive_episode(state, episode_reward, solved)
                    observation, info = env.reset()
                    state.scene.clear()
                    self._capture_terrain(env, state)
                    self._capture_scene(env, state, force=True)
                    episode_reward = 0.0
                    episode_steps = 0
                    state.built = []
                    state.produced = 0
                    state.reached = []

                if not self.showcase:
                    self.results.put((observation, info, reward, done))
        except Exception as error:  # noqa: BLE001
            self.failed = error
            self.monitor.match(self.index).alive = False
            self.ready.set()
            print(f"env {self.index} died: {error!r}", flush=True)
            # Unblock whoever is waiting on this worker. Without it the main loop waits
            # for a reply that will never come, every other environment sits idle behind
            # it, and the run looks frozen with the CPU nearly idle. Observed after ~3,500
            # steps, with no error visible because stdout was still buffered.
            try:
                self.results.put_nowait(None)
            except Exception:
                pass
        finally:
            # Without this every run leaves its Mindustry servers alive: they hold their
            # ports, so the next run cannot bind them, and they quietly eat memory. An
            # earlier session left eighteen of them running.
            if self.recorder is not None:
                # Closed before the file is deleted. Windows refuses to unlink an open
                # file, and the refusal came out of a `finally`, so it took the rest of
                # the cleanup with it and leaked the server.
                self.recorder.finish()
            if self.archive is not None:
                # The episode in progress when a run stops has no result, so its partial
                # recording is dropped rather than left to be mistaken for a real one.
                self.archive.discard(self.episode)
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass


def stack_masks(infos, window: int, n_types: int, n_blocks: int) -> dict[str, np.ndarray]:
    """Collect per-environment masks into batched arrays the network can use."""
    types, blocks, positions = [], [], []
    for info in infos:
        mask = info.get("action_mask", {})
        types.append(mask.get("type", np.ones(n_types, bool)))
        blocks.append(mask.get("block", np.ones(n_blocks, bool)))

        position = mask.get("position")
        if position is None:
            position = np.ones((window, window), bool)
        positions.append(position.reshape(-1))

    return {
        "type": np.array(types),
        "block": np.array(blocks),
        "position": np.array(positions),
    }


def ensure_jar() -> Path:
    """The bridge plugin, built if it is not there yet.

    A fresh clone has no jar, and the failure without this is a StopIteration from a glob
    deep inside a worker thread, which says nothing about the missing build step.
    """
    libs = Path("bridge") / "build" / "libs"
    existing = sorted(libs.glob("*.jar"))
    if existing:
        return existing[0]

    print("building the bridge plugin (first run only)...", flush=True)
    gradle = "gradlew.bat" if sys.platform == "win32" else "./gradlew"
    result = subprocess.run([gradle, "jar", "-q"], cwd="bridge", capture_output=True, text=True)
    built = sorted(libs.glob("*.jar"))
    if not built:
        raise SystemExit(
            "could not build the bridge plugin. Java 17 is required. "
            + (result.stderr or result.stdout).strip()
        )
    return built[0]


def save_checkpoints(agent, monitor, generation, best_mean, args) -> float | None:
    """Write the weights, keeping the versions worth comparing.

    Three files answer three different questions. `beta.pt` is what a run resumes from.
    `beta-best.pt` is the one to play against, and it is chosen on the mean across a
    generation rather than on a single episode: the luckiest episode of a mediocre policy
    routinely beats the average episode of a good one, so ranking on the best episode
    promotes noise. `beta-genNNNN.pt` is the trail, which is the only way to answer
    whether the agent is better than it was an hour ago by actually running both.

    A generation's mean is not enough on its own either. An episode here runs for
    thousands of steps, so a generation closes two to eight of them, and on this task one
    that automates production scores about +110 against about -35 for one that does not.
    A single lucky episode therefore moves the mean by a hundred points and a checkpoint
    picked on it is a checkpoint picked on luck. Measured: `beta-best.pt` chosen that way
    scored -9.5 on held-out worlds against -9.2 for a masked random policy. So a
    candidate has to be built from enough episodes to mean anything.
    """
    agent.save(args.out / "beta.pt")

    if args.save_every > 0 and agent.updates % args.save_every == 0:
        name = f"beta-gen{agent.updates:04d}.pt"
        agent.save(args.out / name)
        monitor.annotate_generation(agent.updates, checkpoint=name)

    mean = monitor.recent_mean(episodes=MIN_EPISODES_FOR_BEST)
    if mean is None:
        return best_mean

    if best_mean is None or mean > best_mean:
        agent.save(args.out / "beta-best.pt")
        monitor.annotate_generation(agent.updates, best=True, checkpoint="beta-best.pt")
        print(f"  new best: {mean:+.3f} mean over the last {MIN_EPISODES_FOR_BEST}+ episodes",
              flush=True)
        return mean

    return best_mean


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envs", type=int, default=6)
    parser.add_argument("--steps", type=int, default=100_000, help="total environment steps")
    parser.add_argument("--task", default="T1_copper", choices=sorted(tasks.CURRICULUM))
    # On by default: an agent that edits the world from nowhere is not playing the game.
    # The disembodied mode stays reachable because it trains far faster and serves as an
    # upper bound on what the embodied one should manage.
    parser.add_argument("--embodied", action="store_true", default=True)
    parser.add_argument("--direct", dest="embodied", action="store_false",
                        help="edit the world directly instead of flying a unit")
    parser.add_argument("--window", type=int, default=DEFAULT_SIZE)
    parser.add_argument("--channels", type=int, default=14,
                        help="observation channels, pinned so parallel envs agree")
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--worlds", type=int, default=0,
                        help="train across at most N worlds instead of the whole pool; "
                             "the held-out set is unaffected either way")
    parser.add_argument("--port", type=int, default=8800)
    parser.add_argument("--watch", action="store_true", default=True,
                        help="play one extra match at a speed a person can follow")
    parser.add_argument("--no-watch", dest="watch", action="store_false",
                        help="train only; nothing to join in the game")
    parser.add_argument("--watch-speed", type=int, default=2,
                        help="simulation speed for the watched match, 1 is realtime")
    parser.add_argument("--record", action="store_true", default=True,
                        help="archive each match's best episodes")
    parser.add_argument("--no-record", dest="record", action="store_false")
    parser.add_argument("--replays", default="replays/live")
    parser.add_argument("--open", action="store_true", default=True,
                        help="open the dashboard in a browser")
    parser.add_argument("--no-open", dest="open", action="store_false")
    parser.add_argument("--scene-fps", type=int, default=10,
                        help="animation frames collected per second and per match")
    parser.add_argument("--root", default="mindustry-beta")
    parser.add_argument("--out", type=Path, default=Path("checkpoints"))
    parser.add_argument("--designs", default="docs/designs.json",
                        help="structures discovered by the forge, offered to "
                             "the policy as single actions; pass an empty string to train "
                             "on the primitives alone")
    parser.add_argument("--resume", action="store_true", default=True,
                        help="continue from checkpoints/beta.pt when it exists")
    parser.add_argument("--fresh", dest="resume", action="store_false",
                        help="start from a new policy, ignoring any checkpoint")
    parser.add_argument("--save-every", type=int, default=10,
                        help="keep a numbered checkpoint every N generations, 0 to disable")
    args = parser.parse_args()

    # The dashboard port is claimed first, and it doubles as the lock on the whole run.
    # Two runs at once share the same server directories and the same bridge ports, so
    # each clears the other's servers and reloads worlds underneath it, while the
    # dashboard describes a run that is no longer the one doing the work. Claiming the
    # port before anything is destroyed turns that into a refusal.
    monitor = TrainingMonitor(title=f"beta / {args.task}")
    url = monitor.serve(args.port, strict=True)

    # A killed run leaves its servers holding the ports, and the next one then
    # fails with "no environment started" and no hint as to why.
    kill_servers()

    ensure_jar()
    # A dashboard of grey squares is a fresh clone missing its sprites, and there is no
    # way to guess that from looking at it.
    ensure_assets()
    print(f"dashboard: {url}", flush=True)
    if args.open:
        # Opened before the environments start: they take the better part of a minute to
        # come up, and an empty dashboard that fills in beats a blank terminal that looks
        # like nothing is happening.
        webbrowser.open(url)

    # The showcase is an extra environment, not one of the training ones. `--envs 16`
    # means sixteen environments feeding the learner, plus one more to look at.
    showcase_index = args.envs if args.watch else -1
    workers = [EnvWorker(i, args, monitor) for i in range(args.envs)]
    if args.watch:
        workers.append(EnvWorker(showcase_index, args, monitor, showcase=True))
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.ready.wait()

    showcase = next((w for w in workers if w.showcase and w.failed is None), None)
    alive = [w for w in workers if w.failed is None and not w.showcase]
    if not alive:
        raise SystemExit("no environment started")
    print(f"{len(alive)} environments ready", flush=True)
    print("", flush=True)
    print("=" * 68, flush=True)
    print(f"  DASHBOARD   {url}", flush=True)
    print("              every match animated, versions, replays", flush=True)
    if showcase is not None:
        print(f"  IN THE GAME Mindustry -> Play -> Join Game -> "
              f"127.0.0.1:{GAME_PORT_BASE + showcase.index}", flush=True)
        print(f"              match #{showcase.index} at {args.watch_speed}x, alongside the "
              f"training run rather than inside it", flush=True)
    print("=" * 68, flush=True)
    print("", flush=True)

    first = [w.results.get() for w in alive]
    observations = [obs for obs, _ in first]
    infos = [info for _, info in first]

    sample = observations[0]
    config = PPOConfig()
    net = PolicyNet(
        channels=sample["spatial"].shape[0],
        window=args.window,
        globals_size=sample["global"].shape[0],
        n_types=len(infos[0]["action_mask"]["type"]),
        n_blocks=len(infos[0]["action_mask"]["block"]),
    )
    agent = PPO(net, config)

    # `beta.pt` was documented as the file a run resumes from, and nothing ever loaded it.
    # Stopping a run to look at a replay therefore threw away every generation it had, so
    # it is loaded by default when it is there, and `--fresh` is how you say you meant to
    # start over.
    start = args.out / "beta.pt"
    if args.resume and start.exists():
        agent.load(start)
        print(f"resuming from {start} at generation {agent.updates}")

    print(f"policy on {config.device}, {sum(p.numel() for p in net.parameters()):,} parameters")

    if showcase is not None:
        showcase.act = showcase_policy(net, args.window, config.device)

    n_envs = len(alive)
    steps = config.steps_per_env
    device = config.device
    args.out.mkdir(parents=True, exist_ok=True)

    total_steps = 0
    started = time.time()
    #: Mean reward of the best generation so far. None until one produces episodes, which
    #: takes a few updates on a task where an episode spans hundreds of steps.
    best_mean: float | None = None

    while total_steps < args.steps and not monitor.stopping.is_set():
        # Checked between updates rather than between steps. A rollout half collected is
        # not a rollout, and the environments are frozen the moment nobody asks them for a
        # step, so the pause lands within a couple of seconds either way.
        if not monitor.running.is_set():
            # Handed back rather than merely idled. At uncapped speed the engine's frame
            # budget is zero and its loop never sleeps, so twenty-four servers with
            # nothing to do held the machine at 99% through a pause whose whole purpose
            # was to free it. Realtime gives it back; a paused world costs nothing to
            # simulate anyway.
            idle_speed(workers, "1")
            print("paused; resume from the dashboard", flush=True)
            monitor.running.wait()
            idle_speed(workers, None)
            if monitor.stopping.is_set():
                break
            print("resumed", flush=True)

        buffer = RolloutBuffer(
            spatial=torch.zeros((steps, n_envs, *sample["spatial"].shape), dtype=torch.uint8, device=device),
            globals_=torch.zeros((steps, n_envs, sample["global"].shape[0]), device=device),
            actions=torch.zeros((steps, n_envs, 4), dtype=torch.long, device=device),
            log_probs=torch.zeros((steps, n_envs), device=device),
            rewards=torch.zeros((steps, n_envs), device=device),
            dones=torch.zeros((steps, n_envs), device=device),
            values=torch.zeros((steps, n_envs), device=device),
            masks={
                "type": torch.zeros((steps, n_envs, net.n_types), dtype=torch.bool, device=device),
                "block": torch.zeros((steps, n_envs, net.n_blocks), dtype=torch.bool, device=device),
                "position": torch.zeros((steps, n_envs, args.window ** 2), dtype=torch.bool, device=device),
            },
        )

        for step in range(steps):
            batch = {
                "spatial": np.stack([o["spatial"] for o in observations]),
                "global": np.stack([o["global"] for o in observations]),
            }
            masks = stack_masks(infos, args.window, net.n_types, net.n_blocks)

            action, log_prob, _, value = agent.act(batch, masks)
            env_actions = net.to_env_action(action)

            buffer.spatial[step] = torch.as_tensor(batch["spatial"], device=device)
            buffer.globals_[step] = torch.as_tensor(batch["global"], device=device)
            buffer.actions[step] = action
            buffer.log_probs[step] = log_prob
            buffer.values[step] = value
            for key, value_array in masks.items():
                buffer.masks[key][step] = torch.as_tensor(value_array, device=device)

            previous_observations = list(observations)
            previous_infos = list(infos)
            for worker, env_action in zip(alive, env_actions):
                try:
                    worker.requests.put(env_action, timeout=30)
                except queue.Full:
                    print(f"env {worker.index} is not consuming actions")

            observations, infos, rewards, dones = [], [], [], []
            casualties = []
            for slot, worker in enumerate(alive):
                try:
                    result = worker.results.get(timeout=180)
                except queue.Empty:
                    result = None

                if result is None:
                    # Dead or hung: keep its last observation so the batch stays the right
                    # shape for this step, and drop it before the next one.
                    casualties.append(worker)
                    observations.append(previous_observations[slot])
                    infos.append(previous_infos[slot])
                    rewards.append(0.0)
                    dones.append(1.0)
                    continue

                obs, info, reward, done = result
                observations.append(obs)
                infos.append(info)
                rewards.append(reward)
                dones.append(float(done))

            if casualties:
                dead = {id(worker) for worker in casualties}
                for worker in casualties:
                    print(f"dropping env {worker.index}", flush=True)
                    monitor.match(worker.index).alive = False
                # Drop the dead slots from this step's lists too, or the survivors are
                # renumbered against observations that still belong to the dead.
                keep = [i for i, worker in enumerate(alive) if id(worker) not in dead]
                observations = [observations[i] for i in keep]
                infos = [infos[i] for i in keep]
                alive = [alive[i] for i in keep]
                break

            buffer.rewards[step] = torch.as_tensor(rewards, device=device, dtype=torch.float32)
            buffer.dones[step] = torch.as_tensor(dones, device=device, dtype=torch.float32)
            total_steps += n_envs

        if len(alive) != n_envs or any(w.failed for w in alive):
            # It used to print "continuing with N environments" and then shut every worker
            # down and leave the loop, which is the opposite of continuing. A single
            # environment dying ended the whole run, and because the message said the
            # opposite and the exit code was zero, it read as a run that had finished. One
            # did: it stopped at 608,256 steps of a five million step budget, 12% of it,
            # killed by a Windows file lock on a replay while it was being archived.
            #
            # Continuing for real costs nothing here. The rollout buffer is rebuilt from
            # `n_envs` at the top of every iteration, and the network has no idea how many
            # environments feed it, so the only thing to repair is the bookkeeping.
            keep = [i for i, worker in enumerate(alive) if worker.failed is None]
            observations = [observations[i] for i in keep]
            infos = [infos[i] for i in keep]
            alive = [alive[i] for i in keep]
            if not alive:
                print("every environment died, stopping", flush=True)
                break
            n_envs = len(alive)
            # The half-collected rollout goes in the bin. A rollout is a fixed number of
            # steps from a fixed number of environments; there is no honest way to train
            # on one that changed width halfway through.
            print(f"continuing with {n_envs} environments, discarding the partial rollout",
                  flush=True)
            continue

        with torch.no_grad():
            last_value = agent.net.value(
                torch.as_tensor(np.stack([o["spatial"] for o in observations]), device=device),
                torch.as_tensor(np.stack([o["global"] for o in observations]), device=device),
            )

        if agent.updates == 0:
            # One-off check that what reached the buffer is what the masks said. An
            # all-illegal or all-legal buffer produces a policy that looks trained and
            # has learned nothing, and the two are indistinguishable from the loss alone.
            for name, tensor in buffer.masks.items():
                legal = tensor.float().mean().item()
                print(f"  mask {name}: {legal:.1%} legal, shape {tuple(tensor.shape)}")
            print(f"  rewards: min {buffer.rewards.min():.4f} max {buffer.rewards.max():.4f} "
                  f"nonzero {(buffer.rewards != 0).float().mean():.1%}")

        lr = agent.set_learning_rate(total_steps / args.steps)
        stats = agent.update(buffer, last_value)
        generation = monitor.record_generation(agent.updates, stats)
        best_mean = save_checkpoints(agent, monitor, generation, best_mean, args)

        snapshot = monitor.snapshot()["totals"]
        elapsed = time.time() - started
        print(
            f"update {agent.updates:4d}  steps {total_steps:>8,}  "
            f"{total_steps / elapsed:6.1f}/s  lr {lr:.2e}  "
            f"policy {stats['policy_loss']:+.4f}  value {stats['value_loss']:.4f}  "
            # Per head, because the summed entropy hides everything. The position head
            # ranges over 2,304 tiles and its ln swamps the ln(6) of the type, so a policy
            # that has learnt exactly what to build and nothing about where reads as flat.
            f"H[t {stats['entropy_type']:.2f} b {stats['entropy_block']:.2f} "
            f"p {stats['entropy_position']:.2f} r {stats['entropy_rotation']:.2f}]  "
            f"grad[p {stats['grad_policy']:.3f} H {stats['grad_entropy']:.3f} "
            f"tot {stats['grad_total']:.2f}/{PPOConfig.max_grad_norm}]  "
            f"EV {stats['explained_variance']:+.2f}  clip {stats['clip_fraction']:.2f}  "
            f"scale {stats['reward_scale']:.2f}  "
            f"episodes {snapshot['episodes']}  solved {snapshot['solved']}",
            flush=True,
        )



    # Saved on the way out whatever ended the run: a budget reached, a stop from the
    # dashboard, or the last environment dying. Losing an hour of training because the
    # exit went through a different branch is not a failure mode worth keeping.
    agent.save(args.out / "beta.pt")
    print(f"saved generation {agent.updates} to {args.out / 'beta.pt'}", flush=True)

    for worker in workers:
        worker.stop.set()
    for worker in alive:
        worker.requests.put(None)
    for worker in workers:
        worker.thread.join(timeout=30)
    monitor.stop()
    print(f"done, saved to {args.out / 'beta.pt'}")


if __name__ == "__main__":
    main()
