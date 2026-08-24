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
from gamma.monitor import TrainingMonitor
from gamma.replay import ReplayRecorder, _encode_bytes
from tools.extract_sprites import ensure_assets
from gamma.net import PolicyNet
from gamma.ppo import PPO, PPOConfig, RolloutBuffer
from gamma.window import DEFAULT_SIZE, LocalWindow

BRIDGE_PORT_BASE = 7920
GAME_PORT_BASE = 6920


class EnvWorker:
    """One environment on its own thread, driven by a request/response queue pair."""

    def __init__(self, index: int, args, monitor: TrainingMonitor) -> None:
        self.index = index
        self.args = args
        self.monitor = monitor
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

        jar = str(ensure_jar())

        # The watched environment runs at a speed a person can follow. Training speed is
        # several hundred times realtime, which is unwatchable: a ten minute match goes by
        # in a second. One slow environment costs a little throughput and is the only way
        # to actually see what the agent is doing.
        watched = self.index == self.args.watch
        env = MindustryEnv(
            task,
            server_dir=str(Path(self.args.root) / f"env{self.index}"),
            bridge_port=BRIDGE_PORT_BASE + self.index,
            game_port=GAME_PORT_BASE + self.index,
            jar=jar,
            embodied=self.args.embodied,
            speed=str(self.args.watch_speed) if watched else "max",
        )

        # Recorded on the watched match alone. Recording costs a map fetch per episode
        # and a compressed write per step, which is nothing next to one environment and
        # far too much across all of them.
        inner: Any = env
        if watched and self.args.record:
            self.archive = ReplayArchive(Path(self.args.replays) / f"match{self.index}")
            self.recorder = ReplayRecorder(env, self.archive.pending(0))
            self.monitor.register_replays(self.index, self.archive)
            inner = self.recorder

        return LocalWindow(inner, size=self.args.window, channels=self.args.channels), task

    def _archive_episode(self, state, reward: float, solved: bool) -> None:
        """Seal the recording, rank it, and point the next one at a fresh file."""
        if self.archive is None or self.recorder is None:
            return

        self.recorder.finish()
        if self.archive.commit(self.episode, reward, solved) is None:
            self.archive.discard(self.episode)
        self.episode += 1
        self.recorder.path = self.archive.pending(self.episode)
        state.replays = self.archive.listing()

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
            state.watchable = self.index == self.args.watch

            observation, info = env.reset()
            self._capture_terrain(env, state)
            self._capture_scene(env, state, force=True)
            episode_reward = 0.0
            episode_steps = 0
            self.results.put((observation, info))
            self.ready.set()

            while True:
                action = self.requests.get()
                if action is None:
                    break

                observation, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                episode_steps += 1

                raw = info["raw"]
                state.step = episode_steps
                state.tick = float(raw.get("tick", 0))
                state.wave = int(raw.get("wave", 0))
                state.reward = episode_reward
                state.items = dict(raw.get("items", {}))
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
                    self.monitor.record_episode(self.index, episode_reward, solved)
                    self._archive_episode(state, episode_reward, solved)
                    observation, info = env.reset()
                    state.scene.clear()
                    self._capture_terrain(env, state)
                    self._capture_scene(env, state, force=True)
                    episode_reward = 0.0
                    episode_steps = 0
                    state.built = []

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
    """
    agent.save(args.out / "beta.pt")

    if args.save_every > 0 and agent.updates % args.save_every == 0:
        name = f"beta-gen{agent.updates:04d}.pt"
        agent.save(args.out / name)
        monitor.annotate_generation(agent.updates, checkpoint=name)

    mean = generation.get("mean_reward")
    if mean is None:
        return best_mean

    if best_mean is None or mean > best_mean:
        agent.save(args.out / "beta-best.pt")
        monitor.annotate_generation(agent.updates, best=True, checkpoint="beta-best.pt")
        print(f"  new best generation: mean reward {mean:+.3f}", flush=True)
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
    parser.add_argument("--port", type=int, default=8800)
    parser.add_argument("--watch", type=int, default=0,
                        help="index of the match to run at watchable speed, -1 for none")
    parser.add_argument("--watch-speed", type=int, default=2,
                        help="simulation speed for the watched match, 1 is realtime")
    parser.add_argument("--record", action="store_true", default=True,
                        help="record replays of the watched match")
    parser.add_argument("--no-record", dest="record", action="store_false")
    parser.add_argument("--replays", default="replays/live")
    parser.add_argument("--open", action="store_true", default=True,
                        help="open the dashboard in a browser")
    parser.add_argument("--no-open", dest="open", action="store_false")
    parser.add_argument("--scene-fps", type=int, default=10,
                        help="animation frames collected per second and per match")
    parser.add_argument("--root", default="mindustry-beta")
    parser.add_argument("--out", type=Path, default=Path("checkpoints"))
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

    workers = [EnvWorker(i, args, monitor) for i in range(args.envs)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.ready.wait()

    alive = [w for w in workers if w.failed is None]
    if not alive:
        raise SystemExit("no environment started")
    print(f"{len(alive)} environments ready", flush=True)
    print("", flush=True)
    print("=" * 68, flush=True)
    print(f"  DASHBOARD   {url}", flush=True)
    print("              every match animated, versions, replays", flush=True)
    if args.watch >= 0:
        print(f"  IN THE GAME Mindustry -> Play -> Join Game -> "
              f"127.0.0.1:{GAME_PORT_BASE + args.watch}", flush=True)
        print(f"              match #{args.watch} at {args.watch_speed}x, so it is followable",
              flush=True)
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
    print(f"policy on {config.device}, {sum(p.numel() for p in net.parameters()):,} parameters")

    n_envs = len(alive)
    steps = config.steps_per_env
    device = config.device
    args.out.mkdir(parents=True, exist_ok=True)

    total_steps = 0
    started = time.time()
    #: Mean reward of the best generation so far. None until one produces episodes, which
    #: takes a few updates on a task where an episode spans hundreds of steps.
    best_mean: float | None = None

    while total_steps < args.steps:
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
                for worker in casualties:
                    print(f"dropping env {worker.index}", flush=True)
                    monitor.match(worker.index).alive = False
                break

            buffer.rewards[step] = torch.as_tensor(rewards, device=device, dtype=torch.float32)
            buffer.dones[step] = torch.as_tensor(dones, device=device, dtype=torch.float32)
            total_steps += n_envs

        if len(alive) != n_envs or any(w.failed for w in alive):
            alive = [w for w in alive if w.failed is None]
            if not alive:
                print("every environment died, stopping")
                break
            print(f"continuing with {len(alive)} environments")
            n_envs = len(alive)
            first = []
            for worker in alive:
                worker.requests.put(None)
            break

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
            f"entropy {stats['entropy']:.3f}  scale {stats['reward_scale']:.2f}  "
            f"episodes {snapshot['episodes']}  solved {snapshot['solved']}",
            flush=True,
        )



    for worker in alive:
        worker.requests.put(None)
    for worker in alive:
        worker.thread.join(timeout=30)
    monitor.stop()
    print(f"done, saved to {args.out / 'beta.pt'}")


if __name__ == "__main__":
    main()
