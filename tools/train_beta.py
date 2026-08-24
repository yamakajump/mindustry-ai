"""Train Beta, the first agent that actually learns.

    python tools/train_beta.py --envs 8 --steps 200000 --embodied

Runs several Mindustry instances in parallel, collects rollouts, and trains a masked
factored-action PPO policy on them. The dashboard on http://127.0.0.1:8800 shows every
match live while it happens.

Environments run in threads because each one is a separate process the thread only waits
on. The policy itself runs on the GPU in the main thread.
"""

from __future__ import annotations

import argparse
import queue
import threading
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from gamma import tasks
from gamma.cleanup import kill_servers
from gamma.env import MindustryEnv
from gamma.monitor import TrainingMonitor
from gamma.replay import _encode_bytes
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

        jar = str(next((Path("bridge") / "build" / "libs").glob("*.jar")))

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
        return LocalWindow(env, size=self.args.window, channels=self.args.channels), task

    def _capture_terrain(self, env, state) -> None:
        """Fetch the typed map once per episode so the dashboard can draw the real thing.

        Costs one extra request per episode, against tens of kilobytes that would
        otherwise have to ride along with every half-second poll.
        """
        try:
            typed = env._bridge.map()
        except Exception:
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

                done = terminated or truncated
                if done:
                    self.monitor.record_episode(self.index, episode_reward, task.succeeded(raw))
                    observation, info = env.reset()
                    self._capture_terrain(env, state)
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envs", type=int, default=6)
    parser.add_argument("--steps", type=int, default=100_000, help="total environment steps")
    parser.add_argument("--task", default="T1_copper", choices=sorted(tasks.CURRICULUM))
    parser.add_argument("--embodied", action="store_true")
    parser.add_argument("--window", type=int, default=DEFAULT_SIZE)
    parser.add_argument("--channels", type=int, default=14,
                        help="observation channels, pinned so parallel envs agree")
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--port", type=int, default=8800)
    parser.add_argument("--watch", type=int, default=0,
                        help="index of the match to run at watchable speed, -1 for none")
    parser.add_argument("--watch-speed", type=int, default=2,
                        help="simulation speed for the watched match, 1 is realtime")
    parser.add_argument("--root", default="mindustry-beta")
    parser.add_argument("--out", type=Path, default=Path("checkpoints"))
    args = parser.parse_args()

    # A killed run leaves its servers holding the ports, and the next one then
    # fails with "no environment started" and no hint as to why.
    kill_servers()

    monitor = TrainingMonitor(title=f"beta / {args.task}")
    print(f"dashboard: {monitor.serve(args.port)}")

    workers = [EnvWorker(i, args, monitor) for i in range(args.envs)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.ready.wait()

    alive = [w for w in workers if w.failed is None]
    if not alive:
        raise SystemExit("no environment started")
    print(f"{len(alive)} environments ready", flush=True)
    if args.watch >= 0:
        print("", flush=True)
        print("=" * 64, flush=True)
        print(f"  WATCH IT LIVE IN THE REAL GAME", flush=True)
        print(f"  Mindustry -> Play -> Join Game -> 127.0.0.1:{GAME_PORT_BASE + args.watch}", flush=True)
        print(f"  (match #{args.watch}, running at {args.watch_speed}x so it is followable)", flush=True)
        print("=" * 64, flush=True)
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
        monitor.record_generation(agent.updates, stats)

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

        agent.save(args.out / "beta.pt")

    for worker in alive:
        worker.requests.put(None)
    for worker in alive:
        worker.thread.join(timeout=30)
    monitor.stop()
    print(f"done, saved to {args.out / 'beta.pt'}")


if __name__ == "__main__":
    main()
