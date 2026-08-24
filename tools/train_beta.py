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
from gamma.env import MindustryEnv
from gamma.monitor import TrainingMonitor
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
        env = MindustryEnv(
            task,
            server_dir=str(Path(self.args.root) / f"env{self.index}"),
            bridge_port=BRIDGE_PORT_BASE + self.index,
            game_port=GAME_PORT_BASE + self.index,
            jar=jar,
            embodied=self.args.embodied,
        )
        return LocalWindow(env, size=self.args.window), task

    def _run(self) -> None:
        env = None
        try:
            env, task = self._build()
            state = self.monitor.match(self.index)
            state.policy = "beta"
            state.task = task.name
            state.objective = task.description
            state.max_steps = task.max_steps

            observation, info = env.reset()
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
                    episode_reward = 0.0
                    episode_steps = 0
                    state.built = []

                self.results.put((observation, info, reward, done))
        except Exception as error:  # noqa: BLE001
            self.failed = error
            self.monitor.match(self.index).alive = False
            self.ready.set()
            print(f"env {self.index} died: {error}")
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
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--port", type=int, default=8800)
    parser.add_argument("--root", default="mindustry-beta")
    parser.add_argument("--out", type=Path, default=Path("checkpoints"))
    args = parser.parse_args()

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
    print(f"{len(alive)} environments ready")

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

            for worker, env_action in zip(alive, env_actions):
                worker.requests.put(env_action)

            observations, infos, rewards, dones = [], [], [], []
            for worker in alive:
                obs, info, reward, done = worker.results.get()
                observations.append(obs)
                infos.append(info)
                rewards.append(reward)
                dones.append(float(done))

            buffer.rewards[step] = torch.as_tensor(rewards, device=device, dtype=torch.float32)
            buffer.dones[step] = torch.as_tensor(dones, device=device, dtype=torch.float32)
            total_steps += n_envs

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

        snapshot = monitor.snapshot()["totals"]
        elapsed = time.time() - started
        print(
            f"update {agent.updates:4d}  steps {total_steps:>8,}  "
            f"{total_steps / elapsed:6.1f}/s  lr {lr:.2e}  "
            f"policy {stats['policy_loss']:+.4f}  value {stats['value_loss']:.4f}  "
            f"entropy {stats['entropy']:.3f}  scale {stats['reward_scale']:.2f}  "
            f"episodes {snapshot['episodes']}  solved {snapshot['solved']}"
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
