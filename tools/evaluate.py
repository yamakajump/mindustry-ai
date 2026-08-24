"""Score a checkpoint on worlds it has never trained on.

    python tools/evaluate.py checkpoints/beta-best.pt

This is the only measurement in the project that can tell learning from memorising. A
reward curve rising on the maps an agent trained on is exactly what memorisation looks
like, so it answers nothing on its own. The sector pool holds a fifth of the planet back
for this, and nothing in a training run ever touches it.

Two policies are scored side by side unless told otherwise: the checkpoint, and a masked
random policy. The random score is the floor. A checkpoint that does not clear it on
unseen worlds has learned the training set, whatever its training curve says.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from gamma import tasks
from gamma.cleanup import kill_servers
from gamma.env import MindustryEnv
from gamma.net import PolicyNet
from gamma.ppo import PPOConfig
from gamma.window import DEFAULT_SIZE, LocalWindow


@dataclass
class Outcome:
    """What one episode came to."""

    sector: int
    reward: float
    steps: int
    wave: int
    built: int
    solved: bool


def summarise(name: str, outcomes: list[Outcome]) -> dict:
    rewards = [o.reward for o in outcomes]
    return {
        "policy": name,
        "episodes": len(outcomes),
        "sectors": len({o.sector for o in outcomes}),
        "mean_reward": round(statistics.fmean(rewards), 3) if rewards else 0.0,
        # The spread matters as much as the mean here: a policy that scores well on a
        # third of the worlds and collapses on the rest has not generalised, and its mean
        # can look identical to one that copes everywhere.
        "reward_spread": round(statistics.pstdev(rewards), 3) if len(rewards) > 1 else 0.0,
        "worst_reward": round(min(rewards), 3) if rewards else 0.0,
        "mean_wave": round(statistics.fmean(o.wave for o in outcomes), 2) if outcomes else 0,
        "mean_built": round(statistics.fmean(o.built for o in outcomes), 1) if outcomes else 0,
        "solved": sum(o.solved for o in outcomes),
    }


def play(env, policy, task, steps: int) -> Outcome:
    """One episode, greedy where the policy is a network."""
    observation, info = env.reset()
    total = 0.0
    raw = info["raw"]

    for step in range(steps):
        action = policy(observation, info)
        observation, reward, terminated, truncated, info = env.step(action)
        raw = info["raw"]
        total += reward
        if terminated or truncated:
            break

    return Outcome(
        sector=env.sector_index if env.sector_index is not None else -1,
        reward=total,
        steps=step + 1,
        wave=int(raw.get("wave", 0)),
        built=int(raw.get("built", 0)),
        solved=bool(task.succeeded(raw)),
    )


def network_policy(net: PolicyNet, device: str):
    """Sample from the policy, masked. Greedy would collapse a stochastic policy onto one
    action type and measure something the agent never does."""

    def act(observation, info):
        with torch.no_grad():
            spatial = torch.as_tensor(observation["spatial"][None], device=device)
            globals_ = torch.as_tensor(observation["global"][None], device=device)
            mask = info.get("action_mask", {})
            masks = {
                "type": torch.as_tensor(mask["type"][None], device=device),
                "block": torch.as_tensor(mask["block"][None], device=device),
                "position": torch.as_tensor(
                    mask["position"].reshape(1, -1), device=device
                ),
            }
            action, _, _, _ = net.act(spatial, globals_, masks)
            return net.to_env_action(action)[0]

    return act


def random_policy(env):
    """The floor. Masked, because an unmasked random policy is refused almost always and
    would set a floor no policy could fail to clear."""

    def act(observation, info):
        mask = info.get("action_mask", {})
        types = np.flatnonzero(mask["type"])
        blocks = np.flatnonzero(mask["block"])
        positions = np.flatnonzero(mask["position"].reshape(-1))
        window = env.size
        flat = int(np.random.choice(positions)) if positions.size else 0
        return np.array([
            int(np.random.choice(types)) if types.size else 0,
            int(np.random.choice(blocks)) if blocks.size else 0,
            flat % window,
            flat // window,
            np.random.randint(4),
        ])

    return act


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, nargs="?")
    parser.add_argument("--task", default="frontier", choices=sorted(tasks.CURRICULUM))
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--window", type=int, default=DEFAULT_SIZE)
    parser.add_argument("--channels", type=int, default=14)
    parser.add_argument("--port", type=int, default=7910)
    parser.add_argument("--root", default="mindustry-eval")
    parser.add_argument("--no-random", dest="random", action="store_false", default=True)
    parser.add_argument("--out", type=Path, default=Path("docs/measurements/generalisation.json"))
    args = parser.parse_args()

    kill_servers()
    jar = str(next((Path("bridge") / "build" / "libs").glob("*.jar")))
    task = tasks.get(args.task)

    env = LocalWindow(
        MindustryEnv(
            task, server_dir=args.root, bridge_port=args.port, game_port=args.port - 1000,
            jar=jar, embodied=True, speed="max", evaluating=True,
        ),
        size=args.window, channels=args.channels,
    )

    contenders = []
    if args.checkpoint is not None:
        observation, info = env.reset()
        net = PolicyNet(
            channels=observation["spatial"].shape[0],
            window=args.window,
            globals_size=observation["global"].shape[0],
            n_types=len(info["action_mask"]["type"]),
            n_blocks=len(info["action_mask"]["block"]),
        )
        device = PPOConfig().device
        net.load_state_dict(torch.load(args.checkpoint, map_location=device)["net"])
        net.to(device).eval()
        contenders.append((args.checkpoint.stem, network_policy(net, device)))
    if args.random:
        contenders.append(("random", random_policy(env)))

    if not contenders:
        raise SystemExit("nothing to evaluate: pass a checkpoint, or keep the random floor")

    started = time.time()
    report = {"task": task.name, "held_out": True, "episodes_each": args.episodes, "results": []}

    for name, policy in contenders:
        outcomes = [play(env, policy, task, args.max_steps) for _ in range(args.episodes)]
        summary = summarise(name, outcomes)
        report["results"].append(summary)
        print(
            f"{summary['policy']:>12}  reward {summary['mean_reward']:+8.3f} "
            f"+-{summary['reward_spread']:6.3f}  worst {summary['worst_reward']:+8.3f}  "
            f"wave {summary['mean_wave']:5.2f}  built {summary['mean_built']:6.1f}  "
            f"solved {summary['solved']}/{summary['episodes']}  "
            f"over {summary['sectors']} unseen sectors",
            flush=True,
        )

    env.close()
    report["seconds"] = round(time.time() - started, 1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
