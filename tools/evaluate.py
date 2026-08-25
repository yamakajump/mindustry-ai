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
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from gamma import tasks
from gamma.env import MindustryEnv
from gamma.library import load as load_designs
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
    #: Ore a machine delivered to the core. Hand mining is not counted, by the engine.
    produced: int
    #: Milestones the episode reached, by name.
    reached: frozenset[str]
    #: Whether the core was still standing when the episode ended.
    survived: bool


def summarise(name: str, outcomes: list[Outcome]) -> dict:
    rewards = [o.reward for o in outcomes]
    return {
        "policy": name,
        "episodes": len(outcomes),
        "sectors": len({o.sector for o in outcomes}),
        "mean_reward": round(statistics.fmean(rewards), 3) if rewards else 0.0,
        # The uncertainty on that mean, which is what a comparison has to be read against.
        # The spread alone invites the mistake this file made twice: two runs of the same
        # measurement named opposite winners, at 1.3 and 0.7 standard deviations, and both
        # were reported as results.
        "standard_error": round(
            statistics.pstdev(rewards) / math.sqrt(len(rewards)), 3
        ) if len(rewards) > 1 else 0.0,
        # The spread matters as much as the mean here: a policy that scores well on a
        # third of the worlds and collapses on the rest has not generalised, and its mean
        # can look identical to one that copes everywhere.
        "reward_spread": round(statistics.pstdev(rewards), 3) if len(rewards) > 1 else 0.0,
        "worst_reward": round(min(rewards), 3) if rewards else 0.0,
        "mean_wave": round(statistics.fmean(o.wave for o in outcomes), 2) if outcomes else 0,
        "mean_built": round(statistics.fmean(o.built for o in outcomes), 1) if outcomes else 0,
        "mean_produced": round(statistics.fmean(o.produced for o in outcomes), 1) if outcomes else 0,
        "solved": sum(o.solved for o in outcomes),
        "survived": round(
            statistics.fmean(o.survived for o in outcomes), 3) if outcomes else 0.0,
        # How often each rung of the ladder was reached, which says far more than a mean
        # reward does. A policy that never once reaches `automation` has not started
        # playing the game, whatever its score.
        "milestones": {
            stone.name: round(
                sum(stone.name in o.reached for o in outcomes) / len(outcomes), 2
            )
            for stone in tasks.MILESTONES
        } if outcomes else {},
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
        produced=sum(int(a) for a in raw.get("produced", {}).values()),
        # Asked of the task, not of the observation: the automation and variety rungs
        # read the reward's ledger, which no observation carries.
        reached=tasks.reached(task, raw),
        #: Whether the base was still standing. Half of what a run is judged on, and it
        #: was not being measured at all.
        survived=bool(raw.get("has_core", False)),
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
                # Without this the network falls back to the single position mask, which
                # cannot be right for more than one action type at a time. A policy
                # evaluated that way is not the policy that was trained.
                "position_sets": torch.as_tensor(
                    mask["position_sets"].reshape(1, mask["position_sets"].shape[0], -1),
                    device=device,
                ),
            }
            action, _, _, _ = net.act(spatial, globals_, masks)
            return net.to_env_action(action)[0]

    return act


def random_policy(env, seed: int = 20260825):
    """The floor. Masked, because an unmasked random policy is refused almost always and
    would set a floor no policy could fail to clear.

    Seeded, which it was not, and that mattered more than it sounds. Drawing from the
    global generator made the floor different on every run, so the same policy on the same
    held-out sectors was compared against a different opponent each time. Measured: the
    unseeded floor delivered 18.7 ore in one run and 0 in the next, on the same worlds, and
    the two runs disagreed about which policy won.
    """
    rng = np.random.default_rng(seed)

    def act(observation, info):
        mask = info.get("action_mask", {})
        types = np.flatnonzero(mask["type"])
        blocks = np.flatnonzero(mask["block"])
        positions = np.flatnonzero(mask["position"].reshape(-1))
        window = env.size
        flat = int(rng.choice(positions)) if positions.size else 0
        return np.array([
            int(rng.choice(types)) if types.size else 0,
            int(rng.choice(blocks)) if blocks.size else 0,
            flat % window,
            flat // window,
            rng.integers(4),
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
    parser.add_argument("--designs", default="docs/designs.json",
                        help="the structures the policy was trained with. A checkpoint "
                             "trained with them has a wider action head, so evaluating it "
                             "without them fails on a shape mismatch rather than on a "
                             "score, and a checkpoint trained without them must be "
                             "evaluated with an empty string here")
    parser.add_argument("--no-random", dest="random", action="store_false", default=True)
    parser.add_argument("--out", type=Path, default=Path("docs/measurements/generalisation.json"))
    args = parser.parse_args()

    # Deliberately no server cleanup: this runs on its own ports and its own directory,
    # and is meant to be usable while a training run is going. Killing every Mindustry
    # process would take the run down with it.
    jar = str(next((Path("bridge") / "build" / "libs").glob("*.jar")))
    task = tasks.get(args.task)

    designs = ()
    if args.designs and Path(args.designs).is_file():
        designs = tuple(load_designs(Path(args.designs)))

    env = LocalWindow(
        MindustryEnv(
            task, server_dir=args.root, bridge_port=args.port, game_port=args.port - 1000,
            jar=jar, embodied=True, speed="max", evaluating=True, designs=designs,
            window=args.window,
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
            # The action layout, asked of the environment being evaluated on. A network
            # built without it treats every action type as aiming at the same tiles, which
            # is what the trained weights were not trained to do.
            position_set_of_type=env.position_set_of_type,
            block_of_type=env.block_of_type,
            rotation_of_type=env.rotation_of_type,
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
    report = {"task": task.name, "held_out": True, "episodes_each": args.episodes,
              "designs": len(designs), "results": []}

    for name, policy in contenders:
        outcomes = [play(env, policy, task, args.max_steps) for _ in range(args.episodes)]
        summary = summarise(name, outcomes)
        report["results"].append(summary)
        print(
            f"{summary['policy']:>12}  reward {summary['mean_reward']:+8.3f} "
            f"+-{summary['reward_spread']:6.3f}  worst {summary['worst_reward']:+8.3f}  "
            f"wave {summary['mean_wave']:5.2f}  built {summary['mean_built']:6.1f}  "
            f"produced {summary['mean_produced']:8.1f}  "
            f"solved {summary['solved']}/{summary['episodes']}  "
            f"core tenu {summary['survived']:.0%}  "
            f"over {summary['sectors']} unseen sectors",
            flush=True,
        )
        reached = [f"{n} {r:.0%}" for n, r in summary["milestones"].items() if r > 0]
        print(f"{'':>12}  {' | '.join(reached) if reached else 'no milestone reached'}",
              flush=True)

    env.close()
    report["seconds"] = round(time.time() - started, 1)

    # The only line that answers the question this tool exists for, and it says "not
    # enough evidence" far more often than a mean reward suggests.
    if len(report["results"]) == 2:
        first, second = report["results"]
        gap = first["mean_reward"] - second["mean_reward"]
        uncertainty = math.hypot(first["standard_error"], second["standard_error"])
        sigmas = abs(gap) / uncertainty if uncertainty else 0.0
        report["verdict"] = {
            "gap": round(gap, 3), "uncertainty": round(uncertainty, 3),
            "sigmas": round(sigmas, 2),
            "conclusive": sigmas >= 2.0,
        }
        print()
        print(f"{first['policy']} minus {second['policy']}: {gap:+.2f} +-{uncertainty:.2f}, "
              f"{sigmas:.1f} standard deviations")
        if sigmas < 2.0:
            print("Not enough evidence to separate them. More episodes, or a bigger "
                  "difference, or both.")
        else:
            print(f"{first['policy'] if gap > 0 else second['policy']} is ahead.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
