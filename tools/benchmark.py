"""Run every baseline policy on a curriculum task and print the results table.

The output is the markdown that belongs in docs/measurements/baselines.md. Numbers in
that file come from here, never from an estimate.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from gamma import tasks
from gamma.alpha import AlphaPolicy
from gamma.env import MindustryEnv
from gamma.policies import MaskedRandomPolicy, RandomPolicy, run_episode

BRIDGE_PORT = 7870
GAME_PORT = 6870


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="T1_copper", choices=sorted(tasks.CURRICULUM))
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--root", type=Path, default=Path("mindustry-bench"))
    args = parser.parse_args()

    task = tasks.get(args.task)
    jar = str(next((Path("bridge") / "build" / "libs").glob("*.jar")))
    env = MindustryEnv(
        task,
        server_dir=str(args.root / "baselines"),
        bridge_port=BRIDGE_PORT,
        game_port=GAME_PORT,
        jar=jar,
    )

    policies = [
        ("Random", lambda i: RandomPolicy(env.action_space, seed=i)),
        ("Masked random", lambda i: MaskedRandomPolicy(env.action_space, seed=i)),
        ("**Alpha** (scripted)", lambda i: AlphaPolicy(env)),
    ]

    try:
        print(f"{task.name}: {task.description}")
        print(f"{args.episodes} episodes, {task.map_name}, {task.mode}\n")
        print("| Policy | Solved | Mean reward | Mean final copper | Actions accepted |")
        print("|---|---|---|---|---|")

        for name, make in policies:
            runs = []
            for episode in range(args.episodes):
                policy = make(episode)
                if hasattr(policy, "reset"):
                    policy.reset()
                runs.append(run_episode(env, policy))

            solved = sum(run["solved"] for run in runs)
            reward = sum(run["reward"] for run in runs) / len(runs)
            copper = sum(run["items"].get("copper", 0) for run in runs) / len(runs)
            attempts = sum(run["applied"] + run["refused"] for run in runs)
            accepted = sum(run["applied"] for run in runs) / max(1, attempts)

            print(
                f"| {name} | {solved}/{args.episodes} | {reward:+.2f} | "
                f"{copper:.0f} | {accepted * 100:.0f}% |"
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
