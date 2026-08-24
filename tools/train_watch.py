"""Run several matches in parallel and watch them live in a browser.

    python tools/train_watch.py --matches 6

Opens a dashboard on http://127.0.0.1:8800 showing every match at once: what each agent
is building, what it holds, how close it is to the objective, and which runs are ahead.

Each match is its own Mindustry process, because the engine is not thread safe. They are
driven from threads here, which is fine: the threads only wait on sockets.
"""

from __future__ import annotations

import argparse
import threading
import time
import webbrowser
from dataclasses import replace
from pathlib import Path

from gamma import tasks
from gamma.cleanup import kill_servers
from gamma.alpha import AlphaPolicy
from gamma.alpha_embodied import EmbodiedAlphaPolicy
from gamma.env import MindustryEnv
from gamma.monitor import TrainingMonitor
from gamma.policies import MaskedRandomPolicy, RandomPolicy

BRIDGE_PORT_BASE = 7900
GAME_PORT_BASE = 6900

#: Policies cycled across matches, so the dashboard has something to compare.
ROSTER = ("alpha", "masked-random", "random")


def make_policy(name: str, env: MindustryEnv, seed: int):
    if name == "alpha":
        return EmbodiedAlphaPolicy(env) if env.embodied else AlphaPolicy(env)
    if name == "masked-random":
        return MaskedRandomPolicy(env.action_space, seed=seed, env=env)
    return RandomPolicy(env.action_space, seed=seed)


def objective_of(task) -> str:
    return task.description


def run_match(index: int, args, monitor: TrainingMonitor, stop: threading.Event) -> None:
    """Play episodes back to back, publishing state as it goes."""
    task = tasks.get(args.task)
    if args.max_steps:
        task = replace(task, max_steps=args.max_steps)

    policy_name = ROSTER[index % len(ROSTER)]
    state = monitor.match(index)
    state.policy = policy_name
    state.task = task.name
    state.objective = objective_of(task)
    state.max_steps = task.max_steps

    jar = str(next((Path("bridge") / "build" / "libs").glob("*.jar")))
    env = MindustryEnv(
        task,
        server_dir=str(Path(args.root) / f"match{index}"),
        bridge_port=BRIDGE_PORT_BASE + index,
        game_port=GAME_PORT_BASE + index,
        jar=jar,
        embodied=args.embodied,
    )

    try:
        while not stop.is_set():
            policy = make_policy(policy_name, env, seed=index * 977 + state.episode)
            if hasattr(policy, "reset"):
                policy.reset()

            observation, info = env.reset()
            state.core = [int(info["raw"].get("core_x", -1)), int(info["raw"].get("core_y", -1))]
            state.size = [int(info["raw"].get("map_width", 0)), int(info["raw"].get("map_height", 0))]
            state.built = []
            state.applied = state.refused = 0

            total = 0.0
            for step in range(task.max_steps):
                if stop.is_set():
                    break

                action = policy.act(observation, info)
                observation, reward, terminated, truncated, info = env.step(action)
                total += reward

                raw = info["raw"]
                outcome = info.get("action") or {}
                if outcome:
                    state.applied += bool(outcome.get("applied"))
                    state.refused += not outcome.get("applied")

                state.step = step + 1
                state.tick = float(raw.get("tick", 0))
                state.wave = int(raw.get("wave", 0))
                state.reward = total
                state.items = dict(raw.get("items", {}))
                state.unit = dict(raw.get("unit", {}))
                state.action = env.action_types[int(action[0])]
                state.progress = progress_of(task, raw)

                # Only what the agent built, which is what a mini map needs to show.
                if outcome.get("applied") and state.action in ("place", "build"):
                    state.built.append([int(action[2]), int(action[3])])
                    del state.built[:-400]

                if terminated or truncated:
                    break

            monitor.record_episode(index, total, task.succeeded(info.get("raw", {})))
    except Exception as error:  # noqa: BLE001
        state.alive = False
        print(f"match {index} stopped: {error}")
    finally:
        env.close()


def progress_of(task, raw: dict) -> float:
    """How close this match is to its objective, as a fraction.

    Approximate by design: it exists to fill a progress bar, not to score anything.
    """
    if task.name.startswith("GZ"):
        return min(1.0, int(raw.get("wave", 0)) / 10.0)
    copper = int(raw.get("items", {}).get("copper", 0))
    return min(1.0, copper / 250.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matches", type=int, default=4)
    parser.add_argument("--task", default="T1_copper", choices=sorted(tasks.CURRICULUM))
    parser.add_argument("--embodied", action="store_true", help="agents play as players")
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--port", type=int, default=8800)
    parser.add_argument("--root", default="mindustry-train")
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()

    # A killed run leaves its servers holding the ports, and the next one then
    # fails with "no environment started" and no hint as to why.
    kill_servers()

    monitor = TrainingMonitor(title=f"{args.task}{' (embodied)' if args.embodied else ''}")
    url = monitor.serve(args.port)
    print(f"dashboard: {url}")
    if not args.no_open:
        webbrowser.open(url)

    stop = threading.Event()
    threads = [
        threading.Thread(target=run_match, args=(i, args, monitor, stop), daemon=True)
        for i in range(args.matches)
    ]
    for thread in threads:
        thread.start()

    print(f"{args.matches} matches starting, Ctrl+C to stop")
    try:
        while any(thread.is_alive() for thread in threads):
            time.sleep(1)
    except KeyboardInterrupt:
        print("stopping...")
        stop.set()
        for thread in threads:
            thread.join(timeout=30)
    finally:
        monitor.stop()


if __name__ == "__main__":
    main()
