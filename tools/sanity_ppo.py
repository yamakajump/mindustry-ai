"""Can the learner learn anything at all?

    python tools/sanity_ppo.py

Before asking why an agent does not learn Mindustry, it is worth knowing whether it can
learn a task with a single obvious answer. This runs the real `PolicyNet` and the real
`PPO` on a toy environment shaped exactly like the game's: the same spatial observation,
the same factored action space over type, block, position and rotation, the same masks.

The task is the easiest one that still exercises the part of the network that matters. A
single tile of the observation is lit, and the agent has to point at it. Nothing else
scores. A learner that works finds this in a few thousand steps; a learner that does not
is broken in a way no amount of training on a real game will reveal, because on a real
game every failure looks like the task being hard.

It runs in seconds and needs no server, so it is the first thing to reach for when a run
goes nowhere.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from gamma.net import PolicyNet
from gamma.ppo import PPO, PPOConfig, RolloutBuffer


class PointAtIt:
    """Light one tile, reward pointing at it. Vectorised over `envs`.

    Deliberately not a Gym environment: the point is to drive `PPO.update` with the same
    tensors the training loop builds, not to test a wrapper.
    """

    def __init__(self, envs: int, window: int, channels: int, n_types: int, n_blocks: int,
                 rng: np.random.Generator, delay: int = 0) -> None:
        self.envs = envs
        self.window = window
        self.channels = channels
        self.n_types = n_types
        self.n_blocks = n_blocks
        self.rng = rng
        self.targets = self.rng.integers(0, window * window, size=envs)

        #: How many steps pass between the right action and the reward for it.
        #:
        #: Zero is the easy version. Anything else is the question this whole project
        #: turns on: in Mindustry the decision to put a drill on ore is paid hundreds of
        #: steps later, when the ore finally reaches the core, and a discount that cannot
        #: carry credit that far leaves the decision unrewarded whatever the reward says.
        self.delay = delay
        self.pending = [np.zeros(envs, dtype=np.float32) for _ in range(delay + 1)]
        self.clock = 0

        #: How long a correct action stays visible in the observation before it pays.
        #:
        #: This is what makes a delay learnable at all, and it is Mindustry's own shape: a
        #: drill placed on ore stays on the map, so the critic can learn that the state
        #: containing it is worth more, and the temporal-difference error then spikes at
        #: the moment of placement rather than at the moment ore arrives. Without it the
        #: credit for one good action is smeared over every action in the window, and the
        #: signal to noise ratio is one over the delay however far the discount carries.
        self.marks = np.zeros((envs, window * window), dtype=np.float32)
        self.expiry: list[tuple[int, np.ndarray, np.ndarray]] = []

    def observe(self) -> dict[str, np.ndarray]:
        spatial = np.zeros((self.envs, self.channels, self.window, self.window), dtype=np.uint8)
        for env, target in enumerate(self.targets):
            spatial[env, 0, target // self.window, target % self.window] = 255
        # Channel one carries what has already been earned and not yet paid, the way the
        # game's observation carries the drills the agent has already placed.
        spatial[:, 1] = (self.marks * 255).reshape(
            self.envs, self.window, self.window).astype(np.uint8)
        return {
            "spatial": spatial,
            "global": np.zeros((self.envs, 8), dtype=np.float32),
        }

    def masks(self) -> dict[str, np.ndarray]:
        return {
            "type": np.ones((self.envs, self.n_types), dtype=bool),
            "block": np.ones((self.envs, self.n_blocks), dtype=bool),
            "position": np.ones((self.envs, self.window * self.window), dtype=bool),
        }

    def step(self, actions: np.ndarray) -> np.ndarray:
        """One point for pointing at the lit tile, paid now or `delay` steps later."""
        chosen = actions[:, 3] * self.window + actions[:, 2]
        earned = (chosen == self.targets).astype(np.float32)
        hit = chosen[earned > 0]
        rows = np.flatnonzero(earned > 0)
        self.targets = self.rng.integers(0, self.window * self.window, size=self.envs)

        if self.delay == 0:
            return earned

        # Standing in the observation from the moment it is earned until it is paid.
        if rows.size:
            self.marks[rows, hit] = 1.0
            self.expiry.append((self.clock + self.delay, rows, hit))

        slot = self.clock % (self.delay + 1)
        due = self.pending[slot].copy()
        self.pending[slot] = earned
        self.clock += 1

        while self.expiry and self.expiry[0][0] <= self.clock:
            _, rows, hit = self.expiry.pop(0)
            self.marks[rows, hit] = 0.0

        return due


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", type=int, default=16,
                        help="smaller than the game's, so chance is not one in 2304")
    parser.add_argument("--envs", type=int, default=16)
    parser.add_argument("--updates", type=int, default=60)
    parser.add_argument("--entropy-coef", type=float, default=None,
                        help="override, to see what the training default costs")
    parser.add_argument("--delay", type=int, default=0,
                        help="steps between the right action and its reward")
    parser.add_argument("--gamma", type=float, default=None,
                        help="override the discount, to see how far credit carries")
    parser.add_argument("--lam", type=float, default=None,
                        help="override GAE lambda, which is what actually sets the "
                             "credit horizon: it decays as (gamma*lambda)^k, not gamma^k")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    config = PPOConfig()
    if args.entropy_coef is not None:
        config.entropy_coef = args.entropy_coef
    if args.gamma is not None:
        config.gamma = args.gamma
    if args.lam is not None:
        config.gae_lambda = args.lam

    channels, n_types, n_blocks = 14, 6, 30
    net = PolicyNet(channels=channels, window=args.window, globals_size=8,
                    n_types=n_types, n_blocks=n_blocks)
    agent = PPO(net, config)
    world = PointAtIt(args.envs, args.window, channels, n_types, n_blocks, rng,
                      delay=args.delay)

    chance = 1.0 / (args.window * args.window)
    steps = config.steps_per_env
    device = config.device

    print(f"pointing at one tile in {args.window}x{args.window}, chance is {chance:.4%}")
    decay = config.gamma * config.gae_lambda
    print(f"reward delayed by {args.delay} steps, gamma {config.gamma}, "
          f"lambda {config.gae_lambda}")
    print(f"GAE credit decays as (gamma*lambda)^k = {decay:.4f}^k, so its horizon is "
          f"{1 / (1 - decay):.0f} steps and {decay ** max(args.delay, 1):.2%} survives "
          f"the delay")
    print(f"entropy coefficient {config.entropy_coef}, {sum(p.numel() for p in net.parameters()):,} parameters")
    print()

    started = time.time()
    history = []

    for update in range(1, args.updates + 1):
        buffer = RolloutBuffer(
            spatial=torch.zeros((steps, args.envs, channels, args.window, args.window),
                                dtype=torch.uint8, device=device),
            globals_=torch.zeros((steps, args.envs, 8), device=device),
            actions=torch.zeros((steps, args.envs, 4), dtype=torch.long, device=device),
            log_probs=torch.zeros((steps, args.envs), device=device),
            rewards=torch.zeros((steps, args.envs), device=device),
            dones=torch.zeros((steps, args.envs), device=device),
            values=torch.zeros((steps, args.envs), device=device),
            masks={
                "type": torch.zeros((steps, args.envs, n_types), dtype=torch.bool, device=device),
                "block": torch.zeros((steps, args.envs, n_blocks), dtype=torch.bool, device=device),
                "position": torch.zeros((steps, args.envs, args.window ** 2), dtype=torch.bool, device=device),
            },
        )

        hits = 0.0
        for step in range(steps):
            observation = world.observe()
            masks = world.masks()
            action, log_prob, _, value = agent.act(observation, masks)

            buffer.spatial[step] = torch.as_tensor(observation["spatial"], device=device)
            buffer.globals_[step] = torch.as_tensor(observation["global"], device=device)
            buffer.actions[step] = action
            buffer.log_probs[step] = log_prob
            buffer.values[step] = value
            for key, value_array in masks.items():
                buffer.masks[key][step] = torch.as_tensor(value_array, device=device)

            rewards = world.step(net.to_env_action(action))
            hits += float(rewards.mean())
            buffer.rewards[step] = torch.as_tensor(rewards, device=device)
            # With no delay every step is its own episode. With one, the episode has to
            # outlive the delay or the reward lands after the end and is never seen.
            buffer.dones[step] = 1.0 if args.delay == 0 else 0.0

        with torch.no_grad():
            last = agent.net.value(
                torch.as_tensor(world.observe()["spatial"], device=device),
                torch.as_tensor(world.observe()["global"], device=device),
            )

        stats = agent.update(buffer, last)
        rate = hits / steps
        history.append(rate)
        if update % 5 == 0 or update == 1:
            print(f"update {update:3d}  hit rate {rate:7.2%}  ({rate / chance:5.1f}x chance)"
                  f"  entropy {stats['entropy']:6.3f}  policy {stats['policy_loss']:+.4f}")

    best = max(history)
    final = sum(history[-5:]) / min(5, len(history))
    print()
    print(f"chance {chance:.2%}   best {best:.2%}   last five {final:.2%}   "
          f"in {time.time() - started:.0f}s")
    print()
    if final > chance * 5:
        print("VERDICT: the learner learns. Whatever is wrong is not in ppo.py or net.py.")
    elif final > chance * 1.5:
        print("VERDICT: it moves, barely. Something is throttling the update.")
    else:
        print("VERDICT: it does not learn a task with one obvious answer and no horizon.")
        print("         Stop training the game and fix this first.")


if __name__ == "__main__":
    main()
