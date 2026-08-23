# Architecture decisions

Each file records one decision: the context that forced it, what was chosen, what it costs,
and what was rejected. They are append-only. A decision that turns out wrong gets a new
file superseding the old one rather than a quiet edit, because the reasoning that led to a
mistake is worth more than a clean history that hides it.

| # | Decision | Status |
|---|---|---|
| [1](0001-full-action-space.md) | Full action space, factored and masked | accepted |
| [2](0002-three-process-architecture.md) | Three processes, split by language boundary | accepted |
| [3](0003-one-replay-format.md) | One event format for replay and live viewing | accepted |
| [4](0004-curriculum-first.md) | Curriculum of graded tasks before full games | accepted |
| [5](0005-in-repo-ppo.md) | PPO implemented in-repo, not pulled from a library | accepted |

The overall picture these add up to is in [`../architecture.md`](../architecture.md).
