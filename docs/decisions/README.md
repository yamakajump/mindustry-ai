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
| [6](0006-fixed-timestep-acceleration.md) | Fixed timestep, with speed controlled by frame budget | accepted |
| [7](0007-pause-based-synchronous-stepping.md) | Synchronous stepping by pausing, not by blocking | accepted |
| [8](0008-embodied-agent.md) | The agent plays with a body | accepted |
| [9](0009-normalised-inputs.md) | Normalised inputs, and the entropy collapse they caused | accepted |
| [10](0010-live-scene-stream.md) | A scene stream, so training can be watched rather than inferred | accepted |

The overall picture these add up to is in [`../architecture.md`](../architecture.md).
