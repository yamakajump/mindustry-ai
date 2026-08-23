# 5. PPO implemented in-repo, not pulled from a library

Status: accepted
Date: 2026-08-23

## Context

Decision 1 commits to a factored action space with per-head legality masking. That is the
one thing this project cannot compromise on, because it is what keeps the agent capable of
every action a human can perform.

Stable-Baselines3 is the default choice for a Gymnasium environment and would otherwise be
right: mature, tested, five lines to a running agent. But it has no clean support for
autoregressive multi-head action spaces with per-head masks. Bending it into that shape
means fighting the library at every step and inheriting a debugging surface nobody
understands.

## Decision

PPO is implemented in this repository, as readable single-purpose code, following the
CleanRL Gym-uRTS agent, which solves exactly this shape of problem.

Stable-Baselines3 stays as a smoke test only, run against a simplified flat action space,
whose sole job is to answer one question: is the environment broken, or is the agent just
bad?

## Consequences

- More code to write and to own, including the parts that are easy to get subtly wrong.
- Full control over masking, head conditioning and the value head.
- The training loop is readable, which serves the learning goal behind this project. An
  algorithm you have implemented is one you understand.
- The SB3 smoke test is a genuine diagnostic: if it cannot learn a trivial flat task, the
  bug is in the environment, not the policy.
- The environment stays Gymnasium-compatible regardless, so anyone else can point any
  algorithm at it.

## Alternatives rejected

**Stable-Baselines3 as the main trainer.** Fastest path to a first result. Rejected on
action space incompatibility, which is structural rather than a matter of configuration.

**RLlib.** Handles complex action spaces and distributed training properly. Rejected as
heavy machinery whose configuration surface would cost more to learn than the training
loop itself, on a project that runs on one machine.

**An evolutionary algorithm rather than policy gradients.** Genuinely tempting: trivial to
implement, indifferent to sparse and delayed reward, and it parallelises perfectly across
processes, which happens to match the constraint that Mindustry imposes anyway. Rejected
as the primary method because it needs far more episodes than gradient methods, and
episodes are the scarce resource here. Worth revisiting as a comparison once the
environment is fast and stable, and it would make an interesting result either way.
