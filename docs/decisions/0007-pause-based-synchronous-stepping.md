# 7. Synchronous stepping by pausing, not by blocking

Status: accepted
Date: 2026-08-23

## Context

A reinforcement learning environment needs a guarantee that ordinary game servers do not
provide: **the world must not move while the agent is choosing**. Otherwise the state an
action was selected for is already stale when the action lands, episodes are not
reproducible, and a slow policy silently plays a different game from a fast one.

Mindustry runs a real-time loop and has no notion of waiting for anybody. Worse, its state
may only be touched from the game thread, while socket reads block, so the network and the
game cannot share a thread.

## Decision

**Pause the world instead of blocking a thread.**

After each observation is produced, the bridge sets the game state to `State.paused`, which
makes `Logic.update` skip the world entirely. A step request sets it back to `State.playing`
and counts down a fixed number of ticks; when the count reaches zero the world is paused
again and the observation is sent.

Requests cross from the network thread to the game thread through a bounded queue, and
replies travel back the same way. The game thread never blocks, and the network thread
blocks only on a reply it is entitled to expect.

A step spans several ticks. The default is 15, and measurement suggests 30 is better.

## Consequences

- **Exact stepping.** A step of 100 ticks advances game time by exactly 100.00 ticks. The
  fixed timestep from decision 6 is what makes this exact rather than approximate.
- **The world is verifiably frozen** between decisions, asserted by a test that idles for
  1.5 seconds and checks the tick counter has not moved.
- **The server stays responsive** while an agent thinks. Console commands still work, which
  matters more than it sounds: a bridge that freezes the process is undebuggable.
- **Measured cost is negligible**, 0.21 ms for a single-tick step including the round trip
  and the pause cycle. See [`../measurements/throughput.md`](../measurements/throughput.md).
- **One connection at a time**, by design. Two agents stepping one world raises questions
  about who owns time that have no good answer.
- Pausing fires state change events. Nothing depends on them today, but a future component
  reacting to pause would see one per step.

## Alternatives rejected

**Blocking the game thread on a queue** until the action arrives. Simpler to write and it
does deliver synchrony, but it freezes the entire process: no console, no logs advancing,
and a crashed agent leaves a server that looks hung rather than idle.

**Letting the world run freely** and reading state asynchronously. This is how a bot for a
live server would work. Rejected because it makes training non-reproducible and couples the
agent's effective reaction time to whatever the GPU happened to be doing.

**Driving the simulation manually**, calling `Logic.update()` a chosen number of times
rather than using the loop. Gives complete control over time, but bypasses the application
loop that the network layer and the scheduler also live on, so parts of the engine would
starve.

## Protocol notes

One type byte, a four byte big-endian length, then the payload. Length prefixing is
mandatory because TCP is a stream and newline delimiting breaks as soon as binary arrives.
The type byte is currently always JSON, and exists so observation tensors can be sent as
raw binary later without a second protocol or a version bump.

The socket binds to loopback only. It authenticates nothing and executes what it is told,
so it has no business on a public interface.
