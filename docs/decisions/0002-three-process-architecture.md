# 2. Three processes, split by language boundary

Status: accepted
Date: 2026-08-23

## Context

The project needs to run game simulation, neural network training, and visualisation.
Mindustry is Java and its engine is not thread safe. The machine learning ecosystem is
Python. Visualisation is best served by the web.

## Decision

Three processes with narrow interfaces between them.

**`bridge/`**, Java 17, loaded as a JVM mod inside the Mindustry headless server. Exposes
state, executes actions, computes legality masks, records events. Contains no strategy.

**`gamma/`**, Python 3.13. Gymnasium environment, policy network, training loop. Talks to
the bridge over a local socket.

**`viewer/`**, React and Vite. Reads event logs. Runs with the other two processes dead.

## Consequences

- Parallelism is process-level. Each environment is a separate JVM, because the engine
  forbids sharing one across threads.
- A serialisation boundary sits on the hot path between simulation and training. Control
  messages are JSON, observation and action tensors are binary.
- The viewer is deployable as static files, since it has no server dependency. That is
  what allows replays to be published on GitHub Pages.
- Each component is separately testable, and a broken bridge cannot silently look like a
  weak agent.

## Alternatives rejected

**Everything in Java**, training included. Removes the serialisation boundary. Rejected
because the Java deep learning ecosystem would cost more than the boundary saves, and
would make the project unusable by the people most likely to contribute.

**Everything in Python via a reimplementation of Mindustry.** Full control, fast
simulation. Rejected as an enormous amount of work producing a game that is no longer
Mindustry, and whose results would transfer to nothing.

**Embedding a Python interpreter in the JVM.** Rejected as fragile tooling for a gain the
socket already provides.
