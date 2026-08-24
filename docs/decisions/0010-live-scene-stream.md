# 10. A scene stream, so training can be watched rather than inferred

Status: accepted
Date: 2026-08-24

## Context

The dashboard could show that training was happening. It could not show what was
happening. It drew a still picture of the map from the terrain export and a dot for the
agent, refreshed twice a second, and everything that makes a Mindustry match legible was
missing from it: no enemy units, no construction in progress, no shots, no destruction.

The honest answer for a long time was to point at the real game. Every environment is a
real headless server with its game port open, so a client can join and see everything,
animated by the engine itself. That is still the best picture available and the port is
printed on every run.

It is also not an answer to the question being asked. Someone watching a training run
wants six matches at once, the reward on each, the generations behind them, and the
ability to tell in one glance which one is doing something interesting. A game client
shows one match with no context. The two are not substitutes.

## Decision

Add a third channel next to the observation tensor and the map export: a **scene**, a
frame of everything that moves, requested by name and served to the browser through the
monitor.

The three now divide cleanly by audience:

| Channel | Consumer | Contents | Frequency |
|---|---|---|---|
| Observation tensor | the policy | clamped categories, no identity | every step, every environment |
| Map export | the viewer | typed terrain, exact block ids | once per episode |
| **Scene** | the viewer | units, buildings, damage, shots | ten times a second, watched matches |

Three properties make it affordable.

**Deltas at the bridge.** `SceneEncoder` reports a building when it appears, changes or
takes damage, never the full list. A developed base is thousands of buildings and sending
them per frame would cost more than the observation tensor it travels beside. Damage alone
is two numbers rather than six, because a base under fire changes health on hundreds of
buildings a second and nothing else about them.

**Re-aggregation at the monitor.** Bridge deltas are against the bridge's own last frame,
which suits exactly one reader. A browser polling at its own rate would only ever see the
frames it happened to land on and would miss everything in between. `SceneBuffer` folds
the deltas into an authoritative world and stamps every change with a version, so a viewer
asks for what changed since the version it holds and gets exactly that. A viewer too far
behind is resynchronised with the whole world rather than patched with removals that are
no longer retained.

**A wall clock, not a step clock.** At training speed a step takes a couple of
milliseconds, so one frame per step would be hundreds a second for a browser that can draw
ten. The collection is throttled on wall-clock time, and the simulation is left to run
flat out.

## Consequences

The browser now animates: positions carry their fraction so units are interpolated between
frames rather than teleported, construction is hatched until it completes, damage flashes,
destruction leaves a flare, and shots are drawn in flight. The ground is painted once into
an offscreen canvas and blitted, because repainting eighty thousand tiles per frame across
six matches is what would make it unwatchable.

Item flow on conveyors is deliberately absent, and is the one visible thing that cannot be
sent honestly. Items move every tick and there are thousands of them; a frame every
hundred milliseconds would show them teleporting. A viewer that wants them should animate
them from the throughput of the conveyor rather than be told where each one is.

The cost is a second request per collected frame on watched matches. Measured on a
three-environment run: frames of 371 to 772 bytes, and deltas smaller again once a viewer
is up to date.

The real game stays the better picture of one match, and the port is still printed. This
is the better picture of a training run.
