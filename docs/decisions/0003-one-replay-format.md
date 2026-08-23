# 3. One event format for replay and live viewing

Status: accepted
Date: 2026-08-23

## Context

Two things need to be watchable: training as it happens, and matches after the fact.

They look like separate features. Live viewing suggests a streaming protocol, replay
suggests a file format. Building both independently means writing the rendering path
twice and keeping two formats in sync forever.

Live viewing alone is also the weaker feature. Training runs for hours or days across
many parallel environments. Nobody is watching at 4am when something interesting happens,
and a live view cannot be paused, rewound, or compared against an earlier generation.

## Decision

A single append-only event log serves both.

A header carries the map, rules and initial terrain. Every subsequent line is a delta: a
block placed or destroyed, a unit spawned, a wave started, resources changed.

On disk: gzipped JSON Lines. Over the wire: the same lines, in the same order, pushed
through a WebSocket. The viewer consumes an ordered stream of events and does not know or
care which transport delivered it.

Replay is built first. Live falls out of it at near zero additional cost.

## Consequences

- Rendering code exists once.
- Live viewing is a transport swap, not a feature.
- Replays are self-contained artefacts, publishable as static files and playable from the
  README with nothing installed.
- Generation comparison and side-by-side playback become possible, which live streaming
  alone could never offer.
- Event logs must stay compact. Per-tick full state snapshots are not viable, so the
  format is delta-based, which means a corrupt line invalidates everything after it.

## Alternatives rejected

**Live streaming only.** Nothing to rewatch, nothing to compare, nothing to publish.

**Storing action sequences and replaying them through the engine.** Far smaller files, but
it depends on Mindustry being deterministic given identical seeds and inputs, which is
currently unverified (see the open questions in the architecture document). It also
requires a working game install to watch anything, which kills browser playback.

**Video capture.** Trivially shareable, useless for analysis, and enormous.
