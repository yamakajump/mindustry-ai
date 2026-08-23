# 4. Curriculum of graded tasks before full games

Status: accepted
Date: 2026-08-23

## Context

The obvious starting point is to drop the agent into a full survival match and reward it
for surviving waves. It is the most motivating thing to watch and the closest to the goal.

It is also undebuggable. When the agent dies on wave 3, the failure is consistent with a
bug in the bridge, a miscalibrated reward, an agent that has not learned yet, or a map
that is too hard. There is no way to tell which, and the components most likely to be
broken early are the ones hardest to blame from that signal.

Every prior attempt at this problem started with a full game. The longest one survived
five days.

## Decision

A curriculum of graded tasks, growing along two axes at once: task difficulty and map
size. Full games are stage six, not stage one.

T1 delivers 100 copper on a 32x32 map. T2 sustains two ore chains. T3 builds a graphite
chain on 64x64. T4 and T5 introduce waves. T6 is a full survival game. T7 is attack. T8 is
PvP self-play.

Each stage is a benchmark with a number, runnable in seconds.

## Consequences

- A failing run points at a component: the last passing stage says what still works.
- T1 doubles as the end-to-end integration test for the bridge, and runs in CI.
- Published benchmark numbers exist from week two, long before an agent plays a real game.
- Small maps make early training cheap, which matters given how slow the simulator is.
- Curriculum design becomes a real risk: stages that are too far apart stall progress, and
  an agent may overfit to a stage rather than learn transferable behaviour.
- Map size varying across stages forces padding and a validity channel in the observation,
  so the network shape stays constant.

## Alternatives rejected

**Full survival game from the start.** More exciting, undebuggable, and empirically the
approach that killed every previous attempt.

**Attack mode first.** Clearer win condition than survival, but requires a working economy
and a working army simultaneously. Twice the difficulty at the entry point.

**PvP self-play from the start.** Produces an ELO immediately and is the eventual goal.
Rejected because two incompetent agents learn to beat each other incompetently, and can do
so for weeks without either building a viable economy.
