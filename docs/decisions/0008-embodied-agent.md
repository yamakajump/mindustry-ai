# 8. The agent plays with a body

Status: accepted
Date: 2026-08-24

## Context

The bridge edited the world directly: a block appeared instantly, anywhere on the map, for
the price of its materials. No human can do that. A player inhabits a core unit, has to
fly to what they want to build, waits while construction runs, and mines by holding
position over an ore patch until the unit fills.

Training against the god view teaches strategies that do not survive contact with the real
game, and it rules out the eventual goal of an agent playing alongside or against humans
on an ordinary server.

## Decision

The agent inhabits the unit a player's core spawns, and acts through it.

Every limit is the engine's own. `type.buildRange` decides how far it can reach,
`type.mineTier` which ores it can touch, and the build queue is the one the game already
processes for humans. Nothing is reimplemented, so nothing can drift from the real rules.

The agent expresses intent, not outcome: go there, mine that tile, queue this building.
Whether any of it succeeds belongs to the simulation.

The direct mode stays available. It trains far faster and serves as an upper bound:
whatever an embodied agent achieves, a disembodied one had fewer excuses to miss.

## Consequences

- **The action space changes with the mode.** Embodied: move, build, mine, unload, break.
  Direct: place, break. The environment exposes whichever applies.
- **Masks gain a `mineable` layer**, ore the unit is actually allowed and able to reach.
  Ore under a building is excluded, because the engine refuses it: without that exclusion
  the nearest ore to a unit sitting on its core is the core's own footprint, and the agent
  mines nothing for an entire episode while every action is accepted.
- **Episodes get much longer.** Hand mining fills 30 items in minutes of game time, where
  the direct agent produced instantly. This is the real cost of the real game, and it is
  the reason drills matter.
- **Travel becomes part of the problem.** A build order 40 tiles away stays queued and
  nothing is placed until the unit flies over. Verified.

## Verified

On Ground Zero, through the protocol: build range 27.5 tiles; an order at 40 tiles left
queued with nothing placed; the unit travels; it mines 28 copper by hand; a deposit moved
the core from 360 to 480.

## The bug that took longest, and what it was

The embodied baseline mined and banked for hundreds of steps while the core stock never
moved. Two false leads first: the deposit handover, which turned out to work correctly in
isolation (`accepted=29, core 530->559`), and the capacity threshold, which never fired
because the engine banks slightly before the stated capacity.

The actual cause was neither. The mineable mask covers *every* ore the unit can reach, and
the policy took the nearest tile from it. Near a core that is usually sand or lead. The
agent mined diligently, banked diligently, and the copper the task scores never moved.

A useful reminder that "nothing is happening" and "the wrong thing is happening
successfully" look identical from the outside. Instrumenting the engine settled it in one
run, after five rounds of guessing had not.

## Baseline with a body

T1, 450 decisions, hand mining, three episodes each.

| Policy | Mean final copper | Mean reward | Actions accepted |
|---|---|---|---|
| Masked random | 0 | -2.00 | 95% |
| **Embodied Alpha** | **192** | **-0.08** | 100% |

Note that acceptance rate stops being informative once there is a body: `move` is almost
always legal, so a policy can be accepted 95% of the time while bankrupting itself.

## Alternatives rejected

**Keeping only the direct mode.** Faster to train and simpler, but it cannot lead to an
agent that plays on a normal server, which is the point of the project.

**Reimplementing range and mining rules in the bridge.** Would allow tuning them for
training. Rejected for the same reason legality checks are left to the engine: a
reimplementation drifts, and then the agent has learned a game nobody else is playing.
