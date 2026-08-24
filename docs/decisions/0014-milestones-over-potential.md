# 14. Grade the agent on a ladder of firsts, not on a potential function

Status: accepted
Date: 2026-08-24
Supersedes: [13. Reward capability, not stock](0013-reward-capability-not-stock.md)

## Context

Decision 13 replaced a sparse wave counter with potential-based shaping: a bounded
potential over core health, banked ore and buildings standing, paid as
`gamma * phi(s') - phi(s)`. That form has a theorem behind it. [Ng, Harada and Russell
(1999)](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf)
prove it cannot change which policy is optimal, which makes it the safe way to give a
dense signal on a sparse task.

It was safe and it was inert. Measured over eleven updates of a real run: entropy 12.067
to 11.672, policy loss 0.009. The policy did not move off random.

The reason is not subtle in hindsight. Shaping tells an agent that a state it has reached
is better than the one before it. It says nothing about how to reach a state it has never
reached. A policy that has never once built a drill on ore and run a conveyor to the core
gets no information at all from a potential that would reward it for having done so, and
the odds of doing it by accident, with a factored action space over block type, position
and rotation, are not odds worth waiting on.

Two of the three terms were also wrong on their own merits. Buildings standing is
farmable: placing and breaking the same conveyor moves the count and a policy that finds
that out has found a strategy. Core health at weight 2.0 dominated everything and is a
pure penalty the agent cannot influence for the first several hundred steps of an episode.

## What the research actually does

Nobody in the Mindustry community has built a reinforcement learning agent for the game.
The closest thing that exists, [NM](https://github.com/GreenBushy/NM), trains a
perceptron *inside* the game using logic processors, which is a different and rather
wonderful project. So there is no prior art in this game to copy, and the question becomes
what has been measured on games shaped like it.

**[Action Guidance, Huang and Ontanon 2020](https://arxiv.org/abs/2010.03956)** is the
closest published result: microRTS, a build-and-fight game on a grid, PPO with invalid
action masking, the same action structure as this project. They ran sparse reward against
shaped reward on three tasks. On `ProduceCombatUnits` the sparse agent scored **0.00 out
of 10** and the shaped agent **9.57**. The shaping that produced 9.57 is not clever:

| | |
|---|---|
| a building constructed | +1 |
| a resource harvested | +1 |
| a resource returned | +1 |
| a combat unit produced | +7 |
| winning | +5 |

Additive event rewards, graded by difficulty. Their own contribution is a method for
removing the resulting bias afterwards, which is the honest reading: shaping of this kind
works, and its bias is a problem to solve after it has worked, not before.

**[Factorio Learning Environment](https://arxiv.org/abs/2503.09617)** (NeurIPS 2025)
scores agents on a production score: the value of all items **produced**, with item values
scaling in recipe complexity. Flow, not stock. It also tracks *milestones*, a discrete set
of achievements for producing each item type for the first time.

**[Crafter](https://arxiv.org/abs/2109.06780)** pays +1 the first time each of its
twenty-two achievements is unlocked in an episode, and nothing for any repeat, plus a
small health term. Its scoring is a geometric mean over achievement rates, explicitly so
that going from never collecting a diamond to collecting one sometimes counts for more
than going from 90% wood to 95%.

Three working designs, same shape: **first-time events graded by difficulty, plus a
measure of flow, plus a small survival term.** Not one of them is a potential function.

## Decision

`frontier` is graded on that shape.

**A ladder of milestones, each paid once per episode.** Every rung reads a counter the
engine already keeps and that only ever climbs, so "crossed the threshold this step" is a
complete definition of "first time" and the reward function needs no memory. That is also
what makes it unfarmable: a monotonic counter pays its rung at most once, whatever happens
next. Counting what is standing right now, as decision 13 did, pays for building and
breaking the same conveyor forever.

| rung | worth |
|---|---|
| a first drill, conveyor, wall, turret | 5, 3, 3, 5 |
| **the first ore a machine delivered** | **30** |
| a hundred, a thousand, ten thousand of it | 30, 60, 60 |
| a second and third ore automated | 20, 30 |
| a first kill, then twenty-five | 20, 30 |
| a first generator, crafter, unit factory | 15, 25, 25 |
| wave 5, 10, 20 | 15, 30, 60 |

**Flow, at 0.02 per item a machine delivered.** The engine draws the line for us. A
conveyor hands items to a core through `handleItem`, which increments
`state.stats.coreItemCount`; a unit banking a stack goes through `handleStack`, which does
not; a launch loadout is written straight into the inventory. So automated income is
separated from hand mining by the game rather than by anything invented here, and it is
the one number that says a factory exists. Ten thousand delivered is worth roughly the
whole ladder, which is the balance we want: the ladder gets it started and production is
what there is left to optimise afterwards.

**Hand mining at a tenth of that**, so carrying ore in is always the worse way to earn the
same reward, and picking ore up at a tenth again, because without it the first thousand
steps of a fresh policy carry no gradient whatsoever.

**Holding it:** +1 per kill, +2 per wave, -0.5 per building lost, -0.005 per point of core
damage, -50 for losing the core.

`frontier_shaped` keeps the potential version on the same worlds with the same objective,
so the change can be measured rather than argued about.

## Consequences

The farming holes are named rather than hoped about. The one term paid per unit is
automated delivery, and the engine's own comment on that counter warns it can be inflated
with an unloader feeding a conveyor back into the core. That costs titanium and silicon,
which needs a pneumatic drill and a powered smelter, which is several rungs past anything
this agent can reach. It is a hole to watch for once it can, not a hole today. Everything
else is either a monotonic first or a real event in the world.

Direct mode was silently invisible to every statistic the game keeps. Placing a block
through `tile.setBlock` does not announce itself, and a player's build finishes through
the build queue and fires `BlockBuildEndEvent`, which is what the engine's counters listen
to. The event is now fired from the executor, so the two modes are indistinguishable from
outside, which is the only reason direct mode is allowed to exist.

The evaluator now reports how often each rung was reached, which says far more than a mean
reward. A policy that never once reaches `automation` has not started playing the game,
whatever it scores.

Two levers were added alongside, both aimed at the same admission: jumping to 147
generated worlds before ever proving the agent learns on one was the wrong order.
`--worlds N` narrows the training pool without touching the held-out half, so the
measurement stays honest while the basics are in doubt.

If the ladder turns out to be climbable and the agent still stalls above it, the next
lever is intrinsic motivation rather than a longer ladder.
[RND](https://arxiv.org/abs/1810.12894) rewards novelty of observation and is the standard
choice. Adding a nineteenth rung is how a reward function becomes a list of hints.
