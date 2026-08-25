# 16. Pay differences, never levels

Status: accepted
Date: 2026-08-25
Amends: [14. Grade the agent on a ladder of firsts, not on a potential function](0014-milestones-over-potential.md)

## Context

Decision 14 replaced an inert potential function with a ladder of first-time milestones
plus a continuous term for work done at a consumer. The continuous term had three parts:
ore delivered, crafts completed, and generation running.

The first two are differences between two observations. The third was not. It read
`obs["power"]`, the generation burning at that instant, and paid five hundredths of it.
The comment I wrote called this "worth acknowledging while it runs". A test froze it as
intent: `test_a_generator_running_pays_while_it_runs`, asserting 0.2 for a generation of
four, described as "instantaneous rather than cumulative".

An episode is three thousand steps. Paying a level on every step pays it three thousand
times for a machine bought once. That is not an acknowledgement, it is an annuity, and it
is available to any policy that can afford a generator and then do nothing at all.

The agent found it. The run looked healthy while it did, which is the part worth
remembering: mean score climbing smoothly from +21 to +205 over two hundred updates,
best episode +1577, entropy falling, value loss small. Every surface signal said the
policy was learning.

It was learning. What it had found to learn was that a generator prints points.

The archived episodes said so plainly. Across 167 of them the policy built **one**
conveyor line that ever reached a core, 0.6%, having placed 3,674 drills and 3,356
conveyors that led nowhere. The five best-scoring episodes contained zero chains and zero
drills touching a core, so they delivered nothing whatsoever.

The reward trace of the +1577 episode is unambiguous:

| reward per step | number of steps |
|---|---|
| 0.55 | 1,302 |
| 0.45 | 583 |
| 0.35 | 294 |
| 0.25 | 334 |

Exact multiples of 0.05 for thousands of consecutive steps: generation of 11, 9, 7 and 5
burning steadily while nothing else happened at all. Not a delivery, not a craft, not one
ore carried, because any of those would have broken the multiple. Roughly 1,400 of the
1,577 points were the annuity; the milestones, capped at 466 per episode by construction,
were the rest.

## Decision

**Every per-step term in a reward is a difference between two observations. Never a
level.**

The generation term is gone. Power still pays, twice, and both times honestly: once
through the `first_power` milestone, and then through what it enables, since a crafter is
paid per craft and a turret per kill. Paying the level on top was double counting the same
machine on every step of the episode.

The invariant is now tested on every task in the curriculum:

> A step in which nothing changed earns nothing.

`frontier_shaped` is exempt, for a real reason rather than convenience. Potential-based
shaping pays `gamma * phi(s') - phi(s)`, so a static step pays `(gamma - 1) * phi`, which
is negative and cannot be collected: the sum over an episode telescopes to
`gamma^T * phi(last) - phi(first)` regardless of what happens in between. The test asserts
that it is at most zero instead.

## Consequences

The invariant costs nothing to check and does not need a training run to fail. This bug
took two hundred updates and a replay analysis to find; a static observation would have
caught it in under a millisecond.

It also settles a question that kept coming back. A rising reward curve is not evidence
that an agent is learning the task, only that it is learning *something*. The number that
carries the claim is the one measured on the artefact rather than the score: how many
episodes contain a chain that actually reaches a core. That number is now watched
alongside the training curve rather than inferred from it.

Nothing here rules out other farms. It rules out this whole *class* of farm, which is the
only kind of guarantee a reward function can offer.
