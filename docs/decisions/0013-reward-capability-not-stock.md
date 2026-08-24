# 13. Reward capability, not stock, and shape it so it cannot lie

Status: superseded by [14](0014-milestones-over-potential.md)
Date: 2026-08-24

> The shaping described here was measured and did not move the policy: entropy 12.067
> to 11.672 over eleven updates, policy loss 0.009. What replaced it, and what the
> research says about why, is in [decision 14](0014-milestones-over-potential.md).

## Context

`GZ_capture` paid one point per wave survived and nothing else. Waves arrive on a timer
whether the agent builds a turret or stands still, so every episode scored the same and
there was nothing to learn from. Measured over twenty-six generations of a real run: mean
reward 1.000, unchanged from the first to the last, entropy still at 11.6 because the policy
had no reason to leave random.

The obvious repair is to pay for things that look like progress: ore in the core, blocks
placed, damage avoided. The obvious repair is also how an agent learns a trick instead of a
game. Pay for ore in the core and the best policy is to mine by hand forever, because hand
mining fills the core and a drill costs resources to place. Pay per block placed and the
best policy is to place and break the same conveyor until the episode ends.

The requirement is sharper than "a dense reward". The agent has to work out for itself that
a drill mines, that a drill needs ore under it, that ore needs to reach the core. Nothing
should tell it. It should be graded on something that happens to rise faster when it has
figured those out.

## Decision

**Potential-based shaping**, as in [Ng, Harada and Russell,
1999](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf).
A shaping term of the form

```
F(s, s') = gamma * phi(s') - phi(s)
```

provably leaves the optimal policy unchanged, for any `phi` over states. That is the whole
reason to use it: it can guide without teaching a shortcut, because it cannot make a bad
policy optimal. It also cannot be pumped, because the sum along any loop telescopes to
zero: placing and breaking the same conveyor earns exactly nothing.

**The potential measures capability, not stock.** Three terms, all bounded, all functions
of the state alone:

| Term | Weight | Why |
|---|---|---|
| Core alive, as a fraction of full health | 2.0 | Everything else is worth nothing without it, and it is the only term that can be maximised outright |
| Ore banked, on a log | 0.6 | The first hundred copper matters, the ten thousandth does not. Saturating, so hoarding stops paying |
| Buildings standing, on a log | 0.6 | The term that pays for a factory rather than a pile |

The logs are the point of the design. A linear term over ore rewards the largest pile, which
hand mining can produce. A saturating one rewards *getting an economy started*, and once
started the only way to keep the term rising is to raise throughput, which is what a drill
does and a pair of hands does not. The agent is never told that drills mine. It is told the
number it is graded on, and left to notice that the number rises faster when it has them.

Measured on the potential as defined: one drill is worth about half of a thousand banked
copper the instant it is placed, and then it keeps producing.

The task's own reward stays sparse: one point per wave survived, fifty for capture. The
shaping carries the gradient, the objective carries the meaning.

## Consequences

The theorem assumes the potential is zero at a terminal state. It is at a real loss, because
the core is gone and the potential goes with it. It is not at a time limit, which is the
standard episodic approximation and the reason the step budget is generous rather than
tight: a policy that games the truncation has to survive nearly the whole episode to do it,
which is the behaviour we wanted anyway.

Two numbers had to start travelling for this to work: the core's maximum health, so the
health term normalises across core types, and the count of standing buildings, which the
observation never carried.

If the shaping turns out not to be enough on its own, the next thing to reach for is
intrinsic motivation rather than more shaping terms. [Random Network
Distillation](https://arxiv.org/abs/1810.12894) rewards novelty of observation and is the
practical choice for a sparse task. Adding a fourth hand-written term to the potential is
how a reward function becomes a list of hints.
