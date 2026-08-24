# 9. Normalised inputs, and the entropy collapse they caused

Status: accepted
Date: 2026-08-24

## Context

Beta's first training runs looked like they were working: 300 environment steps per
second, episodes completing, checkpoints saving. They learned nothing at all.

The signature was entropy. It started around 3, fell to zero within five updates, and
stayed there. A policy with zero entropy has committed to one action and stopped
exploring, before it has seen enough of the game to have an opinion worth committing to.

## What was actually wrong

The global observation vector was fed to the network raw. It carried `tick` values in the
thousands, `wave_time` up to 14,400, and item counts in the hundreds, side by side with
booleans.

That produced enormous activations, an enormous value head output, and a value loss above
4,000 against a policy loss of 0.3. With `value_coef = 0.5`, the critic's gradient was
four orders of magnitude larger than the policy's. Every update was effectively a
value-function update, and the entropy bonus at 0.03 was noise against it.

Three false starts before finding it:

- **Suspected the masks.** Instrumenting them showed 82% of action types and 56% of
  positions legal, which is healthy.
- **Suspected PPO.** Tested on synthetic data, on CPU and GPU, with partial masks and
  real log-probabilities. Correct every time: entropy 11.8, policy loss moving.
- **Suspected reward sparsity.** Real, and fixed separately, but not this.

Each of those was worth ruling out, and none of them was the cause. The cause was the one
thing every reinforcement learning tutorial says to do and this code had not done.

## Decision

Every global field carries an explicit divisor that brings it into roughly `[0, 1]`, listed
next to the field so the two can never drift apart. The value head is also initialised
small, so the first prediction starts near zero rather than on an arbitrary large number
whose correction would wipe out the policy alongside it.

Rewards are additionally scaled by a running estimate of return standard deviation, which
keeps the critic's targets stable when a task pays 5 per wave and 0.01 per ore.

## Consequences

Measured on the endless task, same configuration, before and after:

| | Before | After |
|---|---|---|
| Value loss | 4,102 | 0.14 |
| Entropy | 3.6, collapsing to 0 in five updates | 11.7, stable |
| Policy loss | 0.0000, frozen | moving, -0.007 to -0.06 |

The spatial tensor was already fine: it is uint8 divided by 255 inside the network, which
is normalisation by another name. Only the scalars were raw.

## What to take from it

"Training runs" and "training learns" are different claims, and only the second one
matters. Throughput, completing episodes and saved checkpoints all looked healthy while
nothing was being learned. Entropy was the metric that told the truth, and it is worth
watching from the first run rather than after five wasted ones.
