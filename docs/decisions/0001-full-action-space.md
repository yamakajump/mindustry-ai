# 1. Full action space, factored and masked

Status: accepted
Date: 2026-08-23

## Context

The agent needs an action space. Mindustry offers roughly 200 block types placeable on
maps up to 500x500 tiles, with rotation and per-block configuration, plus unit commands
and direct avatar control. Flattened, that is on the order of 10^8 discrete actions. No
policy learns over a flat space that size.

The tempting shortcut is a hand-written vocabulary of perhaps 50 high-level actions
("place a drill on the best ore patch", "connect A to B"). It converges quickly and it is
what most prior attempts reached for.

It also caps the agent at the skill of whoever wrote the vocabulary. An agent meant to
beat strong human players cannot have its ceiling set by a hardcoded list.

## Decision

The action space covers **every action a human player can perform**, with no exception.

It is made learnable by two mechanisms rather than by removing capability:

**Factoring.** One action is decomposed into dependent components, each with its own
network head, sampled autoregressively: type, block, position, rotation, configuration,
macro index. The policy makes several small decisions instead of one impossible one. This
follows AlphaStar.

**Legality masking.** Every head is masked to legal choices before sampling, using ground
truth computed by the engine itself. The tiles where a given block can legally be placed
number in the hundreds, not the tens of thousands. The effective space collapses while
expressiveness stays intact.

Scripted macros survive as **one additional action type**, not as the whole vocabulary.
The agent may invoke them or ignore them.

## Consequences

- Observations must be spatial, since the position head needs a per-tile probability map.
  That rules out a compact vector observation and requires a convolutional trunk.
- Stable-Baselines3 becomes unsuitable as the primary trainer (see decision 5).
- First meaningful result moves from days to weeks.
- Macro usage rate becomes a headline metric: when it falls, the agent has surpassed the
  human strategies it was given.
- The bridge grows a real responsibility, computing legality masks every step, and that
  computation sits on the hot path.

## Alternatives rejected

**Hand-written macro vocabulary only.** Converges fastest, and was the initial
recommendation in design discussion. Rejected because the ceiling is set by the author,
which is incompatible with the stated goal of beating human players.

**Raw flat action space, no factoring.** Maximally pure, no human prior at all. Rejected
as not learnable on single-machine hardware within any reasonable horizon.

**LLM writing code against a game API**, in the style of the Factorio Learning
Environment. Produces demos fast and costs no GPU. Rejected because nothing learns: the
model reasons, it does not improve. That is a different project.

## References

- [Gym-uRTS](https://arxiv.org/abs/2105.13807), which demonstrates that invalid action
  masking is the decisive factor for learning in a grid RTS.
