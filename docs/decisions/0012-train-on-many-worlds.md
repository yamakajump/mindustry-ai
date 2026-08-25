# 12. Train on many worlds, and hold some back

Status: accepted
Date: 2026-08-24

## Context

Every task in the curriculum until now named one map. `T1_copper` is Ancient Caldera,
`GZ_capture` is Ground Zero. An agent trained on one map learns that map.

This is the best established result in the field rather than a suspicion.
[CoinRun](https://arxiv.org/abs/1812.02341) and
[Procgen](https://arxiv.org/abs/1912.01588) both measure it directly: an agent given a
fixed set of levels memorises them, the gap between training and unseen performance is
large, and it closes only as the training set runs into the hundreds. Regularisation helps
at the margin and does not substitute for the levels.

Worse, the failure is invisible from inside. A reward curve rising on the maps an agent
trained on is exactly what memorisation produces. Without a held-out set there is no
measurement in the project that can tell the two apart, which means every number reported
so far answers a question nobody was asking.

## Decision

**Every episode draws a different generated world.** Serpulo is a planet of 272 sectors and
194 of them have no hand-made preset, so the game's own planet generator produces them:
their own terrain, their own ore, their own size. Capping on the game's own threat estimate
at 0.40 leaves 147, which is on the right side of where Procgen sees generalisation begin.
The generator is the game's, so this costs no generation code and no invented terrain.

**A fifth of the pool is held back and never trained on.** `tools/evaluate.py` scores a
checkpoint on those worlds against a masked random floor. That is the only measurement here
that separates learning from memorising, and it is the number worth quoting.

The split is a seeded shuffle rather than every fifth index, because sector indices follow
the planet's grid: consecutive ones are neighbours and share a biome. Taking a stride would
hand the evaluation set one region, and the score would mean "a kind of world it has not
seen" rather than "a world it has not seen".

Three things had to change for a generated sector to be playable at all, each found by
loading one:

- **The engine keeps team data across a world load.** The core of the previous map answered
  every question about where the base was, on a map where it did not exist. This was not a
  procedural-sector problem: it corrupted every episode reset in the project, and the local
  window has been centring on a phantom core for as long as there have been episodes.
  `logic.reset()` before loading, which is what the game does.
- **A generated sector was thought to have no core.** In the campaign you land on one and
  the launch places your loadout, so loading it directly seemed to give terrain, spawns and
  no way to play, and a core shard was placed on the first clear ground near the middle.
  That is no longer what happens, and may never have been: probed on four generated
  sectors, the placement never runs, because a core is already there when it is asked to.
  `ensureCore` is dead code on this path. It is left in place because it costs nothing and
  a sector that genuinely arrived without one would still be playable, but nothing should
  be built on top of it without checking first. A curriculum that wanted the base landed
  against ore was, and it did nothing at all.
- **Enemy bases are stamped from a registry a headless server never loads**, and the
  generator walks into a null item. They are switched off; the wave spawns remain, and the
  pressure comes from the waves.

## Consequences

The numbers will get worse before they mean anything. An agent that generalises scores
lower on any single map than one that memorised it, and the training curve now averages
over worlds of varying difficulty rather than repeating one. That is the correct trade and
it should be expected rather than debugged.

The action space was already independent of the map: the local window crops observations
to a fixed size around the agent and translates positions back on the way out. So a map of
430 tiles and one of 334 look identical to the policy, and nothing about varying sizes
reaches the network.

What this does not yet do is choose *which* world to play next. Uniform sampling over the
pool is wasteful once some worlds are solved and others are hopeless.
[Prioritized Level Replay](https://arxiv.org/abs/2010.03934) samples by how much the agent
still has to learn from a level, and [ACCEL](https://arxiv.org/abs/2203.01302) evolves
levels at the frontier of its ability. Both beat uniform sampling on Procgen by a wide
margin. The pool and the split are what they would be built on, and neither is worth adding
before the held-out number says the basics work.
