# 15. Search for the structures, learn where to put them

Status: accepted
Date: 2026-08-24

## Context

A drill on ore delivers nothing. A drill with a conveyor line to the core delivers
everything. Nothing in between delivers anything at all, and that is the whole difficulty
of this game: value lives in connected structures and every partial one is worth exactly
zero.

The policy was asked to find those structures one tile at a time, and it did not.
Measured over 177 archived episodes of a real run, with `tools/analyse_replays.py`:

| | |
|---|---|
| episodes | 177 |
| drills placed | 4,481 |
| conveyors placed | 5,719 |
| episodes where tiles ever met end to end, drill to core | **1** |

The count is bracketed by two readings, one honouring rotations and one ignoring them
entirely, and both give one. The five best-scoring episodes of the run, at +358 to +398,
contain no chain and no drill touching the core: they are several hundred blocks connected
to nothing.

The arithmetic says this was never going to work. `tools/analyse_game.py` reads the game's
own content definitions: a drill only produces on ore, roughly one tile in thirty-three,
and a ten-tile conveyor line is one chance in a million **on rotations alone**, before
choosing which ten of 2,304 tiles. Expected correct lines per 3,000-step episode: 0.002.

Nor is that the top of the ladder. The game has 18 factories and chains four deep, where
nothing partial pays: a silicon smelter eats coal *and* sand *and* power and produces
nothing until all three arrive at once.

## Decision

**Search for the structures. Learn where to put them.**

Two stages that do different jobs, because they are good at different things:

1. **Discovery**, in [mindustry-forge](https://github.com/yamakajump/mindustry-forge). A population of
   layouts, each one stamped into a real game and scored on what it actually delivers to
   the core. Selection, crossover, mutation. Nothing in it knows what a conveyor is for.
2. **Play**, the learned policy, which keeps the part it is good at: which world, which
   patch, when to expand, when to defend.

This split is not invented here. [Reid et al.
(2021)](https://arxiv.org/abs/2102.04871) benchmarked simulated annealing, genetic
programming and evolutionary reinforcement learning on precisely this subproblem in
Factorio, the "logistic transport belt problem", and found search beating general-purpose
learning on it.

**The layouts are the agent's own.** No community schematic is imported and none will be.
Copying a blueprint from someone who already knew the answer would automate a human rather
than beat one, and would prove nothing. The rules of the game are the entire fitness
function. `Schematics.create` exists in the engine, so a design the search discovers can
be captured and handed back as an action, which is how the vocabulary grows without anyone
writing it.

**How a layout is written matters more than how it is searched.** Two genomes were built
and measured against each other on the same world, the same budget and the same seed:

| genome | after 25 generations |
|---|---|
| **cells**, one square at a time | **0 ore**, 0 of 40 delivering, 110 blocks of noise |
| **parts**, drills and lines | **30 ore**, 17 of 40 delivering, still climbing |

A line in the parts genome costs one gene and its rotations are derived from the direction
of travel, so it is correct by construction. That is the same idea as the wave function
collapse in [this walkthrough of evolving Factorio
blueprints](https://www.youtube.com/watch?v=mGOKKtIDNbk), applied to the genome instead of
to the generator: do not make the search cleverer, make the invalid impossible to write
down.

## Where the search went

It left. The search lives in [mindustry-forge](https://github.com/yamakajump/mindustry-forge) and this
repository no longer carries it.

The two turned out to be different jobs with incompatible loops. The search asks what the
best machine for a job is and answers in minutes; this project asks where and when to
build one and answers in hours. Keeping both here meant one repository whose test suite,
readme and roadmap described two audiences, and a search engine that an agent must never
call anyway: a player will wait ten minutes for a good design, an agent mid-match cannot
wait at all.

So the forge fills a warehouse offline and the agent shops from it.
[`gamma/library.py`](../../gamma/library.py) reads that warehouse and
[`gamma/adapt.py`](../../gamma/adapt.py) puts what it finds where the ore actually is.
Those two are agent concerns and stay. The contract between the repositories is the stored
design and nothing else.

## Consequences

The bench has to be built with more care than the search. Three times it answered a
question nobody asked, and each time the search was right and the bench was wrong:

- **Scoring any delivered item** made the answer sand. It covers 3,848 tiles of this map
  against copper's 1,339 and some of it sits against the core, so the winner was eight
  drills and not one conveyor. Correct, optimal, and silent on whether a line can be
  found. The bench now scores one named ore.
- **Ore next to the core** makes a line unnecessary: `Drill.offload` pushes into any
  adjacent building and a core is one. The bench now requires ore several tiles out.
- **Sparing everything within two tiles of the core's centre** protects a five-by-five
  square when a core shard is three across, so the ring around it was never clearable and
  never buildable. The last tile of every line, the one that has to touch the core, was
  silently skipped, and the bench was unwinnable however right a design was. The footprint
  is now read off the map.

**A hint that pays better than the goal is not a hint.** Ore stuck inside a design is what
lifts an incomplete line out of the flat zero it shares with an empty rectangle, and
without it the search has nothing to climb: twenty-five generations, both genomes, nothing
delivered, the population shrinking towards four blocks. Uncapped, the same term was worth
more than the objective. Over eighty generations the population settled at a mean of 182
of which **89% was ore going nowhere**, against about 21 delivered: it had stopped building
lines and started hoarding. It is capped now, a test holds the cap, and the search's own
progress line prints stuck ore beside the score so the next such drift is visible in one
glance rather than in an afternoon.

The cost per block is a knob rather than a constant, for the same reason. At 0.05 it was
set when a good design delivered 3 ore, so a block cost more than it earned and building
nothing was optimal. At 0.01 it stopped mattering once deliveries reached 25, and the
winner sprawled from 19 blocks to 78 for four more ore, at a third of the efficiency. It
has to be read against what a delivery is currently worth.

What this does not yet do is decide anything about the game. The search finds a shape that
delivers on one patch of one world; it says nothing about when to build it, whether to
defend it first, or which of several patches is worth the trip. That is the policy's job,
and giving it these designs as a vocabulary is what
[decision 14](0014-milestones-over-potential.md)'s milestone ladder was reaching for and
could not get to on its own.
