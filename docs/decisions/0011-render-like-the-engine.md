# 11. Reproduce the engine's rendering, do not approximate it

Status: accepted
Date: 2026-08-24

## Context

The live view drew the map with the game's own sprites and still did not look like the
game. Every difference had the same shape: a rule had been guessed instead of read.

- Floors repeated in a visible weave, because one variant of three was packed and tiled.
- Transitions between floors were a field of decorative hooks scattered at random, because
  the direction-to-cell mapping of the edge sheet was assumed rather than checked.
- Rock fields were a checkerboard of the same two glyphs, because static walls were drawn
  tile by tile.
- Nothing cast a shadow and no cliff had a dark interior, because those are two whole
  passes that were simply absent.
- A belt was a painted arrow, a drill never turned, a unit had no thruster and carried
  nothing visible, and ore arriving at the core just changed a number.

Each of those is small. Together they are the difference between a picture of the game and
the game.

## Decision

Read the renderer and reproduce what it does, rather than write something that looks close.
Where a rule could not be read with certainty, measure it off the asset instead of picking
the likely answer.

The pipeline now mirrors `BlockRenderer`, in its order, because the order is load-bearing:

1. **Floors**, variant chosen per tile, then the edge sheets of any neighbour with a
   higher blend group laid over them.
2. **The shadow**, one texel per tile stretched with linear filtering and multiplied in.
   This is the engine's own trick and it is why a base sits on the ground instead of
   floating over it. It costs one blit however large the base grows.
3. **Static walls**, from the 2x2 sheet the engine stamps across aligned pairs.
4. **Buildings**, each layered as the engine layers it: base, the part that turns, the
   housing, the team plate, then the cracks a damaged one wears.
5. **Belt items, units, shots, effects.**
6. **The darkness**, over everything, black deep inside rock and faded at the border.

Two rules were measured rather than assumed:

- **The edge sheet is transposed and upside down.** Its top-left cell carries the fade
  towards the south-east. Found by taking the centre of mass of each of the nine cells.
- **The large wall sheet reassembles as it is stored**, column with x and row against y,
  which is provable because the sheet is the finished rock.

Two more are now taken from the engine at runtime rather than recomputed:

- **The blend group** of every floor, because comparing raw block ids blends a good part of
  a map the wrong way round.
- **The autotiler's answer** for every conveyor, because deciding again on the other side
  draws a junction where the game draws a straight run.

## Consequences

The bridge sends more: shadow and darkness flags per block, blend groups, conveyor shapes
and the items on them, turret aim and recoil, the unit the agent inhabits, what it is
carrying, and where each engine sits on each unit type. All of it is state the engine
already holds, so none of it is invented.

Effects are the exception, and the line is worth stating. A headless server never runs
`Effect.at`, so mining sparks, the square that opens when a block is finished, and the
burst when one is knocked down are played from facts the viewer already has: the agent is
mining, the building reached full progress, the building vanished. The one effect that
could not be inferred, ore handed to the core, is recorded by the bridge at the moment it
happens and sent. Nothing is drawn that did not occur.

What is deliberately not reproduced: the legs of walking units, the barrel parts of a
turret, and the item flow through a junction, which draws nothing in the game either. They
are listed here so the next person knows they were considered rather than missed.
