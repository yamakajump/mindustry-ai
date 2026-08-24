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

A team plate is drawn the way the engine chooses it, and the distinction matters: a team
with its own painted plate gets that plate as it ships, and only a team without one gets
the generic plate multiplied by its colour. Multiplied, never filled. Filling the
silhouette with a flat colour throws away the bevels and the shading, which is what turned
a core into a grey box with nothing on it.

Coverage is checked against what actually appears rather than asserted. Across the maps in
use that is twenty-nine block types, floors, ores, props, static walls and buildings, and
every one of them resolves to a sprite with its variants, its edge sheet, its large sheet
and its overlays as applicable.

Sprites cover the whole game rather than the subset in use, and are fetched one at a time
as something first needs one. Packing them would mean a five-thousand-pixel atlas costing a
hundred megabytes of memory to carry blocks no run will ever build; a match draws a few
dozen, and a fresh dashboard settles at seventy-six loaded images out of fifteen hundred
groups. The point is not the saving, it is that the day the agent unlocks a new tier
nothing needs regenerating: the sprite is already there and loads the first time it is
drawn. They are extracted from the asset jar by `tools/extract_sprites.py`, which the
training entry point runs when they are missing, and they are not committed, because four
thousand files that regenerate in twenty seconds are a build artifact.

What is deliberately not reproduced: the legs of walking units, the barrel parts of a
turret, the laser links between power nodes, and the flame and glow drawers of the crafting
blocks, all of which need per-block state the bridge does not send yet. A block with no
region of its own, a salvo say, falls back to the composed icon the game generates for it,
which is the whole machine in one piece rather than nothing. Item flow through a junction
is not drawn because the game does not draw it either. They are listed so the next person
knows they were considered rather than missed.
