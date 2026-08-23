# 6. Fixed timestep, with speed controlled by frame budget

Status: accepted
Date: 2026-08-23

## Context

Training needs the simulation to run much faster than realtime, and Mindustry offers no
supported way to do that. The headless server targets 60 ticks per second and sleeps for
the remainder of each frame.

Three approaches were considered before reading the engine source, ordered by how much
damage they risked doing to simulation correctness:

1. Run extra `Logic.update()` calls per frame.
2. Inflate `Time.delta` so each tick advances more game time.
3. Shorten the headless loop sleep interval.

Reading the source made all three obsolete, and revealed a better option.

## Decision

Two independent mechanisms, both installed by the bridge plugin at load time.

**A fixed timestep**, by replacing `Core.graphics` with a `MockGraphics` subclass whose
`getDeltaTime()` returns a constant `1/60`.

This works because Mindustry derives simulation speed from `Core.graphics.getDeltaTime()`
in three separate places: `Time.deltaimpl` computes `Time.delta` from it, `Time.updateGlobal`
advances global time with it, and `Logic.update` advances `state.tick` with it. Replacing
the graphics implementation makes all three consistent in one move. `Time.setDeltaProvider`
is public and was the obvious candidate, but it only fixes the first of the three, leaving
`state.tick` advancing at wall-clock rate while the world advances at another.

**A frame budget**, by writing `HeadlessApplication.renderInterval` through reflection.
The field is not final and is read fresh on every loop iteration, so writes take effect
immediately. Zero removes the sleep entirely.

Speed is therefore controlled purely by how often frames occur, never by how much each
frame advances the world.

## Consequences

- **Measured 592x realtime on one instance, 4,806x aggregate across 24.** See
  [`../measurements/throughput.md`](../measurements/throughput.md).
- **The timestep is constant**, which is strictly better than stock behaviour. Variable
  timesteps make runs irreproducible and degrade collision handling under load. This
  removes one of the open questions about determinism raised in the architecture document,
  though determinism as a whole remains unverified.
- **Reflection creates a version dependency.** If Arc renames `renderInterval`, the clock
  degrades. `bridge-status` reports `clock=ok` or `clock=degraded` and a test asserts on it,
  so the failure is loud rather than silent.
- **Intermediate speeds are useless on Windows**, capping near 19x because of sleep
  granularity. Training uses `max`; intermediate speeds are for human observation only.
- `Time.globalTime` still advances from real elapsed time in `updateGlobal`. It drives
  visual effects, so it does not matter headless, but it is inconsistent with everything
  else and would matter if a client ever attached.

## Alternatives rejected

**Inflating `Time.delta`.** Same speed, but units move further per step and can tunnel
through walls and each other. It buys throughput by corrupting the thing being measured.

**Extra logic updates per frame.** Desynchronises anything driven by frame counts rather
than by delta, and doubles the work of any system that assumes one update per frame.

**A custom server launcher** that constructs `HeadlessApplication` with a different render
interval. Avoids reflection, but requires shipping and maintaining a forked launcher rather
than a plugin the stock server loads, for a mechanism reflection reaches in six lines.
