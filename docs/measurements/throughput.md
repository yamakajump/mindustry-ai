# Throughput measurements

Every number here is measured, never estimated, and states the hardware it came from.
Reproduce with `python tools/measure_throughput.py`.

## Reference hardware

AMD Ryzen 9 9950X3D (16 cores, 32 threads), NVIDIA RTX 5090 32 GB, 93 GB RAM,
Windows 11, OpenJDK 17.0.18, Mindustry v159.7.

## Method

Two independent figures are recorded per setting, and they must agree:

- **Ticks per second**, counted inside the engine on `Trigger.beforeGameUpdate`, which
  fires once per game update while a match is live.
- **Game time ratio**, derived from `state.tick` before and after the window, divided by
  wall-clock elapsed time.

Either one alone can lie. A loop that spins without advancing the world inflates the tick
count; a larger timestep inflates game time. Only their agreement proves the simulation
genuinely runs faster. In every measurement below they agree to within 1%.

## Single instance, speed sweep

Map `Ancient_Caldera`, survival, 5 second windows, at the start of a match.

| Speed setting | Ticks per second | Ratio (ticks) | Ratio (game time) |
|---|---|---|---|
| 1 | 60 | 1.0x | 1.0x |
| 2 | 120 | 2.0x | 2.0x |
| 4 | 240 | 4.0x | 4.0x |
| 8 | 463 | 7.7x | 7.8x |
| 16 | 690 | 11.5x | 11.6x |
| 32 | 1,143 | 19.1x | 19.2x |
| 64 | 1,122 | 18.7x | 18.9x |
| max | 35,538 | 592.3x | 597.7x |

### Why intermediate speeds plateau at 19x

Settings from 16 to 64 all land near 19x, then `max` jumps to nearly 600x. That is not a
simulation limit, it is the sleep in the headless loop.

Any speed below `max` computes a per-frame budget and sleeps for the remainder. At 64x the
budget is 260 microseconds, but Windows timer granularity is on the order of 1 to 15
milliseconds, so each sleep overshoots by an order of magnitude. `max` sets the interval to
zero, the loop never sleeps, and the ceiling becomes actual CPU throughput.

**Practical consequence: training always runs at `max`.** Intermediate speeds exist for
watching a match at a comprehensible pace, not for throughput.

## Scaling across processes

The engine is not thread safe, so parallelism means separate JVMs. Each instance binds its
own port. 5 second windows, all instances at `max` simultaneously.

| Instances | Aggregate TPS | Per instance | Efficiency |
|---|---|---|---|
| 1 | 44,164 | 44,164 | 100% |
| 4 | 108,520 | 27,130 | 61% |
| 12 | 197,581 | 16,465 | 37% |
| 24 | 288,375 | 12,016 | 27% |

Per-instance throughput falls as instances are added, which is expected: 24 JVMs on 16
physical cores contend for cores, memory bandwidth and cache. Aggregate throughput still
rises by 6.5x, and aggregate is what matters for filling a training batch.

## Agent decisions per second

Raw ticks flatter the system, because they ignore what an environment actually costs: a
socket round trip, pausing and unpausing the world, and only then the ticks. This is the
figure that governs training. Reproduce with `python tools/measure_steps.py`.

300 steps per row, single instance at `max` speed, world frozen between decisions.

| Ticks per step | Steps/s | Game time ratio | Median latency | p99 |
|---|---|---|---|---|
| 1 | 4,052 | 67.5x | 0.21 ms | 0.75 ms |
| 5 | 3,069 | 255.8x | 0.31 ms | 0.59 ms |
| 15 | 2,247 | 561.7x | 0.43 ms | 0.67 ms |
| 30 | 1,327 | 663.6x | 0.73 ms | 1.07 ms |
| 60 | 597 | 597.0x | 1.24 ms | 9.12 ms |

The protocol overhead is negligible: a step advancing a single tick costs 0.21 ms in
total, round trip and pause cycle included. What costs is the simulation itself, which is
the correct place for the time to go.

**30 ticks per decision looks like the sweet spot**, two decisions per game second at 664x
realtime. Fewer ticks per step buys control the game does not reward; more starts losing
game time ratio while making the agent blind for longer.

### With spatial observations attached

Same setup, but the agent negotiated tensors at handshake. `Ancient_Caldera` is 256x256,
and with 14 channels that is 896 KB per observation.

| | Steps/s | Per step | Throughput |
|---|---|---|---|
| Scalars only | 2,247 | 0.43 ms | negligible |
| With 896 KB tensor | 481 | 2.08 ms | 442 MB/s |

Encoding and shipping the world costs 4.7x the step rate, which is the honest price of
letting the agent see. It also scales with map area, so the 32x32 and 64x64 maps the early
curriculum uses are far cheaper: the same tensor there is 14 KB rather than 896 KB.

Full maps at full speed are the case that will not hold. 300x300 at 2,000 steps per second
would need 3.6 GB/s, which is not happening. The answers when it becomes binding are
delta encoding, a local window around the agent, or simply accepting the lower step rate
that a developed base imposes anyway. Not solved, because it is not yet a problem.

## What this implies

At 288,000 aggregate ticks per second, and a ten minute match costing 36,000 ticks:

- **8 full matches per wall-clock second**
- **roughly 28,800 matches per hour**

That is far beyond the "tens of times realtime" this project was designed around. The
curriculum in [`../architecture.md`](../architecture.md) is comfortably trainable, and the
binding constraint is not simulation speed.

## Limits of these numbers

Stated plainly, because a benchmark quoted out of context becomes a lie.

- **Measured on an almost empty map.** These runs are the opening seconds of a match, with
  no factory built. Mindustry cost scales with blocks, items in transit and units alive. A
  developed base will be materially slower, and that measurement has not been taken yet.
- **No agent in the loop.** Nothing is computing observations, encoding tensors, or waiting
  on a policy. Adding the bridge and a neural network will reduce these figures.
- **Single map, single mode.** `Ancient_Caldera` survival only.
- **Windows only.** The 19x plateau is a Windows timer artefact. Linux, with finer sleep
  granularity, would likely make intermediate speeds behave, though it changes nothing for
  `max`.

## Method notes

The acceleration mechanism is documented in
[`../decisions/0006-fixed-timestep-acceleration.md`](../decisions/0006-fixed-timestep-acceleration.md).
No hypothesis was rejected on measurement grounds: reading the engine source showed that
`Core.graphics` is replaceable, which made the two alternatives considered beforehand
(inflating `Time.delta`, running extra logic updates per frame) unnecessary, and both would
have degraded simulation correctness to buy the same speed.
