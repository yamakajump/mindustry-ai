<div align="center">

# mindustry-ai

**Teaching machines to play [Mindustry](https://github.com/Anuken/Mindustry).**

An open reinforcement learning environment for the automation tower-defense RTS,
plus the agents that learn to survive, build and win in it.

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-early%20development-orange.svg)](#roadmap)
[![Mindustry](https://img.shields.io/badge/Mindustry-v8-orange.svg)](https://github.com/Anuken/Mindustry)

</div>

---

> [!WARNING]
> **Early development. No agent plays the game yet.**
> This README describes what the project is for and reports honestly on where it
> actually stands. Every claim about agent performance in this file is backed by a
> reproducible run, or it is not in this file. See the [roadmap](#roadmap).

## Why Mindustry

[Mindustry](https://github.com/Anuken/Mindustry) is a real-time strategy game where you
mine resources, route them through conveyor networks, refine them into higher tier
materials, and spend the output on turrets and units. Then waves of enemies come to
take it apart.

That makes it an unusually good testbed for machine learning, because winning requires
three skills that rarely appear together in one game:

- **Long-horizon economics.** A silicon smelter pays for itself in ten minutes, not now.
  An agent that only optimises the next second never builds one.
- **Spatial reasoning.** Conveyor layouts, turret coverage and factory footprints are
  geometry problems on a grid, not a list of build orders.
- **Reactive combat.** All of the above has to survive contact with an enemy that
  actively destroys it.

Factorio has [a learning environment](https://arxiv.org/abs/2503.09617). StarCraft II has
[pysc2](https://github.com/google-deepmind/pysc2). Mindustry has nothing. This project is
that missing piece.

## What makes it hard

Being honest about the difficulty up front, because it shapes every design decision here.

| Obstacle | Why it hurts |
|---|---|
| **Enormous action space** | Roughly 200 block types, on maps up to 500x500 tiles, times rotation and configuration, plus per-unit commands. Wider than StarCraft II. |
| **Slow simulation** | Mindustry simulates thousands of blocks and item movements per tick, and cost grows with the factory. Less of a problem than expected: [measured](docs/measurements/throughput.md) at 592x realtime per instance on an empty map. |
| **No thread safety** | The engine is single threaded by design. Parallelism means many JVM processes, not many threads. |
| **Sparse, delayed reward** | The payoff for a good factory arrives minutes after the decisions that built it. |
| **Nothing was built for this** | No observation API, no fast reset, no metrics. That plumbing is most of the work. |

## How it works

Three processes, split by language boundary. A JVM mod inside the Mindustry headless
server exposes state and executes actions. A Python package holds the Gymnasium
environment and the training loop. A static web viewer replays matches.

The design choice that matters most: **the agent can perform every action a human player
can**, with no hand-written vocabulary capping what it is allowed to try. That space is
made learnable by factoring one action into dependent components with their own network
heads, and by masking every head to legal choices computed by the engine itself. Scripted
macros exist, but as one action among many rather than as the whole repertoire, so nothing
stops the agent from ignoring them and inventing something better.

Full details in [`docs/architecture.md`](docs/architecture.md). The reasoning behind each
choice, including the alternatives that were rejected, is in
[`docs/decisions/`](docs/decisions/).

## Watch training happen

```bash
python tools/train_watch.py --matches 6 --embodied
```

Opens a dashboard on `127.0.0.1:8800` with every match running side by side: a mini map
per match showing the core, the agent's unit and everything it has built, plus step,
wave, reward, refusals, core stock, a progress bar against the objective, a leaderboard
and a trend line over recent episodes.

Each match is its own Mindustry process, because the engine is not thread safe. No
dependencies beyond the standard library: a dashboard that needs a web framework
installed is a dashboard that is not running when you want it.

## Watch it play

Two ways, and they are not equivalent.

### In the game, with a remote in your hand

```bash
python tools/install_mod.py
```

Restart Mindustry and press **AI replay** in the main menu. Pick an episode and watch it
play out with a transport bar: play and pause, 0.5x to 8x, step forward and back, scrub
anywhere on the timeline, jump to the core. Space and the arrow keys work too. You keep
full control of the camera the whole time, because it is your game running.

Everything the engine draws is there, conveyor animations, items sliding along them,
block connections, lighting, wave effects, because the game itself is rendering it.

Rewinding is the part worth explaining: a simulation cannot be run backwards. The mod
keeps a full snapshot of the world in memory every forty steps, in the engine's own save
format because that is the only thing that captures items in transit, units and wave
timers. Going back restores the nearest one and replays forward from it.

The installer finds your Mindustry on its own. The Steam build keeps its data inside its
install folder rather than in the usual application data path, which is an easy way to
end up staring at "no mods found".

### Live, over the network

```bash
python tools/watch.py replays/showcase/alpha-t1.jsonl.gz
```

Then **Play → Join Game → 127.0.0.1:6567**. No mod needed: the training server already
speaks Mindustry's network protocol, so a stock client can join it. Less control than the
mod, but this one also works **live during training**, not just on recordings.

### In a browser, for sharing

**[Alpha building a copper chain](https://yamakajump.github.io/mindustry-ai/?replay=replays/alpha-t1.jsonl.gz)**
 | [a random policy going bankrupt](https://yamakajump.github.io/mindustry-ai/?replay=replays/random-t1.jsonl.gz)

Drawn with the game's own sprites, pan and zoom, no install. Being honest about it: this
is a map, not the game. There are no animations, conveyors do not connect visually, and
there is no lighting. It exists so a replay can be shared as a link, which the real client
cannot do.

A full 450 step episode is about 16 KB, because only the typed map and the actions taken
are stored, not the state. The same format will carry live training over a WebSocket. See
[decision 3](docs/decisions/0003-one-replay-format.md).

## Where it stands

The first curriculum task is live, with a benchmark. `T1` asks an agent to grow the core's
copper stock to 250 within 450 decisions, which is the budget before the first wave lands.

**Direct mode**, where the agent edits the world without a body:

| Policy | Solved | Mean reward | Actions accepted |
|---|---|---|---|
| Random | 0/5 | -1.98 | 9% |
| Masked random | 0/5 | -2.00 | 11% |
| **Alpha** (scripted) | **2/5** | **+4.12** | **60%** |

**Embodied**, where it inhabits a unit and plays under a player's limits: it has to fly to
the ore, mine by hand, carry the load back, and can only build within about 27 tiles.

| Policy | Mean final copper | Mean reward |
|---|---|---|
| Masked random | 0 | -2.00 |
| **Embodied Alpha** | **192** | **-0.08** |

Acceptance rate stops being informative once there is a body: `move` is nearly always
legal, so a policy can be accepted 95% of the time while bankrupting itself.

Random policies do not merely score badly, they go bankrupt: they spend the starting
copper on blocks placed at random, build nothing that produces, and every later action is
refused for lack of funds. Masking gets more actions accepted and still scores no better,
because knowing which moves are legal is not knowing which are good. That gap is the part
that has to be learned.

Alpha solving two in five is the calibration working. A baseline that always won would
make the task useless as a target; one that never won would leave no evidence it is
solvable. Full numbers and method in [`docs/measurements/baselines.md`](docs/measurements/baselines.md).

## Milestones

Mindustry players do not have a fixed avatar. They embody a core unit that upgrades with
their core: **Alpha**, then **Beta**, then **Gamma**. The agents follow the same path,
because it maps cleanly onto how capable they actually are.

| Milestone | What it is | Status |
|---|---|---|
| **Alpha** | A scripted baseline. It plays, badly but predictably. The yardstick everything else is measured against. | **solves T1 two times in five** |
| **Beta** | The first agent that genuinely learns. Beats Alpha. | not started |
| **Gamma** | An agent that beats competent human players. | the point of all this |

The name is a double meaning worth stating once: in Mindustry, Gamma is the final form of
the player's own avatar. In reinforcement learning, gamma is the discount factor, the term
in the Bellman equation that decides how much an agent values the future over the present.
Building a factory that pays off in ten minutes is exactly a high-gamma problem.

## Roadmap

Ordered by dependency, not by excitement.

- [x] **Bridge foundation.** A JVM plugin loaded by the headless server, driven from Python, with simulation speed under control.
- [x] **Throughput benchmark.** [Measured](docs/measurements/throughput.md): 592x realtime on one instance, 4,806x across 24.
- [x] **Bridge protocol.** A socket speaking length-prefixed frames, with the world frozen between decisions. [Measured](docs/measurements/throughput.md) at 2,247 decisions/s.
- [x] **Observations.** Spatial tensors on the binary frame type, 14 channels, plus scalars. [Measured](docs/measurements/throughput.md) at 481 steps/s with a 896 KB tensor.
- [x] **Actions.** Placing and breaking blocks, priced against the core, refused by the engine when illegal.
- [x] **Environment.** Gymnasium-compatible: factored action space, masks for every head, reward, curriculum tasks.
- [x] **Alpha.** The scripted baseline, and a [benchmark](docs/measurements/baselines.md) to score anything against it.
- [ ] **Beta.** First learned agent. Clears the early curriculum stages and beats Alpha on them.
- [ ] **Replays.** An event log format, and a web viewer that replays a match tile by tile.
- [ ] **Full games.** Survival on real maps, the whole economy and defence loop.
- [ ] **Generalisation.** Procedurally generated maps, so the agent learns to play rather than to memorise.
- [ ] **Self-play.** Attack and PvP modes, agents trained against each other.
- [ ] **Gamma.** Beat humans.

## Development

Requires **Java 17** and **Python 3.13**. Nothing else: the Mindustry server is downloaded
on demand, pinned to `v159.7`, and Gradle arrives through the wrapper.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install pytest    # .venv/bin/python on Linux and macOS
.venv/Scripts/python -m pytest tests/ -v
```

The tests start real Mindustry servers and drive them through stdin and the agent
socket. There are no mocks, because engine state only exists inside a running engine and a
mock of it would assert nothing. A full run takes about 45 seconds.

Throughput measurements are marked `perf` and excluded by default: a second server busy on
another core skews them. Run them alone with `pytest -m perf`.

To reproduce the throughput figures:

```bash
cd bridge && ./gradlew jar && cd ..
.venv/Scripts/python tools/measure_throughput.py --window 5
```

Add `--skip-scaling` to measure a single instance only. Results and their caveats live in
[`docs/measurements/throughput.md`](docs/measurements/throughput.md).

Once a server is running with the plugin installed, console commands are available:
`bridge-status` reports version, clock state, tick and wave; `bridge-bench <seconds>`
measures throughput over a window; `bridge-speed <n|max>` sets simulation speed;
`bridge-port` reports the agent socket.

Running the benchmark:

```bash
.venv/Scripts/python tools/benchmark.py --episodes 5
```

Driving a game from Python:

```python
from gamma.bridge import Bridge

with Bridge(port=7654, tensor=True) as bridge:
    bridge.reset("Ancient_Caldera", "survival")
    for _ in range(100):
        obs = bridge.step(repeat=30)     # world runs 30 ticks, then freezes again
        print(obs["tick"], obs["wave"], obs["items"])

        spatial = obs["spatial"]         # uint8 array, (channels, height, width)
        copper = spatial[bridge.channels.index("ore_copper")]
```

The port defaults to 7654 and is set per instance with `-Dmindustryai.port=N`, which is how
parallel environments avoid colliding.

Or through the Gymnasium environment, which manages the server for you:

```python
from gamma import tasks
from gamma.env import MindustryEnv
from gamma.alpha import AlphaPolicy
from gamma.policies import run_episode

env = MindustryEnv(tasks.T1_COPPER, jar="bridge/build/libs/mindustry-ai-bridge.jar")
print(run_episode(env, AlphaPolicy(env)))
env.close()
```

## Prior art

Several people have started down this road. None have arrived, which is the main reason
this project exists. Recorded here so nobody repeats the same five days of work.

| Project | Stars | Lifespan | What happened |
|---|---|---|---|
| [mindustry-ia-inteface-mod](https://github.com/LyeZinho/mindustry-ia-inteface-mod) | 1 | 5 days | TCP socket exposing game state. Closest to a real bridge. Abandoned. |
| [ReinforcementLearning_MindustryMod](https://github.com/RaptorNava/ReinforcementLearning_MindustryMod) | 1 | 1 day | Copper delivery with drills and conveyors. Never went further. |
| [SLAIdustry](https://github.com/Sisyph-TC/SLAIdustry) | 0 | none | Empty repository. |
| [Mindustry-Mate](https://github.com/QuantumPickleJar/Mindustry-Mate) | 0 | dead since 2023 | "Will eventually employ reinforcement learning." It did not. |
| [Mindustry-MCP](https://github.com/DeterMination-Wind/Mindustry-MCP) | 3 | active | An MCP server driving the game with an LLM. Different goal, no learning, but the most alive of the bunch. |

If you are working on something in this space, open an issue. Duplicated effort is how
this problem stayed unsolved.

## License

[GPL-3.0](LICENSE), inherited from Mindustry, which this project links against.

Mindustry is created by [Anuken](https://github.com/Anuken) and is not affiliated with
this project.
