"""Curriculum tasks.

Each task states what counts as success, how long an attempt may take, and how reward is
earned. Tasks are deliberately small and measurable: when a run fails, the last task that
still passes says which layer broke. See `docs/decisions/0004-curriculum-first.md`.

Reward on the early tasks is thin on purpose: they measure one mechanic each, and a thin
reward is enough to say whether it works. `frontier`, the task that asks for a whole game,
is graded on a ladder of milestones instead. See `MILESTONES` for what that is and
`docs/decisions/0014-milestones-over-potential.md` for why it replaced a potential
function.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

Observation = dict[str, Any]


@dataclass(frozen=True)
class Task:
    """One curriculum stage."""

    name: str
    description: str
    map_name: str
    mode: str = "survival"

    #: Campaign sector to load instead of a custom map. Ground Zero is the first sector
    #: of the Serpulo campaign, and the game itself defines what winning means there:
    #: survive to wave 10. That beats any threshold invented for a custom map.
    sector: str | None = None

    #: Draw a different generated sector every episode, from the planet's own pool.
    #:
    #: This is the difference between learning a map and learning the game. A fixed map is
    #: memorised; several hundred generated ones cannot be. See `gamma/sectors.py`.
    procedural: bool = False

    #: Cap on the game's own difficulty estimate for a sector. The cheapest curriculum
    #: available, because the number is already computed and attached to every sector.
    threat_limit: float = 1.0

    #: Cap on how many worlds to train across. None means the whole pool.
    #:
    #: Hundreds of worlds is what generalisation needs, and it is the wrong place to start.
    #: An agent that cannot learn one world will not learn a hundred, and a flat curve
    #: across a hundred says nothing about which of the two is broken. So this exists to
    #: shrink the pool while the basics are in doubt, and to be removed once they are not.
    #: The held-out half is never touched by it: the measurement stays honest whatever the
    #: training pool is narrowed to.
    worlds: int | None = None

    #: Items placed in the core on load. A sector loaded directly comes up empty.
    loadout: dict[str, int] | None = None

    #: Game ticks the world advances per agent decision.
    ticks_per_step: int = 30

    #: Hard cap on decisions, so a hopeless episode still ends.
    max_steps: int = 400

    #: Returns True when the episode has been won.
    succeeded: Callable[[Observation], bool] = field(default=lambda obs: False)

    #: Returns True when the episode is lost and cannot recover.
    failed: Callable[[Observation], bool] = field(
        default=lambda obs: bool(obs.get("game_over"))
    )

    #: Reward for the transition between two observations.
    reward: Callable[[Observation, Observation], float] = field(
        default=lambda before, after: 0.0
    )

    #: Bonus added once, when the task is solved.
    success_bonus: float = 10.0

    def core_items(self, obs: Observation, item: str = "copper") -> int:
        return int(obs.get("items", {}).get(item, 0))


def _delivered(item: str, target: int) -> Callable[[Observation], bool]:
    def check(obs: Observation) -> bool:
        return int(obs.get("items", {}).get(item, 0)) >= target

    return check


def _carrying(scale: float) -> Callable[[Observation, Observation], float]:
    """Reward what the unit picks up, not only what reaches the core.

    Core stock alone is too sparse for an embodied agent to learn from: it mines for
    hundreds of steps before a single deposit lands, so every reward in a rollout is
    exactly zero and the advantages carry no signal at all. Measured on a real rollout:
    20 steps, 20 rewards, all zero.

    Picking ore up is genuine progress towards the objective rather than invented
    guidance, which is what separates this from shaping the agent can farm. It is
    deliberately worth a fifth of banking, so carrying without ever returning stays the
    worse strategy.
    """

    def reward(before: Observation, after: Observation) -> float:
        gained = int(after.get("unit", {}).get("carrying", 0)) - int(
            before.get("unit", {}).get("carrying", 0)
        )
        # Only the gain: the drop when banking is already paid by the core term.
        return max(0, gained) * scale

    return reward


def _throughput(item: str, scale: float) -> Callable[[Observation, Observation], float]:
    """Reward the change in stock, positive or negative.

    Spending resources on a drill costs reward now and earns it back later, which is
    exactly the tradeoff the agent has to learn. Rewarding only gains would make
    building look free.
    """

    def reward(before: Observation, after: Observation) -> float:
        gained = int(after.get("items", {}).get(item, 0)) - int(
            before.get("items", {}).get(item, 0)
        )
        return gained * scale

    return reward


def potential(obs: Observation) -> float:
    """How well placed the agent is, as one number.

    This is the potential in potential-based reward shaping (Ng, Harada and Russell,
    1999), whose result is that a reward of the form `gamma * phi(s') - phi(s)` leaves the
    optimal policy unchanged. That is what makes it safe to give a dense signal on a task
    whose real objective is sparse: it cannot teach a shortcut, because the shortest path
    to a high potential is the behaviour the potential measures.

    What it measures is the point. A potential over **stock** rewards hoarding, and an
    agent maximising stock mines by hand forever, because hand mining fills the core and
    a drill costs resources to place. A potential over **capability** rewards the machine
    that fills the core, so building the drill pays back within a few steps and keeps
    paying. Nothing tells the agent that drills mine; it is left to notice that the
    quantity it is graded on rises faster when it has them.

    Three terms, all bounded, all functions of the state alone:

    - **A core that is alive.** Everything else is worth nothing without it, so it carries
      the most weight and is the only term that can reach its own maximum.
    - **What has been banked**, on a log so that the first hundred copper matters and the
      ten thousandth does not. A saturating term cannot be farmed indefinitely.
    - **What is standing**, counted as buildings the team owns, on a log for the same
      reason. This is the term that pays for a factory rather than a pile.
    """
    core_max = float(obs.get("core_max_health", 0.0)) or 1.0
    alive = float(obs.get("core_health", 0.0)) / core_max

    banked = sum(int(amount) for amount in obs.get("items", {}).values())
    built = int(obs.get("built", 0))

    return (
        2.0 * max(0.0, min(1.0, alive))
        + 0.6 * math.log1p(banked) / math.log1p(5000)
        + 0.6 * math.log1p(built) / math.log1p(200)
    )


def shaped(
    terminal: Callable[[Observation, Observation], float],
    discount: float = 0.99,
) -> Callable[[Observation, Observation], float]:
    """Wrap a sparse objective in potential-based shaping.

    The discount has to match the one the learner uses, or the guarantee does not hold.
    """

    def reward(before: Observation, after: Observation) -> float:
        return terminal(before, after) + discount * potential(after) - potential(before)

    return reward


# Milestones ---------------------------------------------------------------------------


def _produced(obs: Observation) -> int:
    """Items a machine delivered to the core.

    The engine separates this from hand mining on its own: a conveyor hands items over
    through `handleItem`, which increments this counter, and a unit banking a stack goes
    through `handleStack`, which does not. So it is automated income, measured by the game
    rather than inferred here, and it is the one number that says a factory exists.
    """
    return sum(int(amount) for amount in obs.get("produced", {}).values())


def _variety(obs: Observation) -> int:
    """How many different items a machine has delivered. Breadth, not volume."""
    return sum(1 for amount in obs.get("produced", {}).values() if int(amount) > 0)


def _placed(category: str) -> Callable[[Observation], int]:
    """Blocks fully built in one of the game's categories, cumulative over the episode."""

    def read(obs: Observation) -> int:
        return int(obs.get("placed", {}).get(category, 0))

    return read


def _stat(name: str) -> Callable[[Observation], int]:
    def read(obs: Observation) -> int:
        return int(obs.get("stats", {}).get(name, 0))

    return read


def _wave(obs: Observation) -> int:
    return int(obs.get("wave", 0))


@dataclass(frozen=True)
class Milestone:
    """Something worth doing once, and worth nothing the second time."""

    name: str
    read: Callable[[Observation], int]
    threshold: int
    value: float


#: The ladder the agent is meant to climb, in the order a player climbs it.
#:
#: Every entry reads a **cumulative** counter, so "crossed the threshold on this step" is a
#: complete definition of "first time" and the reward function needs no memory of the
#: episode. It is also what makes the ladder unfarmable: a counter that only goes up pays
#: its milestone at most once, whatever the agent does afterwards. Counting what is
#: standing right now would pay for building and breaking the same conveyor forever.
#:
#: The values are graded by how hard the step is, which is
#: [Crafter](https://arxiv.org/abs/2109.06780)'s reasoning: going from never mining
#: automatically to doing it once matters, going from nine thousand ore to ten thousand
#: does not.
MILESTONES: tuple[Milestone, ...] = (
    Milestone("first_drill", _placed("production"), 1, 5.0),
    Milestone("first_conveyor", _placed("distribution"), 1, 3.0),
    Milestone("first_wall", _placed("defense"), 1, 3.0),
    Milestone("first_turret", _placed("turret"), 1, 5.0),
    # The moment the whole project is about: ore arrived without a hand carrying it.
    Milestone("automation", _produced, 1, 30.0),
    Milestone("automation_100", _produced, 100, 30.0),
    Milestone("automation_1k", _produced, 1_000, 60.0),
    Milestone("automation_10k", _produced, 10_000, 60.0),
    Milestone("two_ores", _variety, 2, 20.0),
    Milestone("three_ores", _variety, 3, 30.0),
    Milestone("first_kill", _stat("enemy_units_destroyed"), 1, 20.0),
    Milestone("held_a_wave", _stat("enemy_units_destroyed"), 25, 30.0),
    Milestone("first_power", _placed("power"), 1, 15.0),
    Milestone("first_crafter", _placed("crafting"), 1, 25.0),
    Milestone("first_factory", _placed("units"), 1, 25.0),
    Milestone("wave_5", _wave, 5, 15.0),
    Milestone("wave_10", _wave, 10, 30.0),
    Milestone("wave_20", _wave, 20, 60.0),
)


def milestones(before: Observation, after: Observation) -> float:
    """What the step crossed for the first time, summed."""
    return sum(
        stone.value
        for stone in MILESTONES
        if stone.read(before) < stone.threshold <= stone.read(after)
    )


def _banked_by_hand(before: Observation, after: Observation) -> int:
    """Ore the agent carried in itself, as opposed to ore a machine delivered.

    Core stock rises for both reasons and falls when something is built, so the automated
    part is subtracted and the remainder floored at zero. Spending on a drill therefore
    reads as zero hand mining rather than as negative, which is right: paying for a
    machine is not the opposite of mining.
    """
    stock = sum(int(a) for a in after.get("items", {}).values()) - sum(
        int(a) for a in before.get("items", {}).values()
    )
    return max(0, stock - (_produced(after) - _produced(before)))


def _build_and_hold() -> Callable[[Observation, Observation], float]:
    """Pay for the things a player does on the way to holding a sector.

    This replaces a potential function that measured how well placed the agent was and
    paid the difference. That design has a theorem behind it: potential-based shaping
    provably cannot change which policy is optimal. It also did not work here, and the
    reason is not subtle. Shaping guides an agent towards behaviour it already produces
    sometimes; it does nothing to make a policy that has never once built a working drill
    start building one. Measured over eleven updates of a real run: entropy 12.067 to
    11.672, policy loss 0.009. Nothing moved.

    What the literature measures on this exact shape of problem is blunter.
    [Huang and Ontanon](https://arxiv.org/abs/2010.03956) ran sparse against shaped reward
    on microRTS, a build-and-fight game on a grid with the same action structure as this
    one. On the task of producing combat units the sparse agent scored **0.00 out of 10**
    and the shaped agent **9.57**, with shaping as unglamorous as +1 per building, +1 per
    resource returned, +7 per combat unit. The
    [Factorio Learning Environment](https://arxiv.org/abs/2503.09617) scores its agents on
    the value of what they *produce* rather than on what they hold.
    [Crafter](https://arxiv.org/abs/2109.06780) pays +1 the first time each of its
    twenty-two achievements is unlocked in an episode, and nothing afterwards.

    All three working designs are the same shape: **first-time events, graded by
    difficulty, plus a measure of flow, plus a small survival term.** None of them is a
    potential function. So that is what this is.

    The farming holes are worth naming rather than hoping about:

    - The milestones read cumulative counters and pay on the crossing, so each is paid at
      most once per episode however the agent behaves.
    - Automated delivery is the one term paid per unit, and the engine's own counter warns
      that an unloader feeding a conveyor back into the core would inflate it. That costs
      titanium and silicon, which needs a pneumatic drill and a powered smelter, which is
      several milestones past anything this agent can currently reach. It is a hole to
      watch for once it can, not a hole today.
    - Hand mining pays a tenth of automated delivery, so carrying ore in by hand is always
      the worse way to earn the same reward.
    """

    def reward(before: Observation, after: Observation) -> float:
        automated = _produced(after) - _produced(before)
        kills = _stat("enemy_units_destroyed")(after) - _stat("enemy_units_destroyed")(before)
        lost = _stat("buildings_destroyed")(after) - _stat("buildings_destroyed")(before)
        damage = float(before.get("core_health", 0.0)) - float(after.get("core_health", 0.0))
        waves = _wave(after) - _wave(before)

        return (
            milestones(before, after)
            # Flow. Ten thousand ore delivered is worth about as much as the whole ladder
            # of milestones, which is the balance we want: the ladder gets it started, and
            # production is what there is left to optimise once it is going.
            + automated * 0.02
            # A tenth of that for ore carried in by hand, and a little for picking it up at
            # all. Without the second term the first thousand steps of a fresh policy carry
            # no gradient whatsoever, because nothing at all happens.
            + _banked_by_hand(before, after) * 0.002
            + _carrying(0.001)(before, after)
            + kills * 1.0
            + lost * -0.5
            + waves * 2.0
            + max(0.0, damage) * -0.005
            + (-50.0 if before.get("has_core") and not after.get("has_core") else 0.0)
        )

    return reward

#: Steps that fit before the first wave arrives on Ancient_Caldera.
#: The wave lands at tick 14340 and a step is 30 ticks, so 450 decisions is the budget
#: an agent gets before defence becomes part of the problem. T1 is about economy alone,
#: so it deliberately ends first.
_PRE_WAVE_STEPS = 450

T1_COPPER = Task(
    name="T1_copper",
    description="Grow the copper stock to 250 before the first wave.",
    map_name="Ancient_Caldera",
    ticks_per_step=30,
    max_steps=_PRE_WAVE_STEPS,
    # Calibrated against Alpha, not guessed. The scripted baseline finishes on 219 to 267
    # copper depending on how many patches it works, starting from the 200 a survival
    # match grants. An earlier target of 400 was unreachable for any policy in this
    # budget, which makes a benchmark that cannot distinguish good from hopeless.
    succeeded=_delivered("copper", 250),
    reward=lambda before, after: (
        _throughput("copper", 0.01)(before, after) + _carrying(0.002)(before, after)
    ),
)

T2_TWO_ORES = Task(
    name="T2_two_ores",
    description="Sustain both copper and lead into the core.",
    map_name="Ancient_Caldera",
    ticks_per_step=30,
    max_steps=_PRE_WAVE_STEPS,
    succeeded=lambda obs: (
        int(obs.get("items", {}).get("copper", 0)) >= 250
        and int(obs.get("items", {}).get("lead", 0)) >= 120
    ),
    reward=lambda before, after: (
        _throughput("copper", 0.01)(before, after)
        + _throughput("lead", 0.02)(before, after)
    ),
)

T4_SURVIVE = Task(
    name="T4_survive",
    description="Survive five waves.",
    map_name="Ancient_Caldera",
    ticks_per_step=30,
    max_steps=3000,
    succeeded=lambda obs: int(obs.get("wave", 0)) > 5,
    reward=lambda before, after: float(
        int(after.get("wave", 0)) - int(before.get("wave", 0))
    ),
)

CURRICULUM: dict[str, Task] = {
    task.name: task for task in (T1_COPPER, T2_TWO_ORES, T4_SURVIVE)
}


def get(name: str) -> Task:
    if name not in CURRICULUM:
        raise KeyError(f"unknown task {name!r}, known: {sorted(CURRICULUM)}")
    return CURRICULUM[name]


def _survive_and_defend() -> Callable[[Observation, Observation], float]:
    """Waves survived, minus the damage the core takes, plus a little for banking.

    The wave counter alone is what the sector calls progress, and it is also entirely
    outside the agent's control: waves arrive on a timer whether it built a turret or sat
    still. Every episode therefore scored exactly the same and there was nothing to learn
    from. Measured over twenty-six generations of a real run: mean reward 1.000, unchanged
    from the first to the last, with entropy still at 11.6 because the policy had no
    reason to move off random.

    Core damage is the term that closes the loop. It is the one number that answers "did
    the defence hold", it cannot be farmed because a core does not heal, and it stays at
    zero for an agent that keeps the enemy away, which is exactly the behaviour the sector
    asks for.

    Banking is worth a fraction of a wave. Without it the first thousand steps carry no
    gradient at all, because nothing happens until the first wave lands, and an agent with
    an empty core cannot build a defence even once it wants to.

    The wave term is kept even though it is nearly constant per unit of time, because it
    is not constant per episode: dying early forfeits every wave that would have followed,
    which is what makes survival worth something.
    """

    def reward(before: Observation, after: Observation) -> float:
        waves = int(after.get("wave", 0)) - int(before.get("wave", 0))

        # A full core-shard is 1100 health, so losing it outright costs a little over
        # three waves. Enough to outweigh the waves it would have collected by standing
        # still and letting the enemy through.
        damage = float(before.get("core_health", 0.0)) - float(after.get("core_health", 0.0))

        banked = sum(
            int(after.get("items", {}).get(item, 0)) - int(before.get("items", {}).get(item, 0))
            for item in ("copper", "lead")
        )

        return waves + max(0.0, damage) * -0.003 + banked * 0.002

    return reward


GROUND_ZERO = Task(
    name="GZ_capture",
    description="Capture Ground Zero: survive to wave 10, as the campaign defines it.",
    map_name="Ground Zero",
    sector="groundZero",
    # The campaign hands the player a launch loadout; a sector loaded directly does not
    # get one, so it is stated here rather than left to a chain of engine conditions.
    loadout={"copper": 300, "lead": 300},
    ticks_per_step=30,
    max_steps=4000,
    succeeded=lambda obs: int(obs.get("wave", 0)) > 10,
    reward=_survive_and_defend(),
    success_bonus=50.0,
)

CURRICULUM[GROUND_ZERO.name] = GROUND_ZERO


ENDLESS = Task(
    name="endless",
    description="Survive as long as possible while producing as much as possible.",
    map_name="Ancient_Caldera",
    ticks_per_step=30,
    # Effectively unbounded: an episode ends when the core dies, not when a counter does.
    max_steps=100_000,
    # No win condition on purpose. A task with a finishing line teaches an agent to reach
    # it and stop; this one has no ceiling, so the only way to score more is to build
    # something that produces more and survives longer.
    succeeded=lambda obs: False,
    failed=lambda obs: bool(obs.get("game_over")) or not obs.get("has_core", True),
    reward=lambda before, after: (
        # Production, counted across every ore rather than one, so the agent is free to
        # decide what is worth mining.
        sum(
            max(0, int(after.get("items", {}).get(item, 0)) - int(before.get("items", {}).get(item, 0)))
            for item in set(after.get("items", {})) | set(before.get("items", {}))
        ) * 0.01
        # Surviving a wave is worth far more than a few ore, because defence is the part
        # that keeps production possible at all.
        + (int(after.get("wave", 0)) - int(before.get("wave", 0))) * 5.0
        # Losing the core ends everything, and should hurt more than any run of good luck.
        + (-25.0 if before.get("has_core") and not after.get("has_core") else 0.0)
    ),
    success_bonus=0.0,
)

CURRICULUM[ENDLESS.name] = ENDLESS


FRONTIER = Task(
    name="frontier",
    description="Land on a world it has never seen, build an economy, and hold it.",
    # Unused: every episode draws its own generated sector.
    map_name="",
    procedural=True,
    # The planet ranges from 0.30 to 0.74. Capping near the bottom gives roughly a third
    # of the pool, which is still well over a hundred worlds, and leaves somewhere to widen
    # to once the agent copes.
    threat_limit=0.40,
    loadout={"copper": 300, "lead": 300},
    ticks_per_step=30,
    max_steps=3000,
    succeeded=lambda obs: int(obs.get("wave", 0)) > 10,
    failed=lambda obs: bool(obs.get("game_over")) or not obs.get("has_core", True),
    reward=_build_and_hold(),
    success_bonus=50.0,
)

CURRICULUM[FRONTIER.name] = FRONTIER


#: The same world pool and the same objective as `frontier`, graded the old way.
#:
#: It exists so the reward change can be measured rather than argued about. Two runs, one
#: task apart, scored on the same held-out worlds against the same random floor. If the
#: milestones are the wrong call this is the task that says so, and keeping it costs a
#: dozen lines.
FRONTIER_SHAPED = Task(
    name="frontier_shaped",
    description="frontier, graded by potential-based shaping instead of milestones.",
    map_name="",
    procedural=True,
    threat_limit=0.40,
    loadout={"copper": 300, "lead": 300},
    ticks_per_step=30,
    max_steps=3000,
    succeeded=lambda obs: int(obs.get("wave", 0)) > 10,
    failed=lambda obs: bool(obs.get("game_over")) or not obs.get("has_core", True),
    # The sparse objective is the wave counter, exactly as the campaign defines capture,
    # and everything dense comes from the potential. Safe by construction: the shaped term
    # provably cannot change which policy is optimal. Safe and, measured over eleven
    # updates, entirely inert.
    reward=shaped(
        lambda before, after: float(
            int(after.get("wave", 0)) - int(before.get("wave", 0))
        )
    ),
    success_bonus=50.0,
)

CURRICULUM[FRONTIER_SHAPED.name] = FRONTIER_SHAPED
