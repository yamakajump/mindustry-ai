"""Curriculum tasks.

Each task states what counts as success, how long an attempt may take, and how reward is
earned. Tasks are deliberately small and measurable: when a run fails, the last task that
still passes says which layer broke. See `docs/decisions/0004-curriculum-first.md`.

Reward is kept thin on purpose. Heavy shaping teaches an agent to farm the shaping rather
than to play, and the scripted macro library is the intended source of early guidance.
"""

from __future__ import annotations

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
    reward=_throughput("copper", 0.01),
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
    reward=lambda before, after: float(
        int(after.get("wave", 0)) - int(before.get("wave", 0))
    ),
    success_bonus=50.0,
)

CURRICULUM[GROUND_ZERO.name] = GROUND_ZERO
