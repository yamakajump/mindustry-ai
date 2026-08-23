# Baselines

Scores for every policy in the repository, on every curriculum task. This is the table a
learned agent has to beat, and the reason to distrust any claim that is not in it.

Reproduce with `python tools/benchmark.py`.

## T1: copper economy

Grow the core's copper stock to **250** within **450 decisions** of 30 ticks each. The
budget ends just before the first wave lands at tick 14340, so T1 is about economy alone.
A survival start grants 200 copper, so the target is a net gain of 50 after paying for
everything built.

Five episodes per policy, `Ancient_Caldera`, survival.

| Policy | Solved | Mean reward | Mean final copper | Actions accepted |
|---|---|---|---|---|
| Random | 0/5 | -1.98 | 2 | 9% |
| Masked random | 0/5 | -2.00 | 0 | 11% |
| **Alpha** (scripted) | **2/5** | **+4.12** | **212** | **60%** |

### Reading this

**Random policies do not merely score badly, they go bankrupt.** Both finish with
essentially no copper: they spend the starting 200 on blocks placed at random, build
nothing that produces, and then every further action is refused for lack of funds. The
negative reward is the stock they destroyed.

**Masking helps legality, not competence.** The masked policy gets more of its actions
accepted, and still scores marginally worse. Knowing which moves are legal is not the same
as knowing which are good, which is precisely the part that has to be learned.

**Alpha solves it two times in five.** That is the calibration working. A baseline that
always won would make the task useless as a target; one that never won would leave no
evidence the task is solvable at all. The variance comes from how far the nearest copper
happens to be: a distant patch costs a long conveyor line, and the copper spent on it is
copper the target still needs.

### What Alpha does

Find the nearest copper patch, put a mechanical drill on it, route a conveyor line back to
the core, repeat on four patches, then stand still and let the factory run. No defence, no
refinement, no reaction to anything.

It is deliberately unambitious. Alpha exists to be beaten.

## Not yet measured

- **T2** (two ore chains) and **T4** (survive five waves) exist as task definitions but
  have no baseline runs. Alpha mines a single ore and has no defence, so it is expected to
  fail T4 outright; that number belongs here once it has actually been run.
- **Beta**, the first learned agent, has not been built.

## Method notes

Reward on T1 is the change in copper stock times 0.01, plus 10 on success. Spending
resources therefore costs reward immediately and earns it back only if the thing built
produces, which is the tradeoff the agent has to learn. Rewarding gains alone would make
building look free.

Episodes end on success, on losing the core, or at the step limit.
