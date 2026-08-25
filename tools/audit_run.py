"""Look for the ways a reward gets farmed, before a run is believed.

    python tools/audit_run.py

Three exploits were found in one session, and every one of them produced a rising score
while the agent got no better at the game:

- a generator paid per step, an annuity worth 1,400 of one episode's 1,577 points
- a stockpile handed over in a single step, worth 22.1% of a whole run
- one reward ledger shared by twenty-four threads, reporting 1,515 core losses in 225
  episodes and attributing them to the wrong steps

None of them was visible in the training curve, which is the point. All three have a
signature in the archived episodes, and so do the next ones of their kind, so the checks
live here rather than in whatever a person happens to remember to look at.

Each check prints OK or the evidence against it. Nothing here decides whether the agent is
learning; it decides whether the number that says so can be believed.
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path


def read(path: Path) -> tuple[dict, list[dict]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        header = json.loads(handle.readline())
        return header, [json.loads(line) for line in handle]


def load(root: Path, limit: int) -> list[tuple[dict, list[dict]]]:
    """Recent episodes, tolerating the archive being pruned underneath."""
    episodes = []
    for path in sorted(root.rglob("ep*.jsonl.gz"))[-limit:]:
        try:
            episodes.append(read(path))
        except (OSError, EOFError, gzip.BadGzipFile, json.JSONDecodeError):
            # The archive is written and pruned while a run goes, so a file can be half a
            # line long at the moment it is opened. Reading a live archive has to tolerate
            # that, or the audit only works once there is nothing left to audit.
            continue
    return episodes


def report(name: str, ok: bool, detail: str) -> bool:
    print(f"  [{'OK ' if ok else 'ALERTE'}] {name}")
    if detail:
        for line in detail.splitlines():
            print(f"          {line}")
    return ok


#: Terms that are supposed to arrive in one lump, so concentration says nothing about them.
#:
#: A core is lost once and costs fifty, a rung is crossed once and pays up to sixty, a win
#: pays fifty. Measured against the raw reward this check fired at 50.6%, and the top
#: percentile turned out to be 57.5% core losses and 38.5% milestones, which is the design
#: working rather than an exploit. Delivery was 0.6% of it: the opposite of a windfall.
ONE_OFF = frozenset({"core_lost", "milestones", "won", "waves", "kills", "lost"})


def check_concentration(episodes) -> bool:
    """A handful of steps carrying the CONTINUOUS income is what a windfall looks like.

    The stockpile exploit put 22.1% of a run's points into one step per episode, and it
    put them there through delivery. Honest delivery is spread: a supply line pays a
    little on almost every step. So this looks only at the terms meant to trickle, because
    the lumpy ones drown the signal and fire on a run doing exactly what it should.
    """
    flow = [
        sum(abs(v) for k, v in (f.get("terms") or {}).items() if k not in ONE_OFF)
        for _, frames in episodes for f in frames
    ]
    flow = [value for value in flow if value]
    if not flow:
        return report("concentration du revenu continu", True, "aucun revenu continu encore")

    total = sum(flow)
    top = sorted(flow, reverse=True)[: max(1, len(flow) // 100)]
    share = sum(top) / total
    return report(
        "concentration du revenu continu", share < 0.35,
        f"le centieme de steps le mieux paye porte {share:.1%} du revenu continu, "
        f"termes ponctuels exclus",
    )


def check_core_losses(episodes) -> bool:
    """A core can be lost once. More than once means state is leaking between episodes."""
    per = Counter(
        sum(1 for f in frames if (f.get("terms") or {}).get("core_lost"))
        for _, frames in episodes
    )
    worst = max(per, default=0)
    return report(
        "une perte de core par episode au plus", worst <= 1,
        f"distribution : {sorted(per.items())}",
    )


def check_annuity(episodes) -> bool:
    """Income on steps where nothing happened is a level being paid, not a rate.

    The generator term paid 0.05 per unit of generation on every step, so one machine
    bought once printed points for the rest of the episode. The tempting signature is
    repetition, and repetition is the wrong test: that run paid the same value on 43% of
    its steps, while a working conveyor line in the current run pays the same value on
    58% of its steps and is exactly what the agent is supposed to be doing. A check keyed
    on repetition would have missed the bug it was written for and cried wolf on success.

    What separates them is whether anything moved. Delivery raises the core's stock; an
    annuity pays while the world sits still. So this counts steps that paid while neither
    the stock nor the agent did anything.

    Calibrated against both archives, and the margin is thin enough to state rather than
    hide: the farmed run sits at 17.5% and the clean one at 9.2%. Only a fifth of the
    farmed run's steps are quiet enough to qualify, because hand mining moves the stock
    constantly, so the annuity shows through a small window. Treat a reading between the
    two as worth a look, not as a verdict.

    The exact guarantee is not here. It is the unit test that asserts a step in which
    nothing changed earns nothing, which is arithmetic and cannot be fooled by a busy
    episode. This check exists to catch a term that slips past it in a way nobody thought
    to write a test for.
    """
    quiet = paid_quiet = 0
    for _, frames in episodes:
        previous = None
        for frame in frames:
            items = frame.get("items")
            if items is None:
                continue
            if previous is not None and items == previous and not frame.get("act"):
                quiet += 1
                terms = frame.get("terms")
                if terms is None:
                    # Archives recorded before the itemisation existed, which includes
                    # every run this check needs to be calibrated against. The raw reward
                    # is a coarser reading and the right fallback: on a step where nothing
                    # moved and nobody acted, any payment at all is the thing being looked
                    # for.
                    flow = abs(frame.get("reward", 0.0))
                else:
                    flow = sum(abs(v) for k, v in terms.items() if k not in ONE_OFF)
                paid_quiet += flow > 1e-9
            previous = items

    share = paid_quiet / quiet if quiet else 0.0
    return report(
        "pas de revenu quand rien ne bouge", share < 0.14,
        f"{paid_quiet} steps payes sur {quiet} ou ni le stock ni l'agent n'ont bouge "
        f"({share:.1%}). Etalonnage : run farme 17,5%, run propre 9,2%, seuil 14%.",
    )


def check_terms_add_up(episodes) -> bool:
    """The itemisation must equal the reward it itemises.

    They came apart once, when one thread's breakdown was read while another thread's step
    had overwritten it. A reward nobody can attribute is a reward nobody should trust.
    """
    bad = 0
    checked = 0
    for _, frames in episodes:
        for f in frames:
            terms = f.get("terms")
            if terms is None:
                continue
            checked += 1
            if abs(sum(terms.values()) - f.get("reward", 0.0)) > 0.01:
                bad += 1
    return report(
        "le detail egale le total", bad == 0,
        f"{bad} step(s) sur {checked} ou la somme des termes ne fait pas la "
        f"recompense. Un petit nombre fige est attendu dans un run demarre avant "
        f"que le bonus de reussite ne soit itemise ; il doit disparaitre a la "
        f"relance suivante.",
    )


def trend(url: str) -> str:
    """The real progression, taken from the run rather than from the archive.

    The archive cannot answer this and it took a false alarm to notice. It prunes to the
    best five episodes per environment, the most recent three, and every solved one, so
    the older files that survive are precisely the high-scoring ones. Splitting it by age
    therefore compares survivors against a fresh sample and manufactures a collapse every
    single time: it showed 422.8 falling to 16.9, a twenty-five-fold drop, while the
    training losses were perfectly steady and nothing was wrong.

    The dashboard keeps the mean over the last thirty finished episodes for each
    generation, which is the same measure applied to every generation alike.
    """
    try:
        import urllib.request
        with urllib.request.urlopen(url, timeout=5) as answer:
            generations = json.loads(answer.read())["generations"]
    except Exception as error:
        return (f"  tendance indisponible ({type(error).__name__}). L'archive ne "
                f"peut pas la remplacer : elle garde les meilleurs episodes, donc "
                f"toute lecture par anciennete y montre une chute.")

    points = [g for g in generations if g.get("mean_reward") is not None]
    if len(points) < 12:
        return "  pas encore assez de generations pour une tendance"

    step = len(points) // 6
    lines_out = []
    for index in range(6):
        part = points[index * step:(index + 1) * step]
        if not part:
            continue
        mean = sum(g["mean_reward"] for g in part) / len(part)
        wave = max(g.get("best_wave") or 0 for g in part)
        lines_out.append(
            f"  updates {part[0]['update']:5d}-{part[-1]['update']:5d} : "
            f"moyenne {mean:8.1f}   meilleure vague {wave:2d}"
        )
    return "\n".join(lines_out)


def describe_archive(episodes) -> str:
    """What the archive holds right now. A snapshot, deliberately not a trend."""
    delivered = [
        sum((f.get("terms") or {}).get("delivered", 0.0) for f in frames)
        for _, frames in episodes
    ]
    died = sum(
        1 for _, frames in episodes
        if any((f.get("terms") or {}).get("core_lost") for f in frames)
    )
    lengths = sorted(len(frames) for _, frames in episodes)
    return (
        f"  {len(episodes)} episodes conserves   "
        f"livraison moyenne {sum(delivered) / len(episodes):7.1f}   "
        f"core perdu {100 * died / len(episodes):3.0f}%   "
        f"duree mediane {lengths[len(lengths) // 2]:5d}\n"
        f"  (instantane biaise vers le haut : l'archive garde les meilleurs)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("replays/live"))
    parser.add_argument("--dashboard", default="http://127.0.0.1:8800/state",
                        help="ou lire la vraie progression, l'archive en etant incapable")
    parser.add_argument("--limit", type=int, default=400,
                        help="how many recent episodes to read")
    args = parser.parse_args()

    episodes = load(args.root, args.limit)
    if not episodes:
        raise SystemExit(f"no readable episode under {args.root}")

    print(f"{len(episodes)} episodes lus sous {args.root}\n")
    print("failles connues et leurs cousines :")
    passed = all([
        check_concentration(episodes),
        check_core_losses(episodes),
        check_annuity(episodes),
        check_terms_add_up(episodes),
    ])

    print("\nprogression reelle, mesuree sur le run :")
    print(trend(args.dashboard))

    print("\nce que l'archive contient en ce moment :")
    print(describe_archive(episodes))

    print()
    print("AUDIT : rien de suspect" if passed
          else "AUDIT : au moins une alerte, le score n'est pas a prendre au mot")


if __name__ == "__main__":
    main()
