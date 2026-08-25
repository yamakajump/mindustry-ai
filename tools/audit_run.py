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
    """The same non-zero reward, thousands of steps running, is a level being paid.

    The generator term paid 0.05 per unit of generation on every step, so a machine bought
    once printed points for the rest of the episode. Its fingerprint was 1,302 steps paying
    exactly 0.55 and 583 paying exactly 0.45.
    """
    worst = (0.0, 0, 0)
    for _, frames in episodes:
        counts = Counter(round(f["reward"], 4) for f in frames if f.get("reward"))
        counts.pop(0.0, None)
        if not counts:
            continue
        value, repeats = counts.most_common(1)[0]
        if repeats > worst[1]:
            worst = (value, repeats, len(frames))

    value, repeats, length = worst
    share = repeats / length if length else 0.0
    return report(
        "pas de rente par step", share < 0.5,
        f"pire episode : la valeur {value} revient {repeats} fois sur {length} steps "
        f"({share:.0%})\n(une livraison honnete varie, un niveau paye se repete)",
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
        f"{bad} step(s) sur {checked} ou la somme des termes ne fait pas la recompense
"
        f"(un petit nombre fige est attendu dans un run demarre avant que le bonus de
"
        f"reussite ne soit itemise ; il doit disparaitre a la relance suivante)",
    )


def check_progress(episodes) -> str:
    """Not a pass or a fail: the two numbers the score is supposed to stand for.

    A score can rise for reasons that have nothing to do with playing better, which is how
    all three exploits presented. These are measured on the artefact instead: ore that
    actually arrived, and whether the base was still standing at the end.
    """
    half = len(episodes) // 2 or 1
    lines = []
    for label, part in (("premiere moitie", episodes[:half]), ("seconde moitie", episodes[half:])):
        if not part:
            continue
        delivered = [
            sum((f.get("terms") or {}).get("delivered", 0.0) for f in frames)
            for _, frames in part
        ]
        died = sum(
            1 for _, frames in part
            if any((f.get("terms") or {}).get("core_lost") for f in frames)
        )
        score = [sum(f.get("reward", 0.0) for f in frames) for _, frames in part]
        lengths = sorted(len(frames) for _, frames in part)
        lines.append(
            f"  {label:<16} score {sum(score) / len(part):8.1f}   "
            f"livraison {sum(delivered) / len(part):7.1f}   "
            f"core perdu {100 * died / len(part):3.0f}%   "
            f"duree mediane {lengths[len(lengths) // 2]:5d}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("replays/live"))
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

    print("\nce que le score est cense representer :")
    print(check_progress(episodes))

    print()
    print("AUDIT : rien de suspect" if passed
          else "AUDIT : au moins une alerte, le score n'est pas a prendre au mot")


if __name__ == "__main__":
    main()
