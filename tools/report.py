"""Answer questions about a run from what it wrote down, whether or not it is still going.

    python tools/report.py                     # the whole history
    python tools/report.py --last 500          # the last five hundred episodes
    python tools/report.py --since 1400        # from generation 1400 on
    python tools/report.py --split             # early half against late half

This exists because of a question that could not be answered. Asked one morning how the
overnight run had gone, across ten restarts and 4.7 million steps, nothing could say: the
dashboard kept four hundred points in memory and lost them on exit, the replay archive
prunes to the best episodes so reading it by age manufactures a collapse, and checkpoints
hold weights, which do not say where they came from.

Every number here is computed at reading time from raw per-episode counts, so a question
nobody thought to ask in advance can still be asked afterwards.
"""

from __future__ import annotations

import argparse
import statistics
from collections import Counter
from pathlib import Path

from gamma.journal import Journal


def summarise(episodes: list[dict], title: str) -> str:
    """What a set of episodes amounts to, stated so two sets can be compared."""
    if not episodes:
        return f"{title} : aucun episode"

    rewards = [float(e.get("reward", 0.0)) for e in episodes]
    waves = [int(e.get("wave", 0)) for e in episodes]
    lost = sum(1 for e in episodes if e.get("core_lost"))
    steps = [int(e.get("steps", 0)) for e in episodes]

    acted = Counter()
    refused = Counter()
    reasons = Counter()
    blocks = Counter()
    terms = Counter()
    broken = 0
    spread = []
    for episode in episodes:
        acted.update(episode.get("actions") or {})
        refused.update(episode.get("refused") or {})
        reasons.update(episode.get("reasons") or {})
        blocks.update(episode.get("blocks") or {})
        terms.update({k: float(v) for k, v in (episode.get("terms") or {}).items()})
        broken += int(episode.get("self_broken", 0))
        if episode.get("turret_distance") is not None:
            spread.append(float(episode["turret_distance"]))

    n = len(episodes)
    tried = sum(acted.values()) + sum(refused.values())
    # The standard error, because the whole point of a comparison is knowing whether the
    # difference is bigger than the noise, and a mean on its own never says.
    error = statistics.pstdev(rewards) / (n ** 0.5) if n > 1 else 0.0

    out = [
        f"{title} : {n:,} episodes",
        f"  score        {statistics.mean(rewards):9.1f} +/- {error:.1f}   "
        f"mediane {statistics.median(rewards):8.1f}   "
        f"pire {min(rewards):7.1f}   meilleur {max(rewards):8.1f}",
        f"  vague        moyenne {statistics.mean(waves):5.2f}   "
        f"mediane {statistics.median(waves):4.0f}   max {max(waves)}",
        f"  core perdu   {100 * lost / n:5.1f}%",
        f"  duree        mediane {statistics.median(steps):6.0f} pas",
        f"  refus        {100 * sum(refused.values()) / max(1, tried):5.1f}%",
        f"  gachis       {broken / n:6.1f} demolitions de ses propres blocs par episode",
    ]
    if spread:
        out.append(f"  defense      tourelles a {statistics.mean(spread):5.1f} tuiles du core "
                   f"en moyenne (mediane {statistics.median(spread):.1f})")

    out.append("  actions      " + "  ".join(
        f"{k}:{100 * v / max(1, sum(acted.values())):.0f}%" for k, v in acted.most_common(6)))
    if refused:
        out.append("  refuses      " + "  ".join(
            f"{k}:{v / n:.0f}" for k, v in refused.most_common(5)))
    if reasons:
        top = [(k, v) for k, v in reasons.most_common(4) if v]
        out.append("  raisons      " + "  ".join(f"{k}:{v:,}" for k, v in top))
    if blocks:
        out.append("  blocs        " + "  ".join(
            f"{k}:{v / n:.0f}" for k, v in blocks.most_common(6)))
    if terms:
        total = sum(abs(v) for v in terms.values())
        out.append("  points       " + "  ".join(
            f"{k}:{100 * abs(v) / max(1e-9, total):.0f}%"
            for k, v in sorted(terms.items(), key=lambda kv: -abs(kv[1]))[:6]))
    return "\n".join(out)


def curve(episodes: list[dict], buckets: int = 10) -> str:
    """The progression, in equal slices of episodes rather than of time.

    Sliced by count, not by generation number, because a run that restarts resumes its
    generation counter and the episodes between two restarts are not evenly spread.
    """
    if len(episodes) < buckets * 2:
        return "  pas encore assez d'episodes pour une courbe"

    width = len(episodes) // buckets
    lines = []
    for index in range(buckets):
        part = episodes[index * width:(index + 1) * width]
        if not part:
            continue
        rewards = [float(e.get("reward", 0.0)) for e in part]
        mean = statistics.mean(rewards)
        error = statistics.pstdev(rewards) / (len(part) ** 0.5) if len(part) > 1 else 0.0
        lost = 100 * sum(1 for e in part if e.get("core_lost")) / len(part)
        wave = statistics.mean(int(e.get("wave", 0)) for e in part)
        lines.append(
            f"  gen {part[0].get('generation', 0):5d}-{part[-1].get('generation', 0):5d}  "
            f"score {mean:8.1f} +/- {error:5.1f}   vague {wave:4.1f}   core perdu {lost:5.1f}%")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journal", type=Path, default=Path("checkpoints/journal.jsonl"))
    parser.add_argument("--last", type=int, default=0, help="only the last N episodes")
    parser.add_argument("--since", type=int, default=0, help="only from generation N on")
    parser.add_argument("--split", action="store_true",
                        help="compare the first half against the second")
    args = parser.parse_args()

    episodes = Journal(args.journal).read()
    if not episodes:
        raise SystemExit(
            f"{args.journal} est vide ou absent. Il se remplit a partir du prochain "
            f"entrainement : une ligne par episode, jamais elaguee.")

    if args.since:
        episodes = [e for e in episodes if int(e.get("generation", 0)) >= args.since]
    if args.last:
        episodes = episodes[-args.last:]

    print(f"{len(episodes):,} episodes dans {args.journal}\n")
    print(summarise(episodes, "ensemble"))

    print("\nprogression :")
    print(curve(episodes))

    if args.split and len(episodes) >= 4:
        middle = len(episodes) // 2
        print()
        print(summarise(episodes[:middle], "premiere moitie"))
        print()
        print(summarise(episodes[middle:], "seconde moitie"))

        early = [float(e.get("reward", 0.0)) for e in episodes[:middle]]
        late = [float(e.get("reward", 0.0)) for e in episodes[middle:]]
        gap = statistics.mean(late) - statistics.mean(early)
        noise = ((statistics.pvariance(early) / len(early))
                 + (statistics.pvariance(late) / len(late))) ** 0.5
        verdict = ("un progres reel" if gap > 2 * noise else
                   "une regression reelle" if gap < -2 * noise else
                   "indiscernable du bruit")
        print(f"\n  ecart {gap:+.1f} pour un bruit de {noise:.1f} : {verdict}")


if __name__ == "__main__":
    main()
