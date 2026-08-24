"""Splitting the planet into what an agent trains on and what it is judged on.

The split is the only thing that makes a generalisation claim checkable. Without it, a
number that goes up says nothing: memorising the training maps produces exactly the same
curve as learning the game.
"""

from __future__ import annotations

import random

import pytest

from gamma.sectors import EVAL_SHARE, build_pool


def listing(count: int = 200, threat: float = 0.35) -> dict:
    return {"sectors": list(range(count)), "threats": [threat] * count}


def test_nothing_is_in_both_halves() -> None:
    """A sector trained on and then evaluated on measures memory, not generalisation."""
    pool = build_pool(listing())

    assert not set(pool.train) & set(pool.evaluate)
    assert len(pool) == 200


def test_the_held_out_half_is_about_a_fifth() -> None:
    pool = build_pool(listing())

    assert len(pool.evaluate) == pytest.approx(200 / EVAL_SHARE, abs=1)


def test_the_split_is_not_a_band_of_the_planet() -> None:
    """Sector indices follow the planet's grid, so consecutive ones are neighbours and
    share a biome. Taking every fifth index would hand the evaluation set one region and
    turn "a world it has not seen" into "a kind of world it has not seen"."""
    pool = build_pool(listing(200))
    held = sorted(pool.evaluate)

    gaps = {held[i + 1] - held[i] for i in range(len(held) - 1)}
    assert len(gaps) > 1, "the held-out sectors are evenly spaced, which is a band"


def test_the_split_is_the_same_every_run() -> None:
    """Otherwise a checkpoint is evaluated on maps a later run trained on."""
    assert build_pool(listing()).evaluate == build_pool(listing()).evaluate


def test_threat_narrows_the_pool() -> None:
    mixed = {"sectors": [0, 1, 2, 3], "threats": [0.3, 0.9, 0.35, 0.8]}
    pool = build_pool(mixed, threat_limit=0.4)

    assert len(pool) == 2
    assert pool.hardest() <= 0.4


def test_a_threat_nothing_meets_says_the_range() -> None:
    """The mistake is a number picked without looking at the planet, and the message is
    the fastest way to find that out."""
    with pytest.raises(ValueError, match="0.30"):
        build_pool({"sectors": [0, 1], "threats": [0.30, 0.74]}, threat_limit=0.1)


def test_episodes_draw_from_the_half_they_were_asked_for() -> None:
    pool = build_pool(listing())
    rng = random.Random(1)

    trained = {pool.pick(rng) for _ in range(200)}
    held = {pool.pick(rng, evaluating=True) for _ in range(200)}

    assert trained <= set(pool.train)
    assert held <= set(pool.evaluate)
