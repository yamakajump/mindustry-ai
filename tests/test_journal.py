"""The record that outlives the run.

Asked one morning how the overnight run had gone, across ten restarts and 4.7 million
steps, nothing could answer. The dashboard kept four hundred points in memory and lost
them on exit, the archive prunes to the best episodes so reading it by age manufactures a
collapse, and checkpoints hold weights, which do not say where they came from.
"""

from __future__ import annotations

import json
import threading

from gamma.journal import EpisodeTally, Journal


def test_a_line_survives_the_process(tmp_path):
    journal = Journal(tmp_path / "journal.jsonl")
    journal.append({"generation": 1, "reward": 12.0})
    journal.append({"generation": 2, "reward": 15.0})

    assert [e["generation"] for e in Journal(tmp_path / "journal.jsonl").read()] == [1, 2]


def test_a_torn_line_does_not_cost_the_rest(tmp_path):
    """A run killed to apply a fix is the normal way this ends."""
    path = tmp_path / "journal.jsonl"
    path.write_text('{"generation": 1}\n{"generation": 2\n{"generation": 3}\n',
                    encoding="utf-8")

    assert [e["generation"] for e in Journal(path).read()] == [1, 3]


def test_twenty_four_threads_do_not_interleave_their_lines(tmp_path):
    """One torn line would corrupt every reading that follows it, not just its own."""
    journal = Journal(tmp_path / "journal.jsonl")
    threads = [threading.Thread(target=journal.append,
                                args=({"generation": i, "blocks": {"conveyor": i}},))
               for i in range(24)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    lines = (tmp_path / "journal.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 24
    assert sorted(json.loads(line)["generation"] for line in lines) == list(range(24))


def test_the_tally_counts_what_a_diagnosis_needs():
    """Each field earned its place by a diagnosis that could not be made without it."""
    tally = EpisodeTally()
    core = (10, 10)

    tally.note({"reward": 1.0, "wave": 2, "terms": {"delivered": 3.0},
                "act": {"t": "connect", "cells": [[4, 4, "mechanical-drill", 0],
                                                  [5, 4, "conveyor", 0]]}}, core)
    tally.note({"reward": 0.0, "wave": 2, "refused": 1,
                "act": {"t": "connect", "reason": "no route"}}, core)
    tally.note({"reward": 0.5, "wave": 3,
                "act": {"t": "place", "b": "duo", "x": 14, "y": 13}}, core)
    tally.note({"reward": -2.0, "wave": 3, "terms": {"churn": -2.0},
                "act": {"t": "break", "x": 5, "y": 4}}, core)

    out = tally.summary(generation=7)
    assert out["generation"] == 7
    assert out["steps"] == 4 and out["wave"] == 3
    assert out["actions"] == {"connect": 1, "place": 1, "break": 1}
    assert out["refused"] == {"connect": 1}
    assert out["reasons"] == {"connect:no route": 1}
    assert out["blocks"] == {"mechanical-drill": 1, "conveyor": 1, "duo": 1}
    assert out["self_broken"] == 1, "it tore down a conveyor it had just laid"
    assert out["turrets"] == 1 and out["turret_distance"] == 7.0
    assert out["terms"]["delivered"] == 3.0 and out["terms"]["churn"] == -2.0


def test_a_demolition_of_something_it_never_built_is_not_churn():
    """Otherwise clearing a natural wall reads as undoing its own work."""
    tally = EpisodeTally()
    tally.note({"act": {"t": "break", "x": 3, "y": 3}})
    assert tally.summary()["self_broken"] == 0


def test_the_core_being_lost_is_recorded_even_once():
    tally = EpisodeTally()
    tally.note({"terms": {"core_lost": -50.0}, "act": {"t": "noop"}})
    assert tally.summary()["core_lost"] is True
