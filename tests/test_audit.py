"""The audit has to keep working when the thing it reads changes shape.

Its checks are the only thing standing between a rising score and a farmed one, and a
check that has quietly stopped measuring prints exactly what a clean run prints.
"""

from __future__ import annotations

from tools import audit_run

def test_the_annuity_check_still_finds_quiet_steps_when_passes_are_recorded():
    """A check with an empty denominator reports green and measures nothing.

    Once `noop` started being recorded like any other action, every frame carried one, the
    annuity check found no quiet step at all, and it printed "0 payes sur 0" in green. That
    is indistinguishable from a run with nothing to hide.
    """
    frames = [
        {"items": {"copper": 5}, "act": {"t": "noop"}, "terms": {"generating": 3.0}}
        for _ in range(50)
    ]
    frames.insert(0, {"items": {"copper": 5}, "act": {"t": "noop"}, "terms": {}})

    assert not audit_run.check_annuity([({}, frames)]), (
        "fifty paid steps where nothing moved is exactly the exploit this looks for")
