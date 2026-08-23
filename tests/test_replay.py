"""Replays record what happened, small enough to publish and complete enough to watch."""

from __future__ import annotations

import base64
import gzip
import json
import zlib
from pathlib import Path

import numpy as np
import pytest

from gamma import tasks
from gamma.alpha import AlphaPolicy
from gamma.replay import REPLAY_FORMAT, record_episode

SHOWCASE = Path(__file__).resolve().parent.parent / "replays" / "showcase"


def read(path: Path) -> tuple[dict, list[dict]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle]
    return records[0], records[1:]


def plane(encoded: str) -> np.ndarray:
    return np.frombuffer(zlib.decompress(base64.b64decode(encoded)), dtype=np.uint8)


@pytest.fixture(scope="module")
def recorded(env, tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("replays") / "alpha.jsonl.gz"
    policy = AlphaPolicy(env)
    from dataclasses import replace

    env.task = replace(tasks.T1_COPPER, max_steps=40)
    try:
        record_episode(env, policy, path, note="test run")
    finally:
        env.task = tasks.T1_COPPER
    return path


def test_header_describes_the_map(recorded: Path) -> None:
    header, _ = read(recorded)
    assert header["type"] == "header"
    assert header["format"] == REPLAY_FORMAT
    assert header["width"] > 0 and header["height"] > 0
    assert plane(header["solid"]).size == header["width"] * header["height"]
    assert plane(header["ores"]).size == header["width"] * header["height"]
    assert "copper" in header["ore_names"]


def test_frames_carry_the_timeline(recorded: Path) -> None:
    _, records = read(recorded)
    frames = [r for r in records if r["type"] == "frame"]
    assert len(frames) > 1
    assert [f["step"] for f in frames] == sorted(f["step"] for f in frames)
    assert frames[-1]["tick"] > frames[0]["tick"]


def test_blocks_are_stored_as_deltas(recorded: Path) -> None:
    """Storing the full tensor per step would cost about 413 MB an episode."""
    _, records = read(recorded)
    frames = [r for r in records if r["type"] == "frame"]
    assert any("added" in f for f in frames), "no construction was recorded"
    # No frame should carry the whole map; deltas are small by construction.
    assert max(len(f.get("added", [])) for f in frames) < 100


def test_replay_stays_small(recorded: Path) -> None:
    """Small enough to commit and to serve from a static page."""
    assert recorded.stat().st_size < 200_000, "a 40 step replay should be a few KB"


def test_replaying_deltas_reconstructs_the_final_state(recorded: Path) -> None:
    """The viewer rebuilds each frame by applying deltas in order, so that has to work."""
    header, records = read(recorded)
    frames = [r for r in records if r["type"] == "frame"]

    present: set[tuple[int, int]] = set()
    for frame in frames:
        for x, y in frame.get("removed", []):
            present.discard((x, y))
        for x, y in frame.get("added", []):
            present.add((x, y))

    assert present, "no blocks survived to the end"
    solid = plane(header["solid"]).reshape(header["height"], header["width"])
    for x, y in present:
        assert 0 <= x < header["width"] and 0 <= y < header["height"], "block outside the map"
        assert solid[y, x] == 0, "a block was recorded inside a wall"


def test_episode_end_is_recorded(recorded: Path) -> None:
    _, records = read(recorded)
    assert records[-1]["type"] == "end"
    assert "solved" in records[-1]


def test_showcase_replays_are_valid() -> None:
    """The replays committed for the README must stay loadable."""
    files = sorted(SHOWCASE.glob("*.jsonl.gz"))
    assert files, "no showcase replays committed"
    for path in files:
        header, records = read(path)
        assert header["format"] == REPLAY_FORMAT, f"{path.name} uses an old format"
        assert any(r["type"] == "frame" for r in records), f"{path.name} has no frames"
