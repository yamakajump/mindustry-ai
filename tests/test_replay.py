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


def plane(encoded: str, dtype=np.uint8) -> np.ndarray:
    return np.frombuffer(zlib.decompress(base64.b64decode(encoded)), dtype=dtype)


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
    """The header carries the typed map, which is what lets a viewer pick real sprites."""
    header, _ = read(recorded)
    assert header["type"] == "header"
    assert header["format"] == REPLAY_FORMAT
    assert header["width"] > 0 and header["height"] > 0

    tiles = header["width"] * header["height"]
    assert plane(header["floor"], np.uint16).size == tiles
    assert plane(header["overlay"], np.uint16).size == tiles
    assert plane(header["block"], np.uint16).size == tiles
    assert plane(header["rotation"]).size == tiles

    names = {entry["name"] for entry in header["palette"].values()}
    assert "ore-copper" in names, "the palette should name the map's ores"
    assert header["blocks"], "the blocks the agent can place must be listed for sprites"


def test_actions_are_recorded_with_their_block(recorded: Path) -> None:
    """Tile deltas say a tile filled up; only the action says with what."""
    _, records = read(recorded)
    acts = [r["act"] for r in records if r.get("type") == "frame" and "act" in r]
    assert acts, "no actions recorded"
    placed = [a for a in acts if a["t"] == "place"]
    assert placed, "nothing was built"
    assert all(isinstance(a["b"], str) for a in placed), "block identity missing"


def test_frames_carry_the_timeline(recorded: Path) -> None:
    _, records = read(recorded)
    frames = [r for r in records if r["type"] == "frame"]
    assert len(frames) > 1
    assert [f["step"] for f in frames] == sorted(f["step"] for f in frames)
    assert frames[-1]["tick"] > frames[0]["tick"]


def test_what_was_built_is_recoverable_from_the_actions(recorded: Path) -> None:
    """Storing the full tensor per step would cost about 413 MB an episode.

    It used to be stored as a tile diff of the observation's ally-block channel. That
    channel became a forty-eight tile window around a moving agent when the bridge started
    cropping, so the map slid under the diff and every step reported the whole window as
    built and unbuilt at once: 67,896 tiles appearing and 67,766 disappearing per episode,
    for 3,191 blocks actually asked for.

    Nothing read it. What stands is derived by replaying the recorded actions, which is
    what the viewer always did and which no coordinate system can invalidate.
    """
    _, records = read(recorded)
    frames = [r for r in records if r["type"] == "frame"]

    assert not any("added" in f or "removed" in f for f in frames), (
        "tile diffs are back, and they cannot be read once the observation is a window"
    )
    assert any(f.get("act", {}).get("t") in ("place", "stamp") for f in frames), (
        "no construction was recorded, so nothing can be reconstructed from it"
    )


def test_replay_stays_small(recorded: Path) -> None:
    """Small enough to commit and to serve from a static page."""
    assert recorded.stat().st_size < 200_000, "a 40 step replay should be a few KB"


def test_replaying_actions_reconstructs_the_final_state(recorded: Path) -> None:
    """The viewer rebuilds each frame by applying recorded actions in order."""
    header, records = read(recorded)
    frames = [r for r in records if r["type"] == "frame"]

    present: dict[tuple[int, int], str] = {}
    for frame in frames:
        act = frame.get("act")
        if not act:
            continue
        if act["t"] == "place":
            present[(act["x"], act["y"])] = act["b"]
        else:
            present.pop((act["x"], act["y"]), None)

    assert present, "no blocks survived to the end"
    for x, y in present:
        assert 0 <= x < header["width"] and 0 <= y < header["height"], "block outside the map"


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
