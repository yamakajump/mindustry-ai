"""The live view keeps an honest picture of a match, and only keeps what is worth keeping.

These run without a server: the point is the bookkeeping between the bridge and the
browser, which is where a dashboard silently starts lying about what the agent is doing.
"""

from __future__ import annotations

from pathlib import Path

from gamma.archive import ReplayArchive
from gamma.monitor import SceneBuffer, TrainingMonitor


def frame(**fields) -> dict:
    """A bridge frame with everything defaulted, so a test states only what it is about."""
    return {"playing": True, "units": [], "placed": [], "hurt": [], "removed": [],
            "shots": [], "tick": 0, "wave": 0, **fields}


# Scene buffer ---------------------------------------------------------------------

def test_a_new_viewer_receives_the_whole_world() -> None:
    buffer = SceneBuffer()
    buffer.apply(frame(placed=[10, 5, 0, 1, 100, 100]))

    state = buffer.since(0)
    assert state["full"] is True
    assert state["placed"] == [10, 5, 0, 1, 100, 100]


def test_a_viewer_up_to_date_receives_nothing() -> None:
    """The common case by far: polling four times a second, most polls have no news."""
    buffer = SceneBuffer()
    buffer.apply(frame(placed=[10, 5, 0, 1, 100, 100]))

    state = buffer.since(buffer.version)
    assert state["full"] is False
    assert state["placed"] == []
    assert state["removed"] == []


def test_changes_between_two_polls_are_not_lost() -> None:
    """The bridge frames are deltas against itself, so several may pass between polls.

    A viewer that only ever saw the frame it happened to land on would miss every
    building placed in between, and would keep drawing them forever after.
    """
    buffer = SceneBuffer()
    buffer.apply(frame(placed=[10, 5, 0, 1, 100, 100]))
    seen = buffer.version

    buffer.apply(frame(placed=[11, 6, 0, 1, 100, 100]))
    buffer.apply(frame(placed=[12, 7, 0, 1, 100, 100]))

    state = buffer.since(seen)
    assert sorted(state["placed"][::6]) == [11, 12]


def test_damage_updates_a_building_without_replacing_it() -> None:
    buffer = SceneBuffer()
    buffer.apply(frame(placed=[10, 5, 0, 1, 100, 100]))
    buffer.apply(frame(hurt=[10, 42]))

    state = buffer.since(0)
    assert state["placed"] == [10, 5, 0, 1, 42, 100]


def test_a_destroyed_building_is_reported_then_forgotten() -> None:
    buffer = SceneBuffer()
    buffer.apply(frame(placed=[10, 5, 0, 1, 100, 100]))
    seen = buffer.version
    buffer.apply(frame(removed=[10]))

    assert buffer.since(seen)["removed"] == [10]
    assert buffer.since(0)["placed"] == []


def test_a_viewer_too_far_behind_is_resynchronised() -> None:
    """Rather than patched with removals that are no longer retained."""
    buffer = SceneBuffer()
    buffer.apply(frame(placed=[10, 5, 0, 1, 100, 100]))

    assert buffer.since(9999)["full"] is True


def test_a_new_episode_clears_the_previous_base() -> None:
    """Otherwise the first frames of a new match still show the old one."""
    buffer = SceneBuffer()
    buffer.apply(frame(placed=[10, 5, 0, 1, 100, 100]))
    buffer.clear()

    state = buffer.since(0)
    assert state["placed"] == []
    assert state["full"] is True


def test_block_descriptions_accumulate_across_frames() -> None:
    """A viewer joining late still needs the sprite for a block placed an hour ago."""
    buffer = SceneBuffer()
    buffer.apply(frame(placed=[10, 5, 0, 1, 100, 100], blocks={"5": {"name": "duo", "size": 1}}))
    buffer.apply(frame(placed=[11, 6, 0, 1, 100, 100], blocks={"6": {"name": "router", "size": 1}}))

    assert set(buffer.since(0)["blocks"]) == {"5", "6"}


def test_a_frame_from_a_stopped_game_is_ignored() -> None:
    buffer = SceneBuffer()
    buffer.apply({"playing": False, "seq": 3})

    assert buffer.version == 0


# Replay archive -------------------------------------------------------------------

def write(archive: ReplayArchive, number: int) -> None:
    archive.pending(number).write_bytes(b"recorded")


def test_a_committed_episode_is_named_by_its_reward(tmp_path: Path) -> None:
    archive = ReplayArchive(tmp_path)
    write(archive, 0)

    episode = archive.commit(0, 12.5, solved=False)
    assert episode is not None
    assert episode.path.exists()
    assert "pos001250" in episode.path.name


def test_a_negative_reward_is_spelled_out_rather_than_signed(tmp_path: Path) -> None:
    """A minus in a filename is an option to every shell that ever reads it."""
    archive = ReplayArchive(tmp_path)
    write(archive, 0)

    episode = archive.commit(0, -3.0, solved=False)
    assert episode is not None
    assert episode.path.name == "ep000000-neg000300.jsonl.gz"
    assert archive.resolve(episode.path.name) == episode.path


def test_only_the_best_and_the_latest_survive(tmp_path: Path) -> None:
    archive = ReplayArchive(tmp_path, keep_best=2, keep_recent=1)
    for number, reward in enumerate([5.0, 1.0, 4.0, 0.5, 0.2]):
        write(archive, number)
        archive.commit(number, reward, solved=False)

    kept = archive.listing()
    assert [row["reward"] for row in kept] == [5.0, 4.0, 0.2]
    assert len(list(tmp_path.glob("*.jsonl.gz"))) == 3


def test_a_solved_episode_is_kept_whatever_its_reward(tmp_path: Path) -> None:
    """On a hard task there may only ever be one, and it is the whole point of the run."""
    archive = ReplayArchive(tmp_path, keep_best=1, keep_recent=1)
    write(archive, 0)
    archive.commit(0, -1.0, solved=True)
    for number in range(1, 5):
        write(archive, number)
        archive.commit(number, 9.0, solved=False)

    assert any(row["solved"] for row in archive.listing())


def test_a_name_from_outside_the_archive_resolves_to_nothing(tmp_path: Path) -> None:
    """The name arrives in a URL, so it is checked rather than joined to a directory."""
    archive = ReplayArchive(tmp_path)
    write(archive, 0)
    archive.commit(0, 1.0, solved=False)

    assert archive.resolve("../../secrets.env") is None
    assert archive.resolve("ep000000-pos000100.jsonl.gz") is not None


def test_discarding_a_recording_still_open_does_not_raise(tmp_path: Path) -> None:
    """This runs while a worker unwinds from an error, and Windows will not unlink an open
    file. Raising here once took the rest of the cleanup with it and leaked a server."""
    archive = ReplayArchive(tmp_path)
    handle = archive.pending(0).open("wb")
    try:
        archive.discard(0)
    finally:
        handle.close()


def test_a_previous_run_leaves_no_replays_behind(tmp_path: Path) -> None:
    """Its best episode belongs to another policy and would top this run's board."""
    (tmp_path / "ep000001-pos009999.jsonl.gz").write_bytes(b"old")

    ReplayArchive(tmp_path)
    assert list(tmp_path.glob("*.jsonl.gz")) == []


# Generations ----------------------------------------------------------------------

def test_a_generation_can_be_told_which_file_it_produced() -> None:
    """The weights are written after the generation is recorded, so it is annotated."""
    monitor = TrainingMonitor()
    monitor.record_generation(1, {"entropy": 2.0, "value_loss": 0.1})
    monitor.annotate_generation(1, checkpoint="beta-best.pt", best=True)

    snapshot = monitor.snapshot()
    assert snapshot["generations"][0]["checkpoint"] == "beta-best.pt"
    assert snapshot["totals"]["best_generation"] == 1


def test_a_match_carries_its_milestones_to_the_viewer() -> None:
    """The rungs are the part of a match worth looking at. A card showing a reward of 3.2
    says nothing; a card showing that a machine delivered ore says the agent started
    playing."""
    from gamma.monitor import MatchState

    state = MatchState(index=0)
    state.produced = 412
    state.reached = ["first_drill", "automation"]

    payload = state.as_dict()
    assert payload["produced"] == 412
    assert payload["reached"] == ["first_drill", "automation"]


def test_the_dashboard_names_every_rung_in_both_languages() -> None:
    """A rung with no translation renders its own slug on screen, which is how a viewer
    ends up showing `rung_automation_1k` to a person."""
    from gamma import tasks

    page = (Path(__file__).resolve().parents[1] / "viewer" / "dashboard.html").read_text(
        encoding="utf-8"
    )
    for stone in tasks.MILESTONES:
        assert page.count(f"rung_{stone.name}:") == 2, stone.name


def test_throughput_counts_steps_taken_not_steps_into_an_episode() -> None:
    """The figure on screen used to be the average distance into an episode, which fell
    back to zero every time one ended. It read 41 steps/s for a run doing 283."""
    from gamma.monitor import TrainingMonitor

    monitor = TrainingMonitor()
    first, second = monitor.match(0), monitor.match(1)
    first.total_steps, second.total_steps = 4000, 6000
    # Both have just reset, so their position inside the current episode is nearly zero.
    first.step, second.step = 3, 1

    assert monitor.snapshot()["totals"]["steps_per_second"] > 0
    assert first.as_dict()["total_steps"] == 4000


def test_the_best_recording_is_found_across_every_match(tmp_path) -> None:
    """A run leaves one archive per match. Asking a person to open twenty-five folders to
    find the good episode is how the good episode never gets watched."""
    from tools.watch import best_replays, scored

    for match, names in {
        "match3": ["ep000000-neg003803.jsonl.gz", "ep000004-pos039213.jsonl.gz"],
        "match7": ["ep000002-pos000150.jsonl.gz", "ep000009.pending.jsonl.gz"],
    }.items():
        (tmp_path / match).mkdir()
        for name in names:
            (tmp_path / match / name).write_bytes(b"")

    ranked = best_replays(tmp_path)
    assert [round(score, 2) for score, _ in ranked] == [392.13, 1.50, -38.03]
    assert ranked[0][1].name == "ep000004-pos039213.jsonl.gz"
    # A recording still being written has no score and must not be offered.
    assert scored(Path("ep000009.pending.jsonl.gz")) is None
