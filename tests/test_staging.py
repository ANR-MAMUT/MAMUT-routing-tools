"""``StagedTree``: staged files replace the live ones all at once, or not at all."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from mamut_routing_tools.staging import (
    COMMITTING_MARKER,
    STAGING_DIRNAME,
    StagedTree,
    recover_staging,
)


def _live(root: Path) -> None:
    (root / "a").mkdir(parents=True)
    (root / "a" / "x.json").write_text("old x", encoding="utf-8")
    (root / "b.json").write_text("old b", encoding="utf-8")


def test_commit_replaces_and_removes(tmp_path: Path) -> None:
    _live(tmp_path)
    linked_before = tmp_path / "linked-copy"
    os.link(tmp_path / "a" / "x.json", linked_before)
    with StagedTree(tmp_path, "t") as staged:
        staged.stage(tmp_path / "a" / "x.json").write_text("new x", encoding="utf-8")
        staged.stage("c/new.json").write_text("new c", encoding="utf-8")
        staged.remove_on_commit("b.json")
        assert staged.link_live("b.json").read_text(encoding="utf-8") == "old b"
        assert (tmp_path / "a" / "x.json").read_text(encoding="utf-8") == "old x"
        staged.commit()
    assert (tmp_path / "a" / "x.json").read_text(encoding="utf-8") == "new x"
    assert (tmp_path / "c" / "new.json").read_text(encoding="utf-8") == "new c"
    assert not (tmp_path / "b.json").exists()
    # A hard link to the old file keeps the old bytes: replaced, not truncated.
    assert linked_before.read_text(encoding="utf-8") == "old x"
    assert not (tmp_path / STAGING_DIRNAME).exists()


def test_an_exception_discards_everything(tmp_path: Path) -> None:
    _live(tmp_path)
    with pytest.raises(RuntimeError), StagedTree(tmp_path, "t") as staged:
        staged.stage("a/x.json").write_text("new x", encoding="utf-8")
        raise RuntimeError("boom")
    assert (tmp_path / "a" / "x.json").read_text(encoding="utf-8") == "old x"
    assert not (tmp_path / STAGING_DIRNAME).exists()


def test_paths_cannot_leave_the_tree(tmp_path: Path) -> None:
    with StagedTree(tmp_path, "t") as staged, pytest.raises(ValueError):
        staged.stage("../outside.json")


def test_an_interrupted_commit_is_rolled_forward(tmp_path: Path) -> None:
    _live(tmp_path)
    attempt = tmp_path / STAGING_DIRNAME / "crashed-999999999.1"
    (attempt / "a").mkdir(parents=True)
    (attempt / "a" / "x.json").write_text("new x", encoding="utf-8")
    (attempt / COMMITTING_MARKER).write_text(json.dumps({"staged": ["a/x.json"], "removals": ["b.json"]}))
    abandoned = tmp_path / STAGING_DIRNAME / "abandoned-999999998.1"
    abandoned.mkdir()
    (abandoned / "b.json").write_text("half", encoding="utf-8")
    actions = recover_staging(tmp_path)
    assert sorted(actions) == ["discarded abandoned-999999998.1", "rolled forward crashed-999999999.1"]
    assert (tmp_path / "a" / "x.json").read_text(encoding="utf-8") == "new x"
    assert not (tmp_path / "b.json").exists()
    assert not (tmp_path / STAGING_DIRNAME).exists()


def test_staging_of_a_running_process_is_left_alone(tmp_path: Path) -> None:
    busy = tmp_path / STAGING_DIRNAME / f"other-{os.getppid()}.1"
    busy.mkdir(parents=True)
    ours = tmp_path / STAGING_DIRNAME / f"thread-{os.getpid()}.2"
    ours.mkdir()
    assert recover_staging(tmp_path) == []
    assert busy.is_dir() and ours.is_dir()
