"""All-or-nothing replacement of a set of files under a live tree.

A generation step that rewrites several files of a published tree (an
instance, its sidecars and its twins) must not leave the tree half-rewritten
when a later step fails: the files reference each other by sha256, so a mix of
old and new is unloadable. :class:`StagedTree` writes every new file into a
mirror of the live tree under ``<live_root>/.mamut-staging/<label>/``,
where it can be verified with the ordinary loaders (sidecar paths inside the
instance files are relative to the collection root, so passing
``collection_root=staged.root`` resolves them against the staged copies), and
only then moves the files into place.

``commit`` first writes a ``COMMITTING`` list of the staged files, then renames
each one over its live counterpart (``os.replace``: same filesystem, atomic per
file, a hard-linked live file is replaced rather than truncated). A crash in
that window is rolled forward the next time a :class:`StagedTree` is opened on
the same root; a staging directory without the marker is a failed attempt and
is discarded. The lib's discovery and the release builder skip
``.mamut-staging``.
"""

from __future__ import annotations

import json
import os
import shutil
import threading
from pathlib import Path
from typing import Self

STAGING_DIRNAME = ".mamut-staging"
COMMITTING_MARKER = "COMMITTING"


class StagedTree:
    """Stage files for ``live_root``; ``commit`` publishes them, anything else discards them."""

    def __init__(self, live_root: str | Path, label: str) -> None:
        self.live_root = Path(live_root)
        recover_staging(self.live_root)
        self.root = self.live_root / STAGING_DIRNAME / f"{label}-{os.getpid()}.{threading.get_ident()}"
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True)
        self._staged: list[str] = []
        self._removals: list[str] = []
        self._closed = False

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info) -> None:
        if not self._closed:
            self.discard()

    def _relative(self, path: str | Path) -> str:
        candidate = Path(path)
        if candidate.is_absolute():
            candidate = candidate.relative_to(self.live_root)
        if ".." in candidate.parts:
            raise ValueError(f"staged path leaves the tree: {path}")
        return candidate.as_posix()

    def path(self, live_path: str | Path) -> Path:
        """Where ``live_path`` (absolute under the live root, or relative) lives in the staging tree."""
        return self.root / self._relative(live_path)

    def stage(self, live_path: str | Path) -> Path:
        """Return the staging path to write ``live_path``'s new content to; it is published on commit."""
        relative = self._relative(live_path)
        if relative not in self._staged:
            self._staged.append(relative)
        target = self.root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def link_live(self, live_path: str | Path) -> Path:
        """Mirror an unchanged live file into the staging tree for verification; never published."""
        relative = self._relative(live_path)
        source = self.live_root / relative
        target = self.root / relative
        if target.exists():
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)
        return target

    def remove_on_commit(self, live_path: str | Path) -> None:
        """Delete ``live_path`` from the live tree when the staged files are published."""
        relative = self._relative(live_path)
        if relative not in self._removals:
            self._removals.append(relative)

    @property
    def staged(self) -> list[str]:
        return list(self._staged)

    def commit(self) -> list[Path]:
        """Publish every staged file (and apply removals); returns the live paths written."""
        missing = [relative for relative in self._staged if not (self.root / relative).is_file()]
        if missing:
            raise FileNotFoundError(f"staged files were never written: {missing}")
        marker = self.root / COMMITTING_MARKER
        marker.write_text(json.dumps({"staged": self._staged, "removals": self._removals}), encoding="utf-8")
        written = _apply(self.live_root, self.root, self._staged, self._removals)
        shutil.rmtree(self.root)
        _remove_empty_staging_dir(self.live_root)
        self._closed = True
        return written

    def discard(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root)
        _remove_empty_staging_dir(self.live_root)
        self._closed = True


def _apply(live_root: Path, staging_root: Path, staged: list[str], removals: list[str]) -> list[Path]:
    written = []
    for relative in staged:
        source = staging_root / relative
        target = live_root / relative
        if source.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, target)
        written.append(target)
    for relative in removals:
        (live_root / relative).unlink(missing_ok=True)
    return written


def _remove_empty_staging_dir(live_root: Path) -> None:
    staging = live_root / STAGING_DIRNAME
    try:
        staging.rmdir()
    except OSError:
        pass


def _owner_alive(attempt: Path) -> bool:
    """Whether the process that owns a staging dir (``<label>-<pid>.<thread>``) is still running.

    The current process counts as alive: its threads (GUI jobs) clean up their
    own staging dirs.
    """
    pid_text = attempt.name.rsplit("-", 1)[-1].split(".", 1)[0]
    if not pid_text.isdigit():
        return False
    pid = int(pid_text)
    if pid == os.getpid():
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def recover_staging(live_root: str | Path) -> list[str]:
    """Finish interrupted commits and drop abandoned staging dirs under ``live_root``; returns what was done.

    Only dirs whose owning process is gone are touched: builds of other bases
    may be staging into the same collection concurrently.
    """
    staging = Path(live_root) / STAGING_DIRNAME
    if not staging.is_dir():
        return []
    actions = []
    for attempt in sorted(staging.iterdir()):
        if not attempt.is_dir() or _owner_alive(attempt):
            continue
        marker = attempt / COMMITTING_MARKER
        if marker.is_file():
            plan = json.loads(marker.read_text(encoding="utf-8"))
            _apply(Path(live_root), attempt, plan["staged"], plan["removals"])
            actions.append(f"rolled forward {attempt.name}")
        else:
            actions.append(f"discarded {attempt.name}")
        shutil.rmtree(attempt)
    _remove_empty_staging_dir(Path(live_root))
    return actions
