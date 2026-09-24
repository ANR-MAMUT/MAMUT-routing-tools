"""The files that belong to a generated instance, and its content identity.

A generated base ``<city>_<abbr>-n<N>-k<K>[<suffix>]`` owns every file named
``<base>_...`` (the three metric ``.vrp`` / ``.vrp.json``, ``_meta.json``,
``_manifest.json``, the VRPTW twin and its manifest, BKS files), its derive-td
sidecars ``<base>.road.json.gz`` / ``<base>.traffic-<sub>.json.gz`` and TD
twins ``<base>-<model>-<intensity>[.tdvrp].vrp.json``. Sibling bases share a
prefix (``...-k2`` and ``...-k25``, ``...-k2-2``), so ownership is an exact
pattern match, never ``startswith(base)``.

Two generations are the *same instance* when their three CVRPLIB files agree
past the ``NAME`` line (coordinates, demands, capacity and the three
matrices): :func:`instance_content_sha256`. Timestamps live in the JSON files
and are not part of the identity.
"""

from __future__ import annotations

import hashlib
import os
import re
import threading
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path

METRICS = ("shortest", "fastest", "euclidean")
_TD_TWIN = r"-(?:bpr|wave)-(?:light|moderate|heavy)(?:\.tdvrp)?\.vrp\.json"
_SIDECAR = r"\.(?:road|traffic-(?:bpr|wave)-(?:light|moderate|heavy))\.json\.gz"
CLAIM_LOCK_NAME = ".claim.lock"
_CLAIM_THREAD_LOCK = threading.Lock()


def _owner_pattern(base: str) -> re.Pattern[str]:
    return re.compile(rf"{re.escape(base)}(?:_.+|{_TD_TWIN}|{_SIDECAR})")


def belongs_to_base(name: str, base: str) -> bool:
    """Whether the file ``name`` is one of ``base``'s artifacts (and not a sibling base's)."""
    return _owner_pattern(base).fullmatch(name) is not None


def instance_artifact_paths(folder: str | Path, base: str) -> list[Path]:
    """Every file of ``folder`` that belongs to ``base``, sorted."""
    root = Path(folder)
    if not root.is_dir():
        return []
    pattern = _owner_pattern(base)
    return sorted(path for path in root.iterdir() if path.is_file() and pattern.fullmatch(path.name))


def purge_instance_artifacts(folder: str | Path, base: str) -> list[Path]:
    """Delete every artifact of ``base`` (instance files and everything derived from them)."""
    removed = instance_artifact_paths(folder, base)
    for path in removed:
        path.unlink(missing_ok=True)
    return removed


def vrp_text_sha256(text: str) -> str:
    """sha256 of a CVRPLIB text without its ``NAME`` line (the name is not content)."""
    lines = text.splitlines(keepends=True)
    if lines and lines[0].startswith("NAME"):
        lines = lines[1:]
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()


def content_sha256_of_texts(texts: Mapping[str, str]) -> str:
    """Content identity of an instance from its three metric ``.vrp`` texts."""
    digest = hashlib.sha256()
    for metric in METRICS:
        digest.update(metric.encode())
        digest.update(vrp_text_sha256(texts[metric]).encode())
    return digest.hexdigest()


def instance_content_sha256(folder: str | Path, base: str) -> str | None:
    """Content identity of the instance ``base`` on disk; ``None`` when its ``.vrp`` files are absent."""
    root = Path(folder)
    texts = {}
    for metric in METRICS:
        path = root / f"{base}_{metric}.vrp"
        if not path.is_file():
            return None
        texts[metric] = path.read_text(encoding="utf-8")
    return content_sha256_of_texts(texts)


@contextmanager
def claim_lock(folder: str | Path, *, timeout_s: float = 120.0, stale_after_s: float = 600.0) -> Iterator[None]:
    """Serialize name claims in ``folder`` across threads and processes.

    Held while a generation decides which base name it writes and writes it,
    so two jobs regenerating the same configuration cannot both take (or both
    overwrite) the same name. A lock file older than ``stale_after_s`` is a
    crashed holder's and is broken.
    """
    root = Path(folder)
    root.mkdir(parents=True, exist_ok=True)
    lock = root / CLAIM_LOCK_NAME
    deadline = time.monotonic() + timeout_s
    with _CLAIM_THREAD_LOCK:
        while True:
            try:
                handle = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                break
            except FileExistsError:
                try:
                    if time.time() - lock.stat().st_mtime > stale_after_s:
                        lock.unlink(missing_ok=True)
                        continue
                except FileNotFoundError:
                    continue
                if time.monotonic() > deadline:
                    raise TimeoutError(f"another generation holds {lock}") from None
                time.sleep(0.05)
        try:
            os.write(handle, str(os.getpid()).encode())
            os.close(handle)
            yield
        finally:
            lock.unlink(missing_ok=True)
