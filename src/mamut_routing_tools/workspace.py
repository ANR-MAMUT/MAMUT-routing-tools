"""Workspace directory resolution for the local tools.

One self-contained workspace directory holds everything a session produces
(OSM extracts under ``osmdata/``, generated instances under ``instances/``,
GUI state under ``state/``). Resolution order:

1. An explicit ``--output-dir`` / ``output_dir`` argument.
2. ``$MAMUT_TOOLS_WORKSPACE`` when set.
3. ``<repo>/.cache/mamut-tools`` when the current directory is inside a
   checkout of the tools repo (the gitignored dev-loop workspace).
4. ``~/.cache/mamut-tools`` otherwise.
"""

from __future__ import annotations

import os
from pathlib import Path

WORKSPACE_ENV = "MAMUT_TOOLS_WORKSPACE"
_REPO_MARKER = ("pyproject.toml", "src/mamut_routing_tools")


def _enclosing_tools_repo(start: Path) -> Path | None:
    for candidate in (start, *start.parents):
        if all((candidate / marker).exists() for marker in _REPO_MARKER):
            return candidate
    return None


def resolve_workspace(output_dir: str | Path | None = None, *, create: bool = True) -> Path:
    if output_dir is not None:
        workspace = Path(output_dir).expanduser()
    elif os.environ.get(WORKSPACE_ENV):
        workspace = Path(os.environ[WORKSPACE_ENV]).expanduser()
    else:
        repo = _enclosing_tools_repo(Path.cwd())
        base = repo if repo is not None else Path.home()
        workspace = base / ".cache" / "mamut-tools"
    if create:
        workspace.mkdir(parents=True, exist_ok=True)
    return workspace


def osmdata_dir(workspace: Path, *, create: bool = True) -> Path:
    path = workspace / "osmdata"
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def instances_dir(workspace: Path, *, create: bool = True) -> Path:
    path = workspace / "instances"
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def state_dir(workspace: Path, *, create: bool = True) -> Path:
    path = workspace / "state"
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def jobs_dir(workspace: Path, *, create: bool = True) -> Path:
    """Persistent GUI job records.

    Jobs are state rather than generated benchmark artefacts: retaining them
    across GUI restarts makes long-running work auditable without changing the
    existing ``instances/`` layout.
    """

    path = state_dir(workspace, create=create) / "jobs"
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def logs_dir(workspace: Path, *, create: bool = True) -> Path:
    """Per-job text logs, kept next to the persistent job records."""

    path = state_dir(workspace, create=create) / "logs"
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def solutions_dir(workspace: Path, *, create: bool = True) -> Path:
    """Validated solution runs produced by the local workbench."""

    path = workspace / "solutions"
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path
