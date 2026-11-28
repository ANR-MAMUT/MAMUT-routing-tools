from __future__ import annotations

import os
import subprocess
import sys

from mamut_routing_tools.gui.runtime import _pid_alive


def test_pid_alive_recognizes_current_process() -> None:
    assert _pid_alive(os.getpid())


def test_pid_alive_rejects_terminated_process() -> None:
    process = subprocess.Popen([sys.executable, "-c", "pass"])
    process.wait(timeout=5)

    assert not _pid_alive(process.pid)


def test_pid_alive_rejects_non_positive_pid() -> None:
    assert not _pid_alive(0)
    assert not _pid_alive(-1)


def test_start_says_when_a_benchmarks_folder_was_not_applied(tmp_path, monkeypatch) -> None:
    """A server that is already running keeps the folder it started with: `gui start` must say so."""
    from typer.testing import CliRunner

    from mamut_routing_tools.cli import app
    from mamut_routing_tools.gui import runtime

    asked = tmp_path / "benchmarks"
    running = {"already_running": True, "pid": 4242, "url": "http://127.0.0.1:1/", "benchmarks_dir": None}
    monkeypatch.setattr(runtime, "start", lambda workspace, **options: dict(running))
    arguments = ["gui", "start", "--no-open", "--output-dir", str(tmp_path / "workspace")]

    ignored = CliRunner().invoke(app, [*arguments, "--benchmarks-dir", str(asked)])
    assert ignored.exit_code == 0 and "Already running" in ignored.output
    assert "--benchmarks-dir was not applied" in ignored.output

    # The same folder the server started with, or no option at all: nothing to warn about.
    running["benchmarks_dir"] = str(asked.resolve())
    assert "was not applied" not in CliRunner().invoke(app, [*arguments, "--benchmarks-dir", str(asked)]).output
    assert "was not applied" not in CliRunner().invoke(app, arguments).output
