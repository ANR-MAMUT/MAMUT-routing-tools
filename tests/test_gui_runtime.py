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
