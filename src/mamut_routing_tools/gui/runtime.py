"""GUI server lifecycle: start (detached subprocess), stop, status.

State lives in ``<workspace>/state/gui.json`` (pid, port, token, url) with the
server log next to it. The CLI owns the server process; nothing else does.
"""

from __future__ import annotations

import json
import os
import secrets
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from mamut_routing_tools.workspace import state_dir


def _state_path(workspace: Path) -> Path:
    return state_dir(workspace) / "gui.json"


def _log_path(workspace: Path) -> Path:
    return state_dir(workspace) / "gui.log"


def read_state(workspace: Path) -> dict[str, Any] | None:
    path = _state_path(workspace)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def _free_port(host: str) -> int:
    with socket.socket() as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def _health_ok(host: str, port: int, token: str) -> bool:
    import httpx

    try:
        response = httpx.get(
            f"http://{host}:{port}/healthz", headers={"X-Mamut-Token": token}, timeout=2.0
        )
        return response.status_code == 200
    except httpx.HTTPError:
        return False


def status(workspace: Path) -> dict[str, Any]:
    state = read_state(workspace)
    if state is None:
        return {"running": False}
    alive = _pid_alive(int(state["pid"]))
    healthy = alive and _health_ok(state["host"], int(state["port"]), state["token"])
    return {"running": alive, "healthy": healthy, **state}


def start(workspace: Path, *, host: str = "127.0.0.1", port: int = 0) -> dict[str, Any]:
    current = status(workspace)
    if current.get("running"):
        return {**current, "already_running": True}

    resolved_port = port or _free_port(host)
    token = secrets.token_urlsafe(24)
    env = dict(os.environ)
    env["MAMUT_GUI_WORKSPACE"] = str(workspace)
    env["MAMUT_GUI_TOKEN"] = token
    log_file = _log_path(workspace).open("ab")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "mamut_routing_tools.gui.asgi:app",
            "--host",
            host,
            "--port",
            str(resolved_port),
            "--log-level",
            "warning",
        ],
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    state = {
        "pid": process.pid,
        "host": host,
        "port": resolved_port,
        "token": token,
        "url": f"http://{host}:{resolved_port}/?token={token}",
        "workspace": str(workspace),
    }
    _state_path(workspace).write_text(json.dumps(state, indent=1) + "\n")

    for _ in range(60):
        if not _pid_alive(process.pid):
            raise RuntimeError(f"GUI server exited during startup; see {_log_path(workspace)}")
        if _health_ok(host, resolved_port, token):
            return state
        time.sleep(0.5)
    raise RuntimeError(f"GUI server did not become healthy; see {_log_path(workspace)}")


def stop(workspace: Path) -> dict[str, Any]:
    state = read_state(workspace)
    if state is None:
        return {"stopped": False, "reason": "not running"}
    pid = int(state["pid"])
    if _pid_alive(pid):
        os.kill(pid, signal.SIGTERM)
        for _ in range(20):
            if not _pid_alive(pid):
                break
            time.sleep(0.25)
        if _pid_alive(pid):
            os.kill(pid, signal.SIGKILL)
    _state_path(workspace).unlink(missing_ok=True)
    return {"stopped": True, "pid": pid}
