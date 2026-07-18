"""Uvicorn entry point for the detached GUI server process; configuration
travels via environment variables set by `mamut-tools gui start`."""

from __future__ import annotations

import os
from pathlib import Path

from mamut_routing_tools.gui.server import create_app

_workspace = os.environ.get("MAMUT_GUI_WORKSPACE")
_token = os.environ.get("MAMUT_GUI_TOKEN")
if not _workspace or not _token:
    raise RuntimeError("MAMUT_GUI_WORKSPACE and MAMUT_GUI_TOKEN must be set (use 'mamut-tools gui start')")

app = create_app(Path(_workspace), _token)
