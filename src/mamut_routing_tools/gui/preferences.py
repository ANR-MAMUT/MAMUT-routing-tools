"""Durable UI preferences for the workbench GUI.

The browser's own ``localStorage`` is keyed by origin *including the port*, and
``gui start`` picks a free port on every launch, so anything stored there is lost
the next time the server comes up. Preferences that should outlive a restart —
the theme and the panel layout — therefore live in the workspace instead, at
``<workspace>/state/preferences.json``, and are injected into the page so the
first paint already carries them.

Only the keys below are stored. The file is read back and embedded in HTML, so
accepting arbitrary client JSON here would be an injection route; unknown keys
are dropped rather than round-tripped.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mamut_routing_tools.workspace import state_dir

THEMES = ("dark", "light")
# Matches the clamps in static/layout.js. The server repeats them because a
# preferences file can also be hand-edited.
MIN_PANEL_WIDTH = 240
MAX_PANEL_WIDTH = 1200


def _clean_layout(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    layout: dict[str, Any] = {}
    for key in ("leftWidth", "rightWidth"):
        raw = value.get(key)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            continue
        layout[key] = int(min(MAX_PANEL_WIDTH, max(MIN_PANEL_WIDTH, round(raw))))
    for key in ("leftCollapsed", "rightCollapsed"):
        raw = value.get(key)
        if isinstance(raw, bool):
            layout[key] = raw
    return layout or None


def _clean(payload: Any) -> dict[str, Any]:
    """Keep only the recognised preference keys, with values coerced to shape."""
    if not isinstance(payload, dict):
        return {}
    cleaned: dict[str, Any] = {}
    theme = payload.get("theme")
    if theme in THEMES:
        cleaned["theme"] = theme
    layout = _clean_layout(payload.get("layout"))
    if layout is not None:
        cleaned["layout"] = layout
    return cleaned


class PreferenceStore:
    """Read/merge the workspace preference file. Never raises on a bad file."""

    def __init__(self, workspace: Path) -> None:
        self._workspace = workspace

    @property
    def path(self) -> Path:
        return state_dir(self._workspace) / "preferences.json"

    def read(self) -> dict[str, Any]:
        path = self.path
        if not path.is_file():
            return {}
        try:
            return _clean(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            # A corrupt or unreadable preferences file must never stop the GUI
            # from starting; the defaults are perfectly usable.
            return {}

    def merge(self, updates: Any) -> dict[str, Any]:
        """Apply *updates* over the stored preferences and persist the result."""
        cleaned = _clean(updates)
        if not cleaned:
            return self.read()
        merged = self.read()
        for key, value in cleaned.items():
            if key == "layout" and isinstance(merged.get("layout"), dict):
                merged["layout"] = {**merged["layout"], **value}
            else:
                merged[key] = value
        try:
            path = self.path
            path.parent.mkdir(parents=True, exist_ok=True)
            # Write via a sibling temp file so a crash mid-write cannot leave a
            # truncated JSON file behind.
            temp = path.with_suffix(".json.tmp")
            temp.write_text(json.dumps(merged, indent=2), encoding="utf-8")
            temp.replace(path)
        except OSError:
            pass  # read-only workspace — preferences just won't persist
        return merged

    def as_script_literal(self) -> str:
        """The stored preferences as a JS object literal safe to inline in HTML.

        ``<`` is escaped so a value can never close the surrounding ``<script>``
        element or open an HTML comment.
        """
        return json.dumps(self.read()).replace("<", "\\u003c")
