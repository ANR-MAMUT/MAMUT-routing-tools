"""MAMUT-routing-tools: local OSM acquisition, road-graph engine, route
geometry, and instance generation for the MAMUT-routing benchmark project."""

from importlib.metadata import PackageNotFoundError, version as _installed_version

# Derived from the installed distribution metadata, whose single source is the
# `version` key in pyproject.toml. Never hardcode it here: a literal silently
# drifts from the real release, which makes "which version are you running?"
# unanswerable exactly when it matters.
try:
    __version__ = _installed_version("mamut-routing-tools")
except PackageNotFoundError:  # not installed (e.g. run straight from a source tree)
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
