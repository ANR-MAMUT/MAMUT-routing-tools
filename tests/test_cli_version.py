"""CLI version reporting: the `--version` / `-V` flag and metadata agreement."""

from __future__ import annotations

from importlib.metadata import version as installed_version

from typer.testing import CliRunner

import mamut_routing_tools
from mamut_routing_tools.cli import app

runner = CliRunner()


def test_package_version_matches_distribution_metadata() -> None:
    """`__version__` must never drift from the version declared in pyproject.

    It used to be a hardcoded literal and sat three releases behind the real
    one, which made the installed version impossible to identify.
    """
    assert mamut_routing_tools.__version__ == installed_version("mamut-routing-tools")
    assert mamut_routing_tools.__version__ != "0.0.0+unknown"


def test_version_flag_reports_version_and_location() -> None:
    for flag in ("--version", "-V"):
        result = runner.invoke(app, [flag])
        assert result.exit_code == 0, f"{flag} exited {result.exit_code}"
        assert mamut_routing_tools.__version__ in result.stdout
        # The package location disambiguates a PyPI install from a source checkout.
        assert "mamut_routing_tools" in result.stdout


def test_bare_invocation_still_shows_help() -> None:
    """Adding the global callback must not turn a bare call into a no-op."""
    result = runner.invoke(app, [])
    assert "Commands" in result.stdout
