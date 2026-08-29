"""Naming and on-disk layout of a MAMUT benchmark collection (v2, Stream 12').

One base instance = one customer set = one name across every problem type:
``poryos-<city>-n<N>-<method>`` (method in {poi, hyb}). TD subinstances append
``-<model>-<intensity>`` (6 per base). Extra static-only VRPTW TW sets append
``-tw-<set>`` (the ``tw-`` prefix cannot collide with the TD tags); the
TD-paired VRPTW instance keeps the bare base name, mirroring the TDVRPTW
twins that embed the same windows. The family lives in a single family-first
collection repo mounted at ``benchmarks/Poryos2026/``:

    sidecars/<city>/n=<N>/<base>/            shared sidecars of the base
    CVRP/<metric>/<city>/n=<N>/<base>/
    VRPTW/fastest/<city>/n=<N>/<base>/       one file per TW set
    TDVRP/<city>/n=<N>/<base>/<sub>/
    TDVRPTW/<city>/n=<N>/<base>/<sub>/
"""

from __future__ import annotations

import re
from pathlib import Path

#: The default family. Every name/layout helper takes ``family`` so a second
#: collection (Mamut2026) can reuse this module without forking it; the default
#: keeps every existing Poryos2026 call site producing identical names.
FAMILY = "Poryos2026"
METHOD_TAGS = {"poi_categories": "poi", "parametric_attach": "par", "hybrid": "hyb"}

_RELEASE_YEAR_SUFFIX = re.compile(r"\d{4}$")

TW_SET_TD_SHARED = "td-shared"
TW_SET_TIGHT = "tight"
TW_SET_SPREAD = "spread"
EXTRA_TW_SETS = (TW_SET_TIGHT, TW_SET_SPREAD)
ALL_TW_SETS = (TW_SET_TD_SHARED, *EXTRA_TW_SETS)


def family_prefix(family: str = FAMILY) -> str:
    """A family's base-name prefix: its name minus the trailing release year.

    ``Poryos2026 -> poryos``, ``Mamut2026 -> mamut``. A rule rather than a
    lookup table, so adding a family means picking a name and nothing else.
    """
    stem = _RELEASE_YEAR_SUFFIX.sub("", family)
    if not stem:
        raise ValueError(f"family {family!r} has no name part before its release year")
    return stem.lower()


#: Families whose base names carry the fleet lower bound, CVRPLIB style.
#:
#: ``Mamut2026`` follows the X and XL sets, whose names read ``X-n101-k25``: the
#: ``k`` is the minimum number of vehicles the demands need, published as a lower
#: bound rather than as a cap. ``Poryos2026`` predates the convention and its
#: names are already published, so it stays as it is -- which is the only reason
#: this is a set and not simply the rule.
FAMILIES_WITH_FLEET_IN_NAME = frozenset({"Mamut2026"})


def base_instance_name(
    city: str,
    n: int,
    method_tag: str,
    family: str = FAMILY,
    *,
    route_count: int | None = None,
) -> str:
    """``<family>-<city>-n<N>[-k<K>]-<method>``.

    ``route_count`` is the bin-packing minimum fleet. It is only written into the
    name for families in :data:`FAMILIES_WITH_FLEET_IN_NAME`, and passing it for
    a family that does not want it is a caller error rather than a silent no-op:
    a name that sometimes carries ``k`` and sometimes does not is unparseable.
    """
    prefix = f"{family_prefix(family)}-{city}-n{n}"
    if family in FAMILIES_WITH_FLEET_IN_NAME:
        if route_count is None:
            raise ValueError(
                f"{family} base names carry the fleet lower bound; pass route_count"
            )
        prefix = f"{prefix}-k{int(route_count)}"
    elif route_count is not None:
        raise ValueError(
            f"{family} base names do not carry a fleet lower bound; drop route_count"
        )
    return f"{prefix}-{method_tag}".lower()


def subinstance_name(model: str, intensity: str) -> str:
    return f"{model}-{intensity}".lower()


def td_instance_name(base: str, model: str, intensity: str) -> str:
    return f"{base}-{subinstance_name(model, intensity)}"


def vrptw_instance_name(base: str, tw_set: str) -> str:
    """TD-paired set: the bare base name; static-only sets: ``<base>-tw-<set>``."""
    if tw_set == TW_SET_TD_SHARED:
        return base
    if tw_set not in EXTRA_TW_SETS:
        raise ValueError(f"unknown TW set {tw_set!r} (expected one of {ALL_TW_SETS})")
    return f"{base}-tw-{tw_set}"


def sidecar_dir(collection_root: str | Path, city: str, n: int, base: str) -> Path:
    return Path(collection_root) / "sidecars" / city / f"n={n}" / base


def sidecar_relpath(city: str, n: int, base: str, filename: str) -> str:
    """Collection-root-relative sidecar path (the form stored in instance refs)."""
    return f"sidecars/{city}/n={n}/{base}/{filename}"


def cvrp_dir(collection_root: str | Path, metric: str, city: str, n: int, base: str) -> Path:
    return Path(collection_root) / "CVRP" / metric / city / f"n={n}" / base


def vrptw_dir(collection_root: str | Path, city: str, n: int, base: str) -> Path:
    return Path(collection_root) / "VRPTW" / "fastest" / city / f"n={n}" / base


def td_instance_dir(
    collection_root: str | Path, problem_type: str, city: str, n: int, base: str, sub: str
) -> Path:
    if problem_type not in ("TDVRP", "TDVRPTW"):
        raise ValueError(f"problem_type must be TDVRP or TDVRPTW, got {problem_type!r}")
    return Path(collection_root) / problem_type / city / f"n={n}" / base / sub
