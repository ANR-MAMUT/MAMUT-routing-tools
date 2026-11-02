"""BKS campaign specification (TOML, validated with pydantic).

.. code-block:: toml

    [campaign]
    name = "mamut2026-bks-4"
    out = "runs/mamut2026-bks-4"        # relative to this file; never inside a benchmark collection
    authors = "Jane Doe (jdoe)"         # credited in every BKS the harvest writes
    # solvers_config = "solvers.toml"   # default: the user's solvers config

    [select]
    benchmarks_dir = "../benchmarks"
    problem_types = ["CVRP"]
    families = ["Mamut2026"]
    metrics = ["euclidean"]
    min_n = 700
    # max_n, name_regex, instance_ids
    # [select.objective]                # per problem type; default: the family's published objective
    # VRPTW = "HierarchicalVehicleCost"

    [budget]                            # T(n) = clamp(per_customer_s * n, min_s, max_s) * time_scale
    per_customer_s = 21.6
    min_s = 7200
    max_s = 86400
    # fixed_s = 600                     # instead of per_customer_s
    time_scale = 1.0

    [[runs]]
    solver = "filo2"
    seeds = "1-5"

    [[runs]]
    solver = "pyvrp"
    kind = "pyvrp-polish"
    seeds = "3000-3003"
    budget_factor = 1.5
    warm_start = "best"                 # from the best archived run or the stored BKS

    [resources]
    workers = 90
    memory_gb = "auto"                  # 80 % of MemAvailable, or a number
    nice = 5
"""

from __future__ import annotations

import math
import re
import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from mamut_routing_lib.enums import MetricVariant, ObjectiveFunction, ProblemType

__all__ = ["BudgetSpec", "CampaignSpec", "RunSpec", "load_spec", "parse_seeds"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


def parse_seeds(value: str | int | list[int]) -> list[int]:
    """``"1-5,7"`` -> ``[1, 2, 3, 4, 5, 7]``; a list or an int passes through."""
    if isinstance(value, int):
        return [value]
    if isinstance(value, list):
        return [int(item) for item in value]
    seeds: list[int] = []
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            low, high = part.split("-", 1)
            if int(high) < int(low):
                raise ValueError(f"empty seed range {part!r}")
            seeds.extend(range(int(low), int(high) + 1))
        else:
            seeds.append(int(part))
    if not seeds:
        raise ValueError(f"no seeds in {value!r}")
    return seeds


class CampaignSection(_Strict):
    name: str
    out: Path
    authors: str
    solvers_config: Path | None = None

    @field_validator("authors")
    @classmethod
    def _authors(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("authors must name who is credited in the BKS")
        return value

    @field_validator("name")
    @classmethod
    def _name(cls, value: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9._-]+", value):
            raise ValueError("name may only contain letters, digits, '.', '_' and '-'")
        return value


class SelectSection(_Strict):
    benchmarks_dir: Path
    problem_types: list[ProblemType] = Field(default_factory=list)
    families: list[str] = Field(default_factory=list)
    metrics: list[MetricVariant] = Field(default_factory=list)
    instance_ids: list[str] = Field(default_factory=list)
    min_n: int | None = None
    max_n: int | None = None
    name_regex: str | None = None
    objective: dict[ProblemType, ObjectiveFunction] = Field(default_factory=dict)


class BudgetSpec(_Strict):
    fixed_s: float | None = None
    per_customer_s: float | None = None
    min_s: float = 1.0
    max_s: float | None = None
    time_scale: float = 1.0

    @model_validator(mode="after")
    def _one_rule(self) -> "BudgetSpec":
        if (self.fixed_s is None) == (self.per_customer_s is None):
            raise ValueError("set exactly one of budget.fixed_s and budget.per_customer_s")
        return self

    def seconds(self, num_customers: int, factor: float = 1.0) -> int:
        """The wall-clock budget for an instance with ``num_customers`` customers."""
        if self.fixed_s is not None:
            base = self.fixed_s
        else:
            base = float(self.per_customer_s) * num_customers
            base = max(base, self.min_s)
            if self.max_s is not None:
                base = min(base, self.max_s)
        return max(1, int(round(base * self.time_scale * factor)))


class RunSpec(_Strict):
    solver: str
    seeds: str | int | list[int] = "1"
    kind: str | None = None
    budget_factor: float = 1.0
    warm_start: Literal["none", "best"] = "none"
    memory_gb: float | None = None
    options: dict[str, Any] = Field(default_factory=dict)

    @property
    def run_kind(self) -> str:
        return self.kind or f"{self.solver}-seed"

    def seed_list(self) -> list[int]:
        return parse_seeds(self.seeds)

    @field_validator("kind")
    @classmethod
    def _kind(cls, value: str | None) -> str | None:
        if value is not None and not re.fullmatch(r"[A-Za-z0-9._-]+", value):
            raise ValueError("kind may only contain letters, digits, '.', '_' and '-'")
        return value


class ResourcesSection(_Strict):
    workers: int = 1
    memory_gb: float | Literal["auto"] = "auto"
    nice: int | None = 5

    @field_validator("workers")
    @classmethod
    def _workers(cls, value: int) -> int:
        if value < 1:
            raise ValueError("workers must be at least 1")
        return value


class CampaignSpec(_Strict):
    campaign: CampaignSection
    select: SelectSection
    budget: BudgetSpec
    runs: list[RunSpec]
    resources: ResourcesSection = Field(default_factory=ResourcesSection)
    #: The spec file (set by ``load_spec``); relative paths are resolved against its folder.
    path: Path | None = None

    @model_validator(mode="after")
    def _runs(self) -> "CampaignSpec":
        if not self.runs:
            raise ValueError("a campaign needs at least one [[runs]] entry")
        kinds = [run.run_kind for run in self.runs]
        if len(set(kinds)) != len(kinds):
            raise ValueError("every [[runs]] entry needs a distinct kind (set kind = ... for repeated solvers)")
        return self

    @property
    def out(self) -> Path:
        return self.campaign.out

    def resolved(self, base: Path) -> "CampaignSpec":
        def absolute(path: Path | None) -> Path | None:
            if path is None:
                return None
            path = path.expanduser()
            return path if path.is_absolute() else (base / path).resolve()

        campaign = self.campaign.model_copy(
            update={"out": absolute(self.campaign.out), "solvers_config": absolute(self.campaign.solvers_config)}
        )
        select = self.select.model_copy(update={"benchmarks_dir": absolute(self.select.benchmarks_dir)})
        return self.model_copy(update={"campaign": campaign, "select": select})


def load_spec(path: str | Path) -> CampaignSpec:
    """Read and validate a campaign spec; paths become absolute, ``out`` is refused inside a collection."""
    from mamut_routing_lib.sidecars import find_collection_root

    spec_path = Path(path).resolve()
    data = tomllib.loads(spec_path.read_text(encoding="utf-8"))
    spec = CampaignSpec.model_validate({**data, "path": spec_path}).resolved(spec_path.parent)
    out = spec.campaign.out
    try:
        inside = find_collection_root(out)
    except Exception:  # noqa: BLE001 - no collection marker above the out directory
        inside = None
    benchmarks = spec.select.benchmarks_dir
    if inside is not None or (benchmarks is not None and _is_within(out, benchmarks)):
        raise ValueError(f"campaign.out ({out}) must not lie inside a benchmark tree: workers never write next to instances")
    if not math.isfinite(spec.budget.time_scale) or spec.budget.time_scale <= 0:
        raise ValueError("budget.time_scale must be positive")
    return spec


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True
