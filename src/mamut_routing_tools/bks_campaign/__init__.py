"""BKS campaigns: many solver runs whose winners become BKS and whose losers are data.

``mamut-tools bks-campaign plan|baseline|run|harvest|report|status SPEC``, driven by
a TOML spec (``spec``). The steps, as the Mamut2026 campaigns ran them by hand:

1. ``baseline``: freeze the stored BKS before anything runs;
2. ``run``: a resumable, memory-aware process pool; each run is checked and
   archived as ``runs/<key>.json`` (winner or not), never written as a BKS;
3. ``harvest``: the single writer -- each instance's best run, unless clearly
   worse than the baseline, goes through the BKS store (exact costs: strictly
   better only, ties kept);
4. ``report``: ``ledger.csv`` and ``cross-solver.csv``.

Distinct from ``mamut_routing_tools.campaign``, which designs benchmark instances.
"""

from mamut_routing_tools.bks_campaign.archive import load_runs, run_key
from mamut_routing_tools.bks_campaign.harvest import (
    campaign_status,
    harvest,
    harvest_history,
    read_baseline,
    write_baseline,
    write_reports,
)
from mamut_routing_tools.bks_campaign.plan import CampaignPlan, Task, build_plan, select_instances
from mamut_routing_tools.bks_campaign.runner import execute_task, memory_budget_gb, run_tasks
from mamut_routing_tools.bks_campaign.spec import CampaignSpec, load_spec, parse_seeds

__all__ = [
    "CampaignPlan",
    "CampaignSpec",
    "Task",
    "build_plan",
    "campaign_status",
    "execute_task",
    "harvest",
    "harvest_history",
    "load_runs",
    "load_spec",
    "memory_budget_gb",
    "parse_seeds",
    "read_baseline",
    "run_key",
    "run_tasks",
    "select_instances",
    "write_baseline",
    "write_reports",
]
