"""Designed-diversity benchmark campaigns.

Generating a benchmark is not the same problem as generating an instance. An
instance generator answers "make me one of these"; a campaign has to answer
"which ones, so that the set as a whole is worth reporting results on".

The pieces, in the order a campaign uses them:

- :mod:`~.city_profile` measures how far each city's road network departs from
  the Euclidean plane, and strata the cities by it.
- :mod:`~.design` enumerates a balanced candidate pool and evaluates each
  candidate cheaply -- selection and demands only, no distance matrices.
- :mod:`~.descriptors` turns coordinates and demands into the features that say
  what an instance is like, and (after generation) how far its road metrics
  diverge from its Euclidean one.
- :mod:`~.select` keeps the subset that spans the feature space while honouring
  the coverage quotas in :mod:`~.quotas`.
- :mod:`~.poi_capacity` measures the ceiling on POI-only customers per city,
  which is what decides whether a large real-amenity instance is buildable
  there at all.

Generation itself stays where it was: ``generation.single`` for the artifacts
and ``family.build_base`` for the published collection.
"""

from mamut_routing_tools.campaign.city_profile import (
    STRATA,
    CityProfile,
    assign_strata,
    load_profiles,
    profile_cities,
    profile_city,
    save_profiles,
)
from mamut_routing_tools.campaign.descriptors import (
    SELECTION_FEATURES,
    InstanceDescriptors,
    MetricDivergence,
    compute_descriptors,
    compute_divergence,
)
from mamut_routing_tools.campaign.design import (
    CAMPAIGN_POI_CATEGORIES,
    EXCLUDED_POI_CATEGORIES,
    LARGE_SIZE_GRID,
    MAIN_SIZE_GRID,
    POI_SIZE_GRID,
    CandidateSpec,
    EvaluatedCandidate,
    enumerate_candidates,
    enumerate_for_rungs,
    evaluate_candidate,
    evaluate_candidates,
    resolve_cities,
    stable_seed,
)
from mamut_routing_tools.campaign.ladder import (
    DEFAULT_HEADROOM,
    DEFAULT_METHOD_WEIGHTS,
    LadderPlan,
    RungAssignment,
    UnfilledRung,
    assign_rungs,
    load_plan,
    plan_summary,
    save_plan,
    size_ladder,
    weighted_column,
)
from mamut_routing_tools.campaign.poi_capacity import (
    PoiCapacity,
    cities_supporting,
    categories_digest,
    load_capacities,
    measure_capacity,
    measure_cities,
    save_capacities,
)
from mamut_routing_tools.campaign.quotas import (
    POI_SIZE_BUCKETS,
    SIZE_BUCKETS,
    ladder_tier_quotas,
    large_tier_quotas,
    main_tier_quotas,
    poi_size_bucket,
    poi_tier_quotas,
    size_bucket,
)
from mamut_routing_tools.campaign.select import (
    Quota,
    QuotaInfeasibleError,
    diversity_report,
    select_maxmin,
)

__all__ = [
    "assign_rungs",
    "assign_strata",
    "CAMPAIGN_POI_CATEGORIES",
    "CandidateSpec",
    "categories_digest",
    "cities_supporting",
    "CityProfile",
    "compute_descriptors",
    "compute_divergence",
    "DEFAULT_HEADROOM",
    "DEFAULT_METHOD_WEIGHTS",
    "diversity_report",
    "enumerate_candidates",
    "enumerate_for_rungs",
    "evaluate_candidate",
    "evaluate_candidates",
    "EvaluatedCandidate",
    "EXCLUDED_POI_CATEGORIES",
    "InstanceDescriptors",
    "ladder_tier_quotas",
    "LadderPlan",
    "LARGE_SIZE_GRID",
    "large_tier_quotas",
    "load_capacities",
    "load_plan",
    "load_profiles",
    "MAIN_SIZE_GRID",
    "main_tier_quotas",
    "measure_capacity",
    "measure_cities",
    "MetricDivergence",
    "plan_summary",
    "poi_size_bucket",
    "POI_SIZE_BUCKETS",
    "POI_SIZE_GRID",
    "poi_tier_quotas",
    "PoiCapacity",
    "profile_cities",
    "profile_city",
    "Quota",
    "QuotaInfeasibleError",
    "resolve_cities",
    "RungAssignment",
    "save_capacities",
    "save_plan",
    "save_profiles",
    "select_maxmin",
    "SELECTION_FEATURES",
    "size_bucket",
    "SIZE_BUCKETS",
    "size_ladder",
    "stable_seed",
    "STRATA",
    "UnfilledRung",
    "weighted_column",
]
