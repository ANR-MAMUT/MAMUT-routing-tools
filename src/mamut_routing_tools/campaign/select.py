"""Pick the subset that spans the feature space, subject to coverage quotas.

Two things have to be true of a benchmark that claims diversity, and they pull
against each other:

*Spread* -- no two instances should be near-duplicates in feature space, and the
extremes should be represented. That is a max-min (farthest-point) criterion on
the z-scored descriptors.

*Coverage* -- every level of every design axis must appear often enough to be
analysable. Seven demand distributions are useless if one of them shows up
twice. Max-min alone will not deliver that: it chases outliers, and a whole
level can sit in the middle of the cloud and never get picked.

So the greedy runs under residual-feasibility guards. Before each pick it asks,
per quota, whether the remaining budget can still cover the outstanding
minimums; when a quota becomes binding, the pick is restricted to candidates
that reduce its deficit. A swap pass then improves the spread without letting
any quota slip.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Hashable, Mapping, Sequence

import numpy as np

from mamut_routing_tools.campaign.descriptors import SELECTION_FEATURES, descriptor_values
from mamut_routing_tools.campaign.design import EvaluatedCandidate


class QuotaInfeasibleError(ValueError):
    """The pool cannot satisfy the quotas within the requested budget."""


@dataclass(frozen=True)
class Quota:
    """A coverage requirement over one design axis.

    ``key`` maps a candidate to the level it occupies (its demand type, its
    city, its size bucket). ``minimum`` is how many of each level the selection
    must end up with; ``maximum`` caps a level -- ``{city: 1}`` style caps are
    written with :meth:`per_level_maximum`.
    """

    name: str
    key: Callable[[EvaluatedCandidate], Hashable]
    minimum: Mapping[Hashable, int] = ()  # type: ignore[assignment]
    maximum: Mapping[Hashable, int] | None = None
    #: Cap applied to every level, including ones not named in ``maximum``.
    per_level_maximum: int | None = None

    def cap_for(self, level: Hashable) -> int | None:
        if self.maximum is not None and level in self.maximum:
            return int(self.maximum[level])
        return self.per_level_maximum

    def deficit(self, counts: Counter) -> dict[Hashable, int]:
        return {
            level: required - counts.get(level, 0)
            for level, required in dict(self.minimum).items()
            if required > counts.get(level, 0)
        }


def _feature_matrix(candidates: Sequence[EvaluatedCandidate]) -> np.ndarray:
    """Z-scored descriptors, standardized over the *pool*.

    The pool's own spread is the right yardstick: a feature that barely varies
    across everything on offer should not dominate the distance just because its
    raw units are large, and one that varies a lot should not be flattened.
    """
    raw = np.asarray([candidate.feature_vector() for candidate in candidates], dtype=float)
    spread = raw.std(axis=0)
    # A feature that is constant across the pool carries no information; scaling
    # it by ~0 would turn float noise into the dominant axis.
    spread[spread <= 0.0] = 1.0
    return (raw - raw.mean(axis=0)) / spread


def _tie_break(candidate: EvaluatedCandidate) -> str:
    """A total order over candidates, so equal scores resolve reproducibly."""
    spec = candidate.spec
    return f"{spec.city}|{spec.n:06d}|{spec.method}|{spec.seed:010d}"


def _spread_score(distances: np.ndarray, chosen: list[int]) -> tuple[float, ...]:
    """Each chosen point's nearest-neighbour distance, sorted ascending.

    Compared lexicographically, this is max-min with built-in tie-breaking: the
    first entry is the classic minimum pairwise distance, and the rest separate
    the many swaps that leave that minimum untouched.
    """
    if len(chosen) < 2:
        return (float("inf"),)
    block = distances[np.ix_(chosen, chosen)].copy()
    np.fill_diagonal(block, np.inf)
    return tuple(sorted(block.min(axis=1)))


def _shortfall(
    quotas: Sequence[Quota],
    candidates: Sequence[EvaluatedCandidate],
    counts: dict[str, Counter],
    chosen_set: set[int],
    remaining: int,
) -> str | None:
    """Why the quotas can no longer be met, or ``None`` if they still can.

    Walks the pool per quota, and the greedy asks once per remaining candidate
    per pick, so this is the selector's cost centre: measured at 816 candidates,
    100 picks and 8 quotas, ``select_maxmin`` takes about three minutes. Left
    deliberately simple -- an incremental tally would need a fast path that has
    to be kept in agreement with this definition, which is real complexity to
    save three minutes inside a campaign whose generate phase runs for hours.

    Two ways to fail, and they need separating because the fixes differ:

    * not enough *slots* left for the outstanding minimums -- lower a quota or
      raise ``k``;
    * not enough *candidates* left at a level that still owes instances -- the
      pool is too thin there, so raise ``--per-city`` or widen the pool.

    Checked for every quota jointly rather than one at a time. A minimum that is
    satisfiable on its own can still be unreachable once another quota's caps
    are taken into account -- one instance per city, say, when the only
    remaining cities are all in the wrong stratum.
    """
    for quota in quotas:
        deficit = quota.deficit(counts[quota.name])
        outstanding = sum(deficit.values())
        if outstanding > remaining:
            return (
                f"quota {quota.name!r} still needs {outstanding} instance(s) with only "
                f"{remaining} slot(s) left: {deficit}"
            )
        if not deficit:
            continue
        available: Counter = Counter()
        for index, candidate in enumerate(candidates):
            if index in chosen_set:
                continue
            if any(
                (cap := other.cap_for(other.key(candidate))) is not None
                and counts[other.name][other.key(candidate)] >= cap
                for other in quotas
            ):
                continue
            available[quota.key(candidate)] += 1
        for level, short in deficit.items():
            if available[level] < short:
                return (
                    f"quota {quota.name!r} needs {short} more instance(s) at {level!r} "
                    f"but only {available[level]} admissible candidate(s) remain"
                )
    return None


def select_maxmin(
    candidates: Sequence[EvaluatedCandidate],
    k: int,
    quotas: Sequence[Quota] = (),
    *,
    swap_passes: int = 2,
) -> list[EvaluatedCandidate]:
    """``k`` candidates spread over feature space and honouring every quota."""
    if k <= 0:
        raise ValueError("k must be positive")
    if len(candidates) < k:
        raise QuotaInfeasibleError(
            f"pool has {len(candidates)} usable candidates, fewer than the requested {k}"
        )

    features = _feature_matrix(candidates)
    distances = np.linalg.norm(features[:, None, :] - features[None, :, :], axis=2)
    order = sorted(range(len(candidates)), key=lambda i: _tie_break(candidates[i]))
    counts: dict[str, Counter] = {quota.name: Counter() for quota in quotas}
    chosen: list[int] = []
    chosen_set: set[int] = set()

    def level(quota: Quota, index: int) -> Hashable:
        return quota.key(candidates[index])

    def admissible(index: int) -> bool:
        return all(
            (cap := quota.cap_for(level(quota, index))) is None
            or counts[quota.name][level(quota, index)] < cap
            for quota in quotas
        )

    def deficits_reduced(index: int) -> int:
        """How many outstanding quota minimums taking ``index`` would advance."""
        return sum(
            1
            for quota in quotas
            if level(quota, index) in quota.deficit(counts[quota.name])
        )

    def outstanding() -> str:
        return ", ".join(
            f"{quota.name}={quota.deficit(counts[quota.name])}"
            for quota in quotas
            if quota.deficit(counts[quota.name])
        )

    def keeps_quotas_reachable(index: int) -> bool:
        """Would taking ``index`` leave every quota still satisfiable?

        A one-step lookahead rather than a "is this quota binding right now"
        test: it is what stops the greedy walking into a corner where the
        remaining budget cannot cover what the remaining candidates offer.
        """
        trial = {name: counter.copy() for name, counter in counts.items()}
        for quota in quotas:
            trial[quota.name][level(quota, index)] += 1
        return (
            _shortfall(quotas, candidates, trial, chosen_set | {index}, k - len(chosen) - 1)
            is None
        )

    def take(index: int) -> None:
        chosen.append(index)
        chosen_set.add(index)
        for quota in quotas:
            counts[quota.name][level(quota, index)] += 1

    # Fail before doing any work when the pool simply cannot meet the quotas.
    upfront = _shortfall(quotas, candidates, counts, chosen_set, k)
    if upfront is not None:
        raise QuotaInfeasibleError(upfront)

    # Seed with the most extreme admissible candidate: farthest from the pool's
    # centre of mass, so the first pick is an edge of the cloud rather than an
    # arbitrary point in the middle of it.
    from_centre = np.linalg.norm(features - features.mean(axis=0), axis=1)
    start = [i for i in order if admissible(i) and keeps_quotas_reachable(i)]
    if not start:
        raise QuotaInfeasibleError("no first pick leaves the quotas satisfiable")
    take(max(start, key=lambda i: (from_centre[i], _tie_break(candidates[i]))))

    while len(chosen) < k:
        open_picks = [i for i in order if i not in chosen_set and admissible(i)]
        if not open_picks:
            capped = ", ".join(
                quota.name for quota in quotas if quota.per_level_maximum or quota.maximum
            )
            raise QuotaInfeasibleError(
                f"only {len(chosen)} of {k} instances could be picked: every remaining "
                f"candidate is blocked by a per-level cap ({capped or 'none declared'})"
            )
        allowed = [i for i in open_picks if keeps_quotas_reachable(i)]
        if not allowed:
            # One-step lookahead cannot see joint tightness across several
            # quotas at once, so when it finds nothing safe, keep going with
            # whatever still advances a minimum.
            allowed = [i for i in open_picks if deficits_reduced(i) > 0]
        if not allowed:
            reason = _shortfall(quotas, candidates, counts, chosen_set, k - len(chosen))
            raise QuotaInfeasibleError(
                reason
                or f"no candidate advances the outstanding quotas after {len(chosen)} "
                f"pick(s): {outstanding()}"
            )

        # Coverage first, spread within it. While any minimum is outstanding the
        # primary key is how many deficits a pick advances; once they are all
        # met this is exactly pure max-min.
        #
        # The order matters and is not a detail. Chasing spread first and
        # repairing coverage at the end fails whenever the picks are already
        # nearly forced -- one instance per city over 102 cities with a budget of
        # 100 leaves the selector only *which* of a city's candidates to take,
        # and by the time the deficits are noticed the cities that could have
        # filled them are used up. Ties on deficit count are broken by max-min,
        # and early on most candidates advance the same number of quotas, so
        # spread still drives the choice.
        covering = any(quota.deficit(counts[quota.name]) for quota in quotas)
        nearest = distances[np.ix_(allowed, chosen)].min(axis=1)
        best = max(
            range(len(allowed)),
            key=lambda position: (
                deficits_reduced(allowed[position]) if covering else 0,
                nearest[position],
                _tie_break(candidates[allowed[position]]),
            ),
        )
        take(allowed[best])

    unmet = [quota for quota in quotas if quota.deficit(counts[quota.name])]
    if unmet:
        raise QuotaInfeasibleError(
            "the budget could not cover every minimum: "
            + ", ".join(f"{q.name}={q.deficit(counts[q.name])}" for q in unmet)
        )

    chosen = _improve_by_swapping(chosen, candidates, distances, quotas, swap_passes)
    return [candidates[index] for index in chosen]


def _quotas_hold(
    chosen: Sequence[int], candidates: Sequence[EvaluatedCandidate], quotas: Sequence[Quota]
) -> bool:
    for quota in quotas:
        counts = Counter(quota.key(candidates[index]) for index in chosen)
        if quota.deficit(counts):
            return False
        for level, count in counts.items():
            cap = quota.cap_for(level)
            if cap is not None and count > cap:
                return False
    return True


def _improve_by_swapping(
    chosen: list[int],
    candidates: Sequence[EvaluatedCandidate],
    distances: np.ndarray,
    quotas: Sequence[Quota],
    passes: int,
) -> list[int]:
    """Trade selected points for unselected ones while the spread improves.

    The greedy is myopic: an early pick made when few points were placed can end
    up crowding a later one. A swap only lands if it raises the spread score and
    leaves every quota satisfied, so this can never trade coverage for spread.
    """
    best = list(chosen)
    best_score = _spread_score(distances, best)
    for _ in range(max(0, passes)):
        improved = False
        outside = sorted(
            set(range(len(candidates))) - set(best),
            key=lambda i: _tie_break(candidates[i]),
        )
        for position in range(len(best)):
            for incoming in outside:
                trial = list(best)
                trial[position] = incoming
                if not _quotas_hold(trial, candidates, quotas):
                    continue
                score = _spread_score(distances, trial)
                if score > best_score:
                    best, best_score, improved = trial, score, True
                    break
        if not improved:
            break
    return best


def diversity_report(
    selected: Sequence[EvaluatedCandidate],
    pool: Sequence[EvaluatedCandidate],
    quotas: Sequence[Quota] = (),
) -> dict[str, Any]:
    """Coverage, spread and confounding of a selected set, as publishable data.

    ``feature_correlation`` is the part worth reading first. If two descriptors
    are strongly correlated *across the selected set*, the benchmark cannot
    separate their effects on a solver, however wide its spread looks.
    """
    if not selected:
        return {"selected": 0, "pool": len(pool)}

    # Descriptor units, not the selector's transformed space: this block is read
    # by people, and a route size reported as 1.097 when the instance holds three
    # customers per route is a misreport, not a rounding.
    selected_features = np.asarray(
        [descriptor_values(c.descriptors) for c in selected], dtype=float
    )
    pool_features = np.asarray(
        [descriptor_values(c.descriptors) for c in pool], dtype=float
    )

    features: dict[str, Any] = {}
    for position, name in enumerate(SELECTION_FEATURES):
        column, reference = selected_features[:, position], pool_features[:, position]
        pool_range = float(reference.max() - reference.min())
        selected_range = float(column.max() - column.min())
        features[name] = {
            "min": float(column.min()),
            "median": float(np.median(column)),
            "max": float(column.max()),
            "std": float(column.std()),
            # How much of what the pool offered on this axis the selection kept.
            "range_fraction_of_pool": selected_range / pool_range if pool_range > 0 else None,
        }

    # A feature that happens to be constant across the selection has no
    # correlation with anything; np.corrcoef would divide by its zero spread and
    # fill the row with NaN and a warning. Say "undefined" deliberately instead.
    varying = selected_features.std(axis=0) > 0.0
    correlation = np.full((len(SELECTION_FEATURES), len(SELECTION_FEATURES)), np.nan)
    if varying.sum() >= 2:
        block = np.corrcoef(selected_features[:, varying], rowvar=False)
        correlation[np.ix_(varying, varying)] = block
    worst_pair, worst_value = None, 0.0
    for i in range(len(SELECTION_FEATURES)):
        for j in range(i + 1, len(SELECTION_FEATURES)):
            value = correlation[i, j]
            if np.isfinite(value) and abs(value) > abs(worst_value):
                worst_pair = (SELECTION_FEATURES[i], SELECTION_FEATURES[j])
                worst_value = float(value)

    coverage = {
        quota.name: {
            str(level): count
            for level, count in sorted(
                Counter(quota.key(c) for c in selected).items(), key=lambda kv: str(kv[0])
            )
        }
        for quota in quotas
    }
    unmet = {
        quota.name: {str(level): short for level, short in quota.deficit(
            Counter(quota.key(c) for c in selected)
        ).items()}
        for quota in quotas
        if quota.deficit(Counter(quota.key(c) for c in selected))
    }

    standardized = _feature_matrix(list(pool))
    index_of = {id(c): i for i, c in enumerate(pool)}
    chosen = [index_of[id(c)] for c in selected if id(c) in index_of]
    spread = _spread_score(
        np.linalg.norm(standardized[:, None, :] - standardized[None, :, :], axis=2), chosen
    ) if len(chosen) == len(selected) else ()

    return {
        "selected": len(selected),
        "pool": len(pool),
        "coverage": coverage,
        "unmet_quotas": unmet,
        "features": features,
        "feature_correlation": {
            "names": list(SELECTION_FEATURES),
            # JSON has no NaN; an undefined correlation is null.
            "matrix": [
                [float(v) if np.isfinite(v) else None for v in row] for row in correlation
            ],
            "strongest_pair": list(worst_pair) if worst_pair else None,
            "strongest_value": worst_value,
        },
        "min_pairwise_distance": float(spread[0]) if spread else None,
    }
