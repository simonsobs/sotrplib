"""
Crossmatch transient candidates (blind detections with no SO catalog
counterpart) against external catalogs, and rank the matches by how
likely each is to be the true counterpart.

Each external catalog supplies matches with a chance-coincidence
probability P_chance (see `source_catalog.external_catalogs`). Matches
are then scored as

    match_score = type_prior * (1 - P_chance)

so that brightness and rarity enter through P_chance (a bright nearby
galaxy is far rarer than a 20th-mag star, so a close match to one is far
less likely to be a coincidence) while the type prior encodes which
object classes are more plausible counterparts (a nearby galaxy takes
precedence over a background quasar of comparable chance probability).
"""

import numpy as np
import structlog
from astropy import units as u
from structlog.types import FilteringBoundLogger

from sotrplib.source_catalog.external_catalogs import ExternalCatalog
from sotrplib.sources.sources import CrossMatch, MeasuredSource

DEFAULT_TYPE_PRIORS: dict[str, float] = {
    "galaxy": 1.0,
    "star": 0.7,
    "agn": 0.5,
    "quasar": 0.4,
    "sso": 1.0,
    "unknown": 0.2,
}

## coarse external-catalog object classes -> RegisteredSource.source_type
SOURCE_TYPE_MAP: dict[str, str] = {
    "galaxy": "extragalactic",
    "quasar": "extragalactic",
    "agn": "extragalactic",
    "star": "star",
    "sso": "sso",
}


def rank_crossmatches(
    matches: list[CrossMatch],
    type_priors: dict[str, float] | None = None,
) -> list[CrossMatch]:
    """
    Score and sort crossmatches, most likely counterpart first.

    Sets `probability` (association probability, 1 - P_chance) and
    `match_score` (prior-weighted score used for the ordering) on each
    match. Matches without a chance probability get a neutral 0.5.
    """
    priors = type_priors or DEFAULT_TYPE_PRIORS
    unknown_prior = priors.get("unknown", 0.2)
    for match in matches:
        prior = priors.get(match.source_type or "unknown", unknown_prior)
        association = (
            1.0 - match.chance_probability
            if match.chance_probability is not None
            else 0.5
        )
        match.probability = association
        match.match_score = prior * association
    return sorted(matches, key=lambda m: m.match_score, reverse=True)


class TransientCrossmatcher:
    """
    Query a set of external catalogs around each transient candidate and
    attach a prioritized list of possible counterparts.

    The match radius per candidate is the larger of `match_radius` and
    `position_error_scale` times the candidate's positional uncertainty.
    Matches with match_score below `min_score` are dropped. If
    `set_source_type` is True, the candidate's source_type is set from
    the best match (e.g. galaxy/quasar -> "extragalactic").
    """

    def __init__(
        self,
        catalogs: list[ExternalCatalog],
        match_radius: u.Quantity = 2.0 * u.arcmin,
        position_error_scale: float = 3.0,
        type_priors: dict[str, float] | None = None,
        min_score: float = 0.0,
        set_source_type: bool = True,
        log: FilteringBoundLogger | None = None,
    ):
        self.catalogs = catalogs
        self.match_radius = match_radius
        self.position_error_scale = position_error_scale
        self.type_priors = type_priors or DEFAULT_TYPE_PRIORS
        self.min_score = min_score
        self.set_source_type = set_source_type
        self.log = log or structlog.get_logger()

    def _candidate_match_radius(self, candidate: MeasuredSource) -> u.Quantity:
        if candidate.err_ra is None or candidate.err_dec is None:
            return self.match_radius
        position_error = np.sqrt(candidate.err_ra**2 + candidate.err_dec**2)
        return max(self.match_radius, self.position_error_scale * position_error)

    def crossmatch_candidate(self, candidate: MeasuredSource) -> list[CrossMatch]:
        """
        Return the ranked list of external-catalog matches for one candidate.
        """
        radius = self._candidate_match_radius(candidate)
        matches = []
        for catalog in self.catalogs:
            matches.extend(
                catalog.cone_search(ra=candidate.ra, dec=candidate.dec, radius=radius)
            )
        matches = rank_crossmatches(matches, self.type_priors)
        return [m for m in matches if m.match_score >= self.min_score]

    def crossmatch(self, candidates: list[MeasuredSource]) -> list[MeasuredSource]:
        """
        Attach ranked external crossmatches to each transient candidate.
        The first entry in `crossmatches` is the most likely counterpart.
        """
        log = self.log.bind(
            func_name="transient_crossmatcher.crossmatch",
            n_candidates=len(candidates),
            catalogs=[c.catalog_name for c in self.catalogs],
        )
        n_matched = 0
        for candidate in candidates:
            matches = self.crossmatch_candidate(candidate)
            if candidate.crossmatches is None:
                candidate.crossmatches = []
            candidate.crossmatches.extend(matches)
            if not matches:
                continue
            n_matched += 1
            best = matches[0]
            if self.set_source_type and best.source_type in SOURCE_TYPE_MAP:
                candidate.source_type = SOURCE_TYPE_MAP[best.source_type]
            log.info(
                "transient_crossmatcher.best_match",
                source_id=candidate.source_id,
                match_id=best.source_id,
                match_type=best.source_type,
                match_score=best.match_score,
                chance_probability=best.chance_probability,
                separation=best.angular_separation,
                catalog=best.catalog_name,
                n_matches=len(matches),
            )
        log.info(
            "transient_crossmatcher.complete",
            n_matched=n_matched,
            n_unmatched=len(candidates) - n_matched,
        )
        return candidates
