"""
Map matching: group transient candidates from different maps of the same
observation (e.g. the optics tubes and bands observing simultaneously), so
that an event can be required to show up in more than one array before it
is kept as a transient candidate, and rank the events by how significant
their combined detections are.

Map matching needs every map's results at once, so the runner holds each
map's source-output inputs in memory (as a MapResult) until the matcher has
run, and only then writes the source outputs.
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import structlog
import uuid7
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from scipy.special import gammaln, ndtri_exp
from scipy.stats import chi2
from structlog.types import FilteringBoundLogger

from sotrplib.sifter.core import SifterResult
from sotrplib.sources.sources import MapMatch, MeasuredSource


@dataclass
class MapResult:
    """
    Everything one map's analysis hands to the source outputs, plus the map
    metadata the matcher needs. Deliberately excludes the map itself, so it
    stays small when sent back from a process-pool worker.
    """

    map_name: str
    mapcat_id: Any
    array: str | None
    frequency: str | None
    observation_start: Time | None
    observation_end: Time | None
    forced_photometry_candidates: list[MeasuredSource]
    sifter_result: SifterResult
    pointing_sources: list[MeasuredSource] = field(default_factory=list)
    injected_sources: list = field(default_factory=list)
    from_database: bool = False
    "True if the map came from mapcat, i.e. has a processing status to close."


@dataclass
class MapMatchGroup:
    """Transient candidates judged to be the same event, from one observation."""

    members: list[tuple[MapResult, MeasuredSource]]
    match: MapMatch
    "Also set as each member candidate's map_match."
    ra: u.Quantity
    dec: u.Quantity
    "SNR**2-weighted mean position of the members."

    def to_dict(self) -> dict:
        return {
            **self.match.model_dump(),
            "ra_deg": self.ra.to_value(u.deg),
            "dec_deg": self.dec.to_value(u.deg),
            "arrays": sorted({r.array for r, _ in self.members if r.array}),
            "bands": sorted({r.frequency for r, _ in self.members if r.frequency}),
            "members": [
                {
                    "map_name": result.map_name,
                    "mapcat_id": str(result.mapcat_id) if result.mapcat_id else None,
                    "array": result.array,
                    "frequency": result.frequency,
                    "snr": c.snr,
                    "flux_mJy": c.flux.to_value(u.mJy) if c.flux is not None else None,
                    "ra_deg": c.ra.to_value(u.deg),
                    "dec_deg": c.dec.to_value(u.deg),
                    "observation_mean_time": (
                        c.observation_mean_time.isot
                        if c.observation_mean_time is not None
                        else None
                    ),
                }
                for result, c in self.members
            ],
        }


def map_match_significance(snrs: list[float | None]) -> tuple[float, float]:
    """
    Combine one event's per-map detection SNRs into (combined_snr,
    significance). combined_snr = sqrt(sum snr_i**2). Under the null of
    independent Gaussian noise in each map, sum snr_i**2 is chi-squared with
    n_maps degrees of freedom; significance is its survival probability
    expressed as a one-sided Gaussian-equivalent sigma, so it grows both with
    each map's SNR and with the number of maps the event was detected in
    (e.g. one SNR-11 detection ~ 10.9 sigma, two SNR-5 detections ~ 6.7).
    Computed from the log probability, so it stays finite for bright events.
    """
    good = [s for s in snrs if s is not None and np.isfinite(s)]
    if not good:
        return 0.0, 0.0
    total = float(np.sum(np.square(good)))
    log_p = _chi2_logsf(total, df=len(good))
    return float(np.sqrt(total)), float(-ndtri_exp(log_p))


def _chi2_logsf(total: float, df: int) -> float:
    """
    log of the chi-squared survival function. scipy's logsf underflows to
    -inf for very large totals (bright events: sum snr**2 in the thousands),
    so fall back to the asymptotic expansion of the upper incomplete gamma
    function there: log Q(s, x) ~ (s-1) log x - x - lgamma(s) + log(1 + (s-1)/x),
    with s = df/2, x = total/2.
    """
    log_p = float(chi2.logsf(total, df=df))
    if np.isfinite(log_p):
        return log_p
    s, x = df / 2, total / 2
    return float((s - 1) * np.log(x) - x - gammaln(s) + np.log1p((s - 1) / x))


class MapMatcher(ABC):
    @abstractmethod
    def match(
        self, results: list[MapResult]
    ) -> tuple[list[MapResult], list[MapMatchGroup]]:
        """
        Group the transient candidates across results. Returns the (updated)
        results -- use these rather than the inputs -- and the groups.
        """
        return


class EmptyMapMatcher(MapMatcher):
    """No map matching: every transient candidate is kept as-is."""

    def match(
        self, results: list[MapResult]
    ) -> tuple[list[MapResult], list[MapMatchGroup]]:
        return results, []


class MultiArrayMapMatcher(MapMatcher):
    """
    Groups transient candidates within `radius` of each other in different
    maps whose observation time ranges overlap, and keeps a group's
    candidates as transient candidates only if they were detected by at
    least `min_arrays` distinct arrays (optics tubes). The same array in two
    bands counts once. Candidates in smaller groups (including events seen
    in a single map) are moved to the sifter result's
    unconfirmed_transient_candidates.

    Every transient candidate, confirmed or not, gets a map_match (see
    MapMatch) summarizing its group, including a significance combining the
    number of maps and each map's SNR (see map_match_significance) and a
    rank within the run: confirmed groups first, then by significance.

    If summary_directory is set, the ranked groups are also written there as
    JSON, one file per run.
    """

    def __init__(
        self,
        radius: u.Quantity = 1.5 * u.arcmin,
        min_arrays: int = 2,
        summary_directory: Path | None = None,
        log: FilteringBoundLogger | None = None,
    ):
        self.radius = radius
        self.min_arrays = min_arrays
        self.summary_directory = summary_directory
        self.log = log or structlog.get_logger()

    def match(
        self, results: list[MapResult]
    ) -> tuple[list[MapResult], list[MapMatchGroup]]:
        clusters: list[list[tuple[MapResult, MeasuredSource]]] = []
        for observation in self._group_by_observation(results):
            clusters.extend(self._cluster_observation(observation))

        groups = self._rank(clusters)

        confirmed = {
            id(candidate)
            for group in groups
            if group.match.confirmed
            for _, candidate in group.members
        }
        for result in results:
            sifted = result.sifter_result
            candidates = sifted.transient_candidates
            sifted.transient_candidates = [c for c in candidates if id(c) in confirmed]
            sifted.unconfirmed_transient_candidates = [
                c for c in candidates if id(c) not in confirmed
            ]

        for group in groups:
            if group.match.confirmed:
                self.log.info(
                    "map_matcher.confirmed_group",
                    rank=group.match.rank,
                    significance=round(group.match.significance, 2),
                    combined_snr=round(group.match.combined_snr, 2),
                    n_maps=group.match.n_maps,
                    arrays=sorted({r.array for r, _ in group.members if r.array}),
                    maps=[r.map_name for r, _ in group.members],
                    ra=group.ra.to_value(u.deg),
                    dec=group.dec.to_value(u.deg),
                    map_match_id=group.match.match_id,
                )
        self.log.info(
            "map_matcher.completed",
            n_maps=len(results),
            n_candidates=sum(len(g.members) for g in groups),
            n_groups=len(groups),
            n_confirmed_groups=sum(g.match.confirmed for g in groups),
            min_arrays=self.min_arrays,
        )

        if self.summary_directory is not None:
            self._write_summary(groups, results)

        return results, groups

    def _rank(
        self, clusters: list[list[tuple[MapResult, MeasuredSource]]]
    ) -> list[MapMatchGroup]:
        """Score each cluster, rank them, and set each member's map_match."""
        scored = []
        for members in clusters:
            arrays = {r.array for r, _ in members if r.array is not None}
            bands = {r.frequency for r, _ in members if r.frequency is not None}
            combined_snr, significance = map_match_significance(
                [c.snr for _, c in members]
            )
            scored.append(
                (
                    members,
                    len(arrays) >= self.min_arrays,
                    arrays,
                    bands,
                    combined_snr,
                    significance,
                )
            )
        scored.sort(key=lambda s: (not s[1], -s[5]))

        groups = []
        for rank, (
            members,
            confirmed,
            arrays,
            bands,
            combined_snr,
            significance,
        ) in enumerate(scored, start=1):
            match = MapMatch(
                match_id=str(uuid7.create()),
                confirmed=confirmed,
                n_maps=len({id(r) for r, _ in members}),
                n_arrays=len(arrays),
                n_bands=len(bands),
                combined_snr=combined_snr,
                significance=significance,
                rank=rank,
            )
            for _, candidate in members:
                candidate.map_match = match
            ra, dec = self._mean_position(members)
            groups.append(MapMatchGroup(members=members, match=match, ra=ra, dec=dec))
        return groups

    @staticmethod
    def _mean_position(
        members: list[tuple[MapResult, MeasuredSource]],
    ) -> tuple[u.Quantity, u.Quantity]:
        coords = SkyCoord(
            ra=u.Quantity([c.ra for _, c in members]),
            dec=u.Quantity([c.dec for _, c in members]),
        )
        weights = np.array([c.snr**2 if c.snr is not None else 0.0 for _, c in members])
        if not np.any(weights > 0):
            weights = np.ones(len(members))
        xyz = coords.cartesian.xyz.value @ weights
        mean = SkyCoord(x=xyz[0], y=xyz[1], z=xyz[2], representation_type="cartesian")
        mean = mean.represent_as("unitspherical")
        return mean.lon.to(u.deg), mean.lat.to(u.deg)

    def _write_summary(
        self, groups: list[MapMatchGroup], results: list[MapResult]
    ) -> Path:
        starts = [r.observation_start for r in results if r.observation_start]
        label = min(starts).isot[:19] if starts else Time.now().isot[:19]
        path = Path(self.summary_directory) / (
            f"map_match_summary_{label.replace('T', '-').replace(':', '-')}.json"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "min_arrays": self.min_arrays,
                    "radius_arcmin": self.radius.to_value(u.arcmin),
                    "maps": [r.map_name for r in results],
                    "groups": [g.to_dict() for g in groups],
                },
                indent=2,
            )
        )
        self.log.info("map_matcher.summary_written", path=str(path))
        return path

    @staticmethod
    def _group_by_observation(results: list[MapResult]) -> list[list[MapResult]]:
        """
        Split results into sets whose observation time ranges overlap
        (transitively). Results without a time range are each on their own.
        """
        timed, untimed = [], []
        for r in results:
            has_times = (
                r.observation_start is not None and r.observation_end is not None
            )
            (timed if has_times else untimed).append(r)

        observations: list[list[MapResult]] = []
        current_end = None
        for result in sorted(timed, key=lambda r: r.observation_start):
            if current_end is not None and result.observation_start <= current_end:
                observations[-1].append(result)
                current_end = max(current_end, result.observation_end)
            else:
                observations.append([result])
                current_end = result.observation_end

        return observations + [[r] for r in untimed]

    def _cluster_observation(
        self, observation: list[MapResult]
    ) -> list[list[tuple[MapResult, MeasuredSource]]]:
        """
        Link candidates within radius of each other in different maps of one
        observation (union-find, so chains across arrays/bands form one
        cluster). Every candidate ends up in exactly one cluster.
        """
        nodes = [
            (result, candidate)
            for result in observation
            for candidate in result.sifter_result.transient_candidates
        ]
        if not nodes:
            return []

        coords = SkyCoord(
            ra=u.Quantity([c.ra for _, c in nodes]),
            dec=u.Quantity([c.dec for _, c in nodes]),
        )
        idx1, idx2, _, _ = coords.search_around_sky(coords, self.radius)

        parent = list(range(len(nodes)))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i, j in zip(idx1, idx2):
            if i < j and nodes[i][0] is not nodes[j][0]:
                parent[find(i)] = find(j)

        members_by_root: dict[int, list[int]] = {}
        for i in range(len(nodes)):
            members_by_root.setdefault(find(i), []).append(i)

        return [[nodes[i] for i in indices] for indices in members_by_root.values()]
