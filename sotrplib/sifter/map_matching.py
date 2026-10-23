"""
Map matching: group the transient candidates of all the maps of a run (e.g.
the optics tubes and bands that observe at the same time), so that an event
must be detected in more than one array to stay a transient candidate, and
rank the groups by the significance of their combined detections.

The map matcher labels each grouped candidate with the group_id and
group_rank of its group. Candidates that are not in a confirmed group move
to the sifter result's noise_candidates and keep their labels.

The map matcher compares all the maps of the run. It does not apply a time
cut. Thus, the maps that go into the run set the time range of the matching.

Map matching needs the results of all the maps at once. Thus, the runner
keeps the source-output inputs of each map in memory (as a MapResult) until
the map matcher has run, and only then writes the source outputs.

The map matcher does not write files. It returns a MapMatchSummary, and the
map-match outputs (sotrplib.outputs.core.MapMatchOutput) write it.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

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
from sotrplib.sources.sources import MeasuredSource


@dataclass
class MapResult:
    """
    Everything one map's analysis hands to the source outputs, plus the map
    metadata the matcher needs. Deliberately excludes the map itself, so it
    stays small when sent back from a process-pool worker. One MapResult is
    one analyzed map: the matcher never links two candidates of one result.
    """

    map_name: str
    mapcat_id: Any
    array: str | None
    frequency: str | None
    observation_start: Time | None
    "Used only to name the map-match output files."
    forced_photometry_candidates: list[MeasuredSource]
    sifter_result: SifterResult
    pointing_sources: list[MeasuredSource] = field(default_factory=list)
    injected_sources: list = field(default_factory=list)
    from_database: bool = False
    "True if the map came from mapcat, i.e. has a processing status to close."
    map_type: str = "depth1_map"
    "The map's mapcat map_type, for its processing status (depth-1 map or coadd)."


@dataclass
class MapMatchGroup:
    """
    Transient candidates that the map matcher judged to be the same event.
    Each member candidate has group_id and group_rank set to the values of
    its group.
    """

    group_id: UUID
    rank: int
    "1 = most significant group of the run; confirmed groups rank first, then "
    "the other notable groups (a detection at or above high_sig, or at least "
    "min_arrays arrays), then the rest."
    confirmed: bool
    "Detected by at least the matcher's min_arrays distinct arrays, with at "
    "least one detection at or above the matcher's high_sig."
    notable: bool
    "Confirmed, or one detection at or above high_sig, or at least min_arrays "
    "arrays."
    max_snr: float | None
    "Highest SNR of the detections in the group."
    n_maps: int
    n_arrays: int
    n_bands: int
    combined_snr: float
    "sqrt of the sum of squared per-map SNRs."
    significance: float
    "Gaussian-equivalent significance of combined_snr**2 under the chi-squared "
    "distribution with n_maps degrees of freedom (i.e. of independent noise "
    "in every map reaching these SNRs)."
    ra: u.Quantity
    dec: u.Quantity
    "SNR**2-weighted mean position of the members."
    members: list[tuple[MapResult, MeasuredSource]]

    def to_dict(self) -> dict:
        return {
            "group_id": str(self.group_id),
            "rank": self.rank,
            "confirmed": self.confirmed,
            "notable": self.notable,
            "max_snr": self.max_snr,
            "n_maps": self.n_maps,
            "n_arrays": self.n_arrays,
            "n_bands": self.n_bands,
            "combined_snr": self.combined_snr,
            "significance": self.significance,
            "ra_deg": self.ra.to_value(u.deg),
            "dec_deg": self.dec.to_value(u.deg),
            "arrays": sorted({r.array for r, _ in self.members if r.array}),
            "bands": sorted({r.frequency for r, _ in self.members if r.frequency}),
            "members": [
                {
                    "measurement_id": str(c.measurement_id),
                    "map_name": c.map_name or result.map_name,
                    "map_id": _str_or_none(c.map_id or result.mapcat_id),
                    "array": result.array,
                    "frequency": result.frequency,
                    "snr": float(c.snr) if c.snr is not None else None,
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


def _str_or_none(x: Any) -> str | None:
    return str(x) if x is not None else None


@dataclass
class MapMatchSummary:
    """
    The result of one map-matching run: all the groups, in the sequence of
    their rank, and the settings of the matcher. The map-match outputs write
    it.
    """

    groups: list[MapMatchGroup] = field(default_factory=list)
    map_names: list[str] = field(default_factory=list)
    observation_start: Time | None = None
    "The earliest start time of the maps of the run."
    settings: dict = field(default_factory=dict)
    "The settings of the matcher, e.g. min_arrays, high_sig, low_sig and "
    "radius_arcmin."

    @property
    def notable_groups(self) -> list[MapMatchGroup]:
        return [g for g in self.groups if g.notable]

    def to_dict(self, include_all_groups: bool = False) -> dict:
        """
        Return the summary as a JSON-compatible dict. If include_all_groups
        is False, the dict lists only the notable groups, and counts the
        other groups in n_groups_not_listed.
        """
        listed = self.groups if include_all_groups else self.notable_groups
        return {
            **self.settings,
            "maps": list(self.map_names),
            "n_groups": len(self.groups),
            "n_groups_not_listed": len(self.groups) - len(listed),
            "groups": [g.to_dict() for g in listed],
        }

    @classmethod
    def from_results(
        cls,
        results: list[MapResult],
        groups: list[MapMatchGroup] | None = None,
        settings: dict | None = None,
    ) -> "MapMatchSummary":
        starts = [r.observation_start for r in results if r.observation_start]
        return cls(
            groups=groups or [],
            map_names=[r.map_name for r in results],
            observation_start=min(starts) if starts else None,
            settings=settings or {},
        )


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
    ) -> tuple[list[MapResult], MapMatchSummary]:
        """
        Group the transient candidates across results. Returns the (updated)
        results -- use these rather than the inputs -- and a summary that
        contains the groups.
        """
        return


class EmptyMapMatcher(MapMatcher):
    """No map matching: every transient candidate is kept as-is."""

    def match(
        self, results: list[MapResult]
    ) -> tuple[list[MapResult], MapMatchSummary]:
        return results, MapMatchSummary.from_results(results)


class MultiArrayMapMatcher(MapMatcher):
    """
    Groups the transient candidates of all the results that are within
    `radius` of each other in different maps. A group is confirmed if at
    least `min_arrays` distinct arrays (optics tubes) detected it and at
    least one detection has SNR >= `high_sig`. The same array in two bands
    counts once.

    Only candidates with SNR >= `low_sig` are grouped. To use the low_sig
    detections, the blind search threshold and the sifter's snr cut must be
    at or below low_sig (e.g. 3), so that the sifter keeps them as transient
    candidates.

    Every grouped candidate, confirmed or not, gets the group_id and
    group_rank of its group. The rank is within the run: confirmed groups
    first, then the other notable groups (a detection >= high_sig, or
    >= min_arrays arrays), then the rest, each by significance (see
    map_match_significance).

    The candidates of confirmed groups stay in transient_candidates. All the
    other transient candidates move to noise_candidates: those of groups that
    are not confirmed keep their group_id, and those below low_sig have no
    group_id.

    The matcher compares all the results that it gets, with no time cut.
    Thus, give it only the maps that must be compared, for example the maps
    of one observation.

    The matcher does not write files. To write the groups, add a map-match
    output (e.g. JSONMapMatchOutput) to the runner.
    """

    def __init__(
        self,
        radius: u.Quantity = 1.5 * u.arcmin,
        min_arrays: int = 2,
        high_sig: float = 5.0,
        low_sig: float = 3.0,
        log: FilteringBoundLogger | None = None,
    ):
        if low_sig > high_sig:
            raise ValueError(f"low_sig ({low_sig}) must be <= high_sig ({high_sig})")
        self.radius = radius
        self.min_arrays = min_arrays
        self.high_sig = high_sig
        self.low_sig = low_sig
        self.log = log or structlog.get_logger()

    @property
    def settings(self) -> dict:
        return {
            "min_arrays": self.min_arrays,
            "high_sig": self.high_sig,
            "low_sig": self.low_sig,
            "radius_arcmin": self.radius.to_value(u.arcmin),
        }

    def match(
        self, results: list[MapResult]
    ) -> tuple[list[MapResult], MapMatchSummary]:
        groups = self._rank(self._cluster(results))

        confirmed = {
            id(candidate)
            for group in groups
            if group.confirmed
            for _, candidate in group.members
        }
        for result in results:
            sifted = result.sifter_result
            candidates = sifted.transient_candidates
            sifted.transient_candidates = [c for c in candidates if id(c) in confirmed]
            sifted.noise_candidates = list(sifted.noise_candidates) + [
                c for c in candidates if id(c) not in confirmed
            ]

        for group in groups:
            if group.confirmed:
                self.log.info(
                    "map_matcher.confirmed_group",
                    rank=group.rank,
                    significance=round(group.significance, 2),
                    combined_snr=round(group.combined_snr, 2),
                    n_maps=group.n_maps,
                    arrays=sorted({r.array for r, _ in group.members if r.array}),
                    maps=[r.map_name for r, _ in group.members],
                    ra=group.ra.to_value(u.deg),
                    dec=group.dec.to_value(u.deg),
                    group_id=str(group.group_id),
                )
        self.log.info(
            "map_matcher.completed",
            n_maps=len(results),
            n_candidates=sum(len(g.members) for g in groups),
            n_groups=len(groups),
            n_confirmed_groups=sum(g.confirmed for g in groups),
            n_notable_groups=sum(g.notable for g in groups),
            min_arrays=self.min_arrays,
            high_sig=self.high_sig,
            low_sig=self.low_sig,
        )

        return results, MapMatchSummary.from_results(
            results, groups=groups, settings=self.settings
        )

    def _rank(
        self, clusters: list[list[tuple[MapResult, MeasuredSource]]]
    ) -> list[MapMatchGroup]:
        """Score each cluster, rank them, and label each member."""
        scored = []
        for members in clusters:
            arrays = {r.array for r, _ in members if r.array is not None}
            bands = {r.frequency for r, _ in members if r.frequency is not None}
            snrs = [c.snr for _, c in members if c.snr is not None]
            max_snr = float(max(snrs)) if snrs else None
            has_seed = bool(max_snr is not None and max_snr >= self.high_sig)
            multi_array = len(arrays) >= self.min_arrays
            confirmed = bool(multi_array and has_seed)
            # 0: confirmed, 1: other notable groups, 2: the rest.
            tier = 0 if confirmed else 1 if (has_seed or multi_array) else 2
            notable = tier <= 1
            combined_snr, significance = map_match_significance(snrs)
            scored.append(
                dict(
                    members=members,
                    tier=tier,
                    confirmed=confirmed,
                    notable=notable,
                    max_snr=max_snr,
                    n_maps=len({id(r) for r, _ in members}),
                    n_arrays=len(arrays),
                    n_bands=len(bands),
                    combined_snr=combined_snr,
                    significance=significance,
                )
            )
        scored.sort(key=lambda s: (s["tier"], -s["significance"]))

        groups = []
        for rank, s in enumerate(scored, start=1):
            members = s.pop("members")
            s.pop("tier")
            ra, dec = self._mean_position(members)
            group = MapMatchGroup(
                group_id=uuid7.create(),
                rank=rank,
                ra=ra,
                dec=dec,
                members=members,
                **s,
            )
            for _, candidate in members:
                candidate.group_id = group.group_id
                candidate.group_rank = rank
            groups.append(group)
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

    def _cluster(
        self, results: list[MapResult]
    ) -> list[list[tuple[MapResult, MeasuredSource]]]:
        """
        Link candidates within radius of each other in different maps
        (union-find, so chains across arrays/bands form one cluster). Every
        candidate with SNR >= low_sig ends up in exactly one cluster.
        """
        nodes = [
            (result, candidate)
            for result in results
            for candidate in result.sifter_result.transient_candidates
            if candidate.snr is not None and candidate.snr >= self.low_sig
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
