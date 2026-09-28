"""
Map matching: group transient candidates from different maps of the same
observation (e.g. the optics tubes and bands observing simultaneously), so
that an event can be required to show up in more than one array before it
is kept as a transient candidate.

Map matching needs every map's results at once, so the runner holds each
map's source-output inputs in memory (as a MapResult) until the matcher has
run, and only then writes the source outputs.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import structlog
import uuid7
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from structlog.types import FilteringBoundLogger

from sotrplib.sifter.core import SifterResult
from sotrplib.sources.sources import MeasuredSource


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

    map_match_id: str
    members: list[tuple[str, MeasuredSource]]
    "(map_name, candidate) pairs."
    arrays: set[str]
    confirmed: bool


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

    Every transient candidate, confirmed or not, gets a map_match_id shared
    with the rest of its group.
    """

    def __init__(
        self,
        radius: u.Quantity = 1.5 * u.arcmin,
        min_arrays: int = 2,
        log: FilteringBoundLogger | None = None,
    ):
        self.radius = radius
        self.min_arrays = min_arrays
        self.log = log or structlog.get_logger()

    def match(
        self, results: list[MapResult]
    ) -> tuple[list[MapResult], list[MapMatchGroup]]:
        groups: list[MapMatchGroup] = []
        for observation in self._group_by_observation(results):
            groups.extend(self._match_observation(observation))

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
            sifted.unconfirmed_transient_candidates = [
                c for c in candidates if id(c) not in confirmed
            ]

        for group in groups:
            if group.confirmed:
                self.log.info(
                    "map_matcher.confirmed_group",
                    map_match_id=group.map_match_id,
                    arrays=sorted(group.arrays),
                    maps=[name for name, _ in group.members],
                    ra=group.members[0][1].ra.to_value("deg"),
                    dec=group.members[0][1].dec.to_value("deg"),
                )
        self.log.info(
            "map_matcher.completed",
            n_maps=len(results),
            n_candidates=sum(len(g.members) for g in groups),
            n_groups=len(groups),
            n_confirmed_groups=sum(g.confirmed for g in groups),
            min_arrays=self.min_arrays,
        )
        return results, groups

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

    def _match_observation(self, observation: list[MapResult]) -> list[MapMatchGroup]:
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

        # union-find over candidates, linking only detections in different maps
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

        groups = []
        for indices in members_by_root.values():
            map_match_id = str(uuid7.create())
            arrays = {
                nodes[i][0].array for i in indices if nodes[i][0].array is not None
            }
            for i in indices:
                nodes[i][1].map_match_id = map_match_id
            groups.append(
                MapMatchGroup(
                    map_match_id=map_match_id,
                    members=[(nodes[i][0].map_name, nodes[i][1]) for i in indices],
                    arrays=arrays,
                    confirmed=len(arrays) >= self.min_arrays,
                )
            )
        return groups
