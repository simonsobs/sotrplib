"""
The core dependency for the sifter
"""

import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

import astropy.units as u
import numpy as np
import structlog
from astropydantic import AstroPydanticQuantity
from structlog.types import FilteringBoundLogger

from sotrplib.maps.core import ProcessableMap
from sotrplib.sifter.transient_crossmatch import TransientCrossmatcher
from sotrplib.source_catalog.core import SourceCatalog
from sotrplib.sources.sources import MeasuredSource


@dataclass
class SifterResult:
    source_candidates: list[MeasuredSource]
    transient_candidates: list[MeasuredSource]
    noise_candidates: list[MeasuredSource]


class SiftingProvider(ABC):
    @abstractmethod
    def sift(
        self,
        sources: list[MeasuredSource],
        catalogs: list[SourceCatalog],
        input_map: ProcessableMap,
    ) -> SifterResult:
        raise NotImplementedError


class EmptySifter(SiftingProvider):
    def sift(
        self,
        sources: list[MeasuredSource],
        catalogs: list[SourceCatalog],
        input_map: ProcessableMap,
    ) -> SifterResult:
        return SifterResult(
            source_candidates=[], transient_candidates=[], noise_candidates=[]
        )


class SimpleCatalogSifter(SiftingProvider):
    radius: u.Quantity
    "Radius to search within"
    log: FilteringBoundLogger
    "Structlog logger"
    method: Literal["closest", "all"]
    "Return just the closest match or all of them"

    def __init__(
        self,
        radius: u.Quantity = u.Quantity(2.0, "arcmin"),
        method: Literal["closest", "all"] = "all",
        transient_crossmatcher: TransientCrossmatcher | None = None,
        log: FilteringBoundLogger | None = None,
    ):
        self.radius = radius
        self.method = method
        self.transient_crossmatcher = transient_crossmatcher
        self.log = log or structlog.get_logger()

    def sift(
        self,
        sources: list[MeasuredSource],
        catalogs: list[SourceCatalog],
        input_map: ProcessableMap,
    ) -> SifterResult:
        source_candidates = []
        transient_candidates = []

        for i, source in enumerate(sources):
            log = self.log.bind(source=i, ra=source.ra, dec=source.dec)

            matches = list(
                itertools.chain(
                    *[
                        c.crossmatch(
                            ra=source.ra,
                            dec=source.dec,
                            radius=self.radius,
                            method=self.method,
                        )
                        for c in catalogs
                    ]
                )
            )

            if matches:
                source.crossmatches = matches
                log = log.bind(number_of_matches=len(matches))
                log = log.info("sifter.simple.matched")
                source_candidates.append(source)
            else:
                log = log.info("sifter.simple.no_match")
                transient_candidates.append(source)

        if self.transient_crossmatcher is not None:
            transient_candidates = self.transient_crossmatcher.crossmatch(
                transient_candidates
            )

        result = SifterResult(
            source_candidates=source_candidates,
            transient_candidates=transient_candidates,
            noise_candidates=[],
        )

        return result


class DefaultSifter(SiftingProvider):
    radius_1Jy: AstroPydanticQuantity[u.Jy]
    "matching radius for a 1Jy source, arcmin"
    min_match_radius: AstroPydanticQuantity[u.arcmin]
    "minimum matching radius, i.e. for a zero flux source, arcmin"
    ra_jitter: AstroPydanticQuantity[u.arcmin]
    "jitter in the ra direction, in arcmin, to add to the uncertainty of the source position."
    dec_jitter: AstroPydanticQuantity[u.arcmin]
    "jitter in the dec direction, in arcmin, to add to the uncertainty of the source position."
    cuts: dict[str, list[float]]
    "a dictionary of cuts to apply to the source candidates, in the form of {cut_name: [min_value, max_value]}."
    crossmatch_with_gaia: bool
    "if True, will crossmatch with gaia catalog and add the gaia source name to the source_id."
    crossmatch_with_million_quasar: bool
    "if True, will crossmatch with the million quasar catalog and add the source name to the source_id."
    additional_catalogs: dict[str, Any]
    "a dictionary of additional catalogs to crossmatch with, in the form of {name:catalog}."
    transient_crossmatcher: TransientCrossmatcher | None
    "if set, crossmatch transient candidates with external catalogs and rank the matches."
    crossmatch_sso_trajectories: bool
    "if True, additionally check noise/transient candidates against solar-system-object "
    "trajectories sampled across the map's full observation window (via any catalog in "
    "`catalogs` that implements get_sso_generators_in_map, e.g. SOCat) rather than only "
    "each object's position at a single map-mean time. Matters for week coadds, where an "
    "SSO can move well outside a single-snapshot match radius over the observation window; "
    "for depth-1 maps the static per-map crossmatch already covers this."
    sso_crossmatch_radius: AstroPydanticQuantity[u.arcmin]
    "match radius for the SSO trajectory crossmatch."
    sso_crossmatch_cadence: AstroPydanticQuantity[u.hour]
    "sampling cadence for the SSO trajectory crossmatch -- see sotrplib.sifter.sso_crossmatch."
    log: FilteringBoundLogger
    "Structlog logger for logging information during the sifting process."
    debug: bool
    "If True, will log debug information."

    def __init__(
        self,
        radius_1Jy: u.Quantity = u.Quantity(30.0, "arcmin"),
        min_match_radius: u.Quantity = u.Quantity(1.5, "arcmin"),
        ra_jitter: u.Quantity = u.Quantity(0.0, "arcmin"),
        dec_jitter: u.Quantity = u.Quantity(0.0, "arcmin"),
        cuts: dict[str, list[float]] | None = None,
        crossmatch_with_gaia: bool = True,
        crossmatch_with_million_quasar: bool = True,
        additional_catalogs: dict[str, Any] | None = None,
        transient_crossmatcher: TransientCrossmatcher | None = None,
        crossmatch_sso_trajectories: bool = False,
        sso_crossmatch_radius: u.Quantity = u.Quantity(5.0, "arcmin"),
        sso_crossmatch_cadence: u.Quantity = u.Quantity(2.0, "hour"),
        log: FilteringBoundLogger | None = None,
        debug: bool = False,
    ):
        self.radius_1Jy = radius_1Jy
        self.min_match_radius = min_match_radius
        self.ra_jitter = ra_jitter
        self.dec_jitter = dec_jitter
        self.cuts = cuts or {
            "fwhm": [0.5, 2.5],
            "snr": [5.0, np.inf],
            "observation_mean_time": [1, np.inf],
        }
        self.crossmatch_with_gaia = crossmatch_with_gaia
        self.crossmatch_with_million_quasar = crossmatch_with_million_quasar
        self.additional_catalogs = additional_catalogs or dict()
        self.transient_crossmatcher = transient_crossmatcher
        self.crossmatch_sso_trajectories = crossmatch_sso_trajectories
        self.sso_crossmatch_radius = sso_crossmatch_radius
        self.sso_crossmatch_cadence = sso_crossmatch_cadence
        self.log = log or structlog.get_logger()
        self.debug = debug

        log = self.log.bind(**self.__dict__)
        log.info("sifter.defaultsifter.configured")

        return

    def sift(
        self,
        sources: list[MeasuredSource],
        catalogs: list[SourceCatalog],
        input_map: ProcessableMap,
    ) -> SifterResult:
        from .crossmatch import sift

        map_freq = input_map.frequency
        arr = input_map.array
        source_candidates, transient_candidates, noise_candidates = sift(
            extracted_sources=sources,
            catalog_sources=list(
                itertools.chain(
                    *[c.get_sources_in_map(input_map=input_map) for c in catalogs]
                )
            ),
            input_map=input_map,
            radius1Jy=self.radius_1Jy,
            min_match_radius=self.min_match_radius,
            ra_jitter=self.ra_jitter,
            dec_jitter=self.dec_jitter,
            crossmatch_with_gaia=self.crossmatch_with_gaia,
            crossmatch_with_million_quasar=self.crossmatch_with_million_quasar,
            additional_catalogs=self.additional_catalogs,
            map_freq=map_freq,
            arr=arr,
            cuts=self.cuts,
            log=self.log,
            debug=self.debug,
        )

        if self.crossmatch_sso_trajectories:
            noise_candidates, transient_candidates, sso_candidates = (
                self._crossmatch_sso_trajectories(
                    noise_candidates=noise_candidates,
                    transient_candidates=transient_candidates,
                    catalogs=catalogs,
                    input_map=input_map,
                )
            )
            source_candidates = source_candidates + sso_candidates

        if self.transient_crossmatcher is not None:
            transient_candidates = self.transient_crossmatcher.crossmatch(
                transient_candidates
            )

        return SifterResult(
            source_candidates=source_candidates,
            transient_candidates=transient_candidates,
            noise_candidates=noise_candidates,
        )

    def _crossmatch_sso_trajectories(
        self,
        noise_candidates: list[MeasuredSource],
        transient_candidates: list[MeasuredSource],
        catalogs: list[SourceCatalog],
        input_map: ProcessableMap,
    ) -> tuple[list[MeasuredSource], list[MeasuredSource], list[MeasuredSource]]:
        """
        Check noise/transient candidates against solar-system-object
        trajectories from any catalog in `catalogs` that implements
        get_sso_generators_in_map (e.g. SOCat) -- unlike the static
        catalog_sources crossmatch above (one position per object, at the
        map's mean time), this samples each object's trajectory across
        the map's full observation window, which matters for week coadds
        where an object can move well outside a single-snapshot match
        radius. Run before transient_crossmatcher so candidates already
        explained by a known SSO don't also spend external-catalog
        crossmatch calls (e.g. Gaia).

        Returns (remaining_noise, remaining_transient, sso_matched), with
        matched candidates removed from their original bucket.
        """
        from astropy.time import Time

        from sotrplib.sifter.sso_crossmatch import crossmatch_sso_trajectories

        if input_map.observation_start is None or input_map.observation_end is None:
            self.log.warning(
                "sifter.defaultsifter.sso_crossmatch.no_observation_time",
                map_id=input_map.map_id,
            )
            return noise_candidates, transient_candidates, []

        sso_generators = {}
        for catalog in catalogs:
            get_generators = getattr(catalog, "get_sso_generators_in_map", None)
            if get_generators is None:
                continue
            sso_generators.update(get_generators(input_map))

        if not sso_generators:
            return noise_candidates, transient_candidates, []

        candidates = noise_candidates + transient_candidates
        crossmatch_sso_trajectories(
            candidates=candidates,
            sso_generators=sso_generators,
            t_min=Time(input_map.observation_start),
            t_max=Time(input_map.observation_end),
            radius=self.sso_crossmatch_radius,
            cadence=self.sso_crossmatch_cadence,
            log=self.log,
        )

        matched_ids = {id(c) for c in candidates if c.source_type == "sso"}
        remaining_noise = [c for c in noise_candidates if id(c) not in matched_ids]
        remaining_transient = [
            c for c in transient_candidates if id(c) not in matched_ids
        ]
        sso_matched = [c for c in candidates if id(c) in matched_ids]

        return remaining_noise, remaining_transient, sso_matched
