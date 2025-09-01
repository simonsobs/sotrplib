"""
The core dependency for the sifter
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

import astropy.units as u
import numpy as np
import structlog
from structlog.types import FilteringBoundLogger

from sotrplib.maps.core import ProcessableMap
from sotrplib.source_catalog.core import SourceCatalog
from sotrplib.sources.finding import BlindSourceCandidate
from sotrplib.sources.sources import ForcedPhotometrySource


@dataclass
class SifterResult:
    source_candidates: list[ForcedPhotometrySource]
    transient_candidates: list[ForcedPhotometrySource]
    noise_candidates: list[ForcedPhotometrySource]


class SiftingProvider(ABC):
    @abstractmethod
    def sift(
        self,
        sources: list[BlindSourceCandidate],
        input_map: ProcessableMap,
    ) -> SifterResult:
        raise NotImplementedError


class EmptySifter(SiftingProvider):
    def sift(
        self,
        sources: list[BlindSourceCandidate],
        input_map: ProcessableMap,
    ) -> SifterResult:
        return SifterResult(
            source_candidates=[], transient_candidates=[], noise_candidates=[]
        )


class SimpleCatalogSifter(SiftingProvider):
    catalog: SourceCatalog
    "The source catalog to sift against"
    radius: u.Quantity
    "Radius to search within"
    log: FilteringBoundLogger
    "Structlog logger"
    method: Literal["closest", "all"]
    "Return just the closest match or all of them"

    def __init__(
        self,
        catalog: SourceCatalog,
        radius: u.Quantity = u.Quantity(2.0, "arcmin"),
        method: Literal["closest", "all"] = "all",
        log: FilteringBoundLogger | None = None,
    ):
        self.catalog = catalog
        self.radius = radius
        self.method = method
        self.log = log or structlog.get_logger()

    def sift(
        self,
        sources: list[BlindSourceCandidate],
        input_map: ProcessableMap,
    ) -> SifterResult:
        source_candidates = []
        transient_candidates = []

        for i, source in enumerate(sources):
            log = self.log.bind(source=i, ra=source.ra, dec=source.dec)
            matches = self.catalog.crossmatch(
                ra=source.ra, dec=source.dec, radius=self.radius, method=self.method
            )

            if matches:
                source.update_crossmatches(
                    match_names=[m.sourceID for m in matches],
                    match_probabilities=[1.0 / len(matches)] * len(matches),
                )
                log = log.bind(number_of_matches=len(matches))
                log = log.info("sifter.simple.matched")
                source_candidates.append(source)
            else:
                log = log.info("sifter.simple.no_match")
                transient_candidates.append(source)

        result = SifterResult(
            source_candidates=source_candidates,
            transient_candidates=transient_candidates,
            noise_candidates=[],
        )

        return result


class DefaultSifter(SiftingProvider):
    catalog_sources: list
    "the list of sources to match against with this sifter"
    radius_1Jy: float
    "matching radius for a 1Jy source, arcmin"
    min_match_radius: float
    "minimum matching radius, i.e. for a zero flux source, arcmin"
    ra_jitter: float
    "jitter in the ra direction, in arcmin, to add to the uncertainty of the source position."
    dec_jitter: float
    "jitter in the dec direction, in arcmin, to add to the uncertainty of the source position."
    cuts: dict[str, list[float]]
    "a dictionary of cuts to apply to the source candidates, in the form of {cut_name: [min_value, max_value]}."
    crossmatch_with_gaia: bool
    "if True, will crossmatch with gaia catalog and add the gaia source name to the sourceID."
    crossmatch_with_million_quasar: bool
    "if True, will crossmatch with the million quasar catalog and add the source name to the sourceID."
    additional_catalogs: dict[str, Any]
    "a dictionary of additional catalogs to crossmatch with, in the form of {name:catalog}."
    log: FilteringBoundLogger
    "Structlog logger for logging information during the sifting process."
    debug: bool
    "If True, will log debug information."

    def __init__(
        self,
        catalog_sources: list,
        radius_1Jy: float = 30.0,
        min_match_radius: float = 1.5,
        ra_jitter: float = 0.0,
        dec_jitter: float = 0.0,
        cuts: dict[str, list[float]] | None = None,
        crossmatch_with_gaia: bool = True,
        crossmatch_with_million_quasar: bool = True,
        additional_catalogs: dict[str, Any] | None = None,
        log: FilteringBoundLogger | None = None,
        debug: bool = False,
    ):
        self.catalog_sources = catalog_sources
        self.radius_1Jy = radius_1Jy
        self.min_match_radius = min_match_radius
        self.ra_jitter = ra_jitter
        self.dec_jitter = dec_jitter
        self.cuts = cuts or {
            "fwhm": [0.5, 5.0],
            "snr": [5.0, np.inf],
        }
        self.crossmatch_with_gaia = crossmatch_with_gaia
        self.crossmatch_with_million_quasar = crossmatch_with_million_quasar
        self.additional_catalogs = additional_catalogs or dict()
        self.log = log or structlog.get_logger()
        self.debug = debug

        log = self.log.bind(**self.__dict__)
        log.info("sifter.defaultsifter.configured")

        return

    def sift(
        self,
        sources: list[BlindSourceCandidate],
        input_map: ProcessableMap,
    ) -> SifterResult:
        from .crossmatch import sift

        map_freq = input_map.frequency
        arr = input_map.array
        flux_map = input_map.flux
        # TODO: Map IDs
        map_id = ""

        source_candidates, transient_candidates, noise_candidates = sift(
            extracted_sources=sources,
            catalog_sources=self.catalog_sources,
            imap=flux_map,
            radius1Jy=self.radius_1Jy,
            min_match_radius=self.min_match_radius,
            ra_jitter=self.ra_jitter,
            dec_jitter=self.dec_jitter,
            crossmatch_with_gaia=self.crossmatch_with_gaia,
            crossmatch_with_million_quasar=self.crossmatch_with_million_quasar,
            additional_catalogs=self.additional_catalogs,
            map_freq=map_freq,
            map_id=map_id,
            arr=arr,
            cuts=self.cuts,
            log=self.log,
            debug=self.debug,
        )

        return SifterResult(
            source_candidates=source_candidates,
            transient_candidates=transient_candidates,
            noise_candidates=noise_candidates,
        )
