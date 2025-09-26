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
from sotrplib.source_catalog.core import SourceCatalog
from sotrplib.sources.sources import CrossMatch, MeasuredSource


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
        log: FilteringBoundLogger | None = None,
    ):
        self.radius = radius
        self.method = method
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
                source.crossmatches = [
                    CrossMatch(source_id=m.source_id) for m in matches
                ]
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
        log: FilteringBoundLogger | None = None,
        debug: bool = False,
    ):
        self.radius_1Jy = radius_1Jy
        self.min_match_radius = min_match_radius
        self.ra_jitter = ra_jitter
        self.dec_jitter = dec_jitter
        self.cuts = cuts or {
            "fwhm_ra": [0.5 * u.arcmin, 5.0 * u.arcmin],
            "fwhm_dec": [0.5 * u.arcmin, 5.0 * u.arcmin],
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
        sources: list[MeasuredSource],
        catalogs: list[SourceCatalog],
        input_map: ProcessableMap,
    ) -> SifterResult:
        from .crossmatch import sift

        map_freq = input_map.frequency
        arr = input_map.array

        source_candidates, transient_candidates, noise_candidates = sift(
            extracted_sources=sources,
            catalog_sources=itertools.chain(
                *[c.sources_in_box(box=input_map.bbox) for c in catalogs]
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

        return SifterResult(
            source_candidates=source_candidates,
            transient_candidates=transient_candidates,
            noise_candidates=noise_candidates,
        )
