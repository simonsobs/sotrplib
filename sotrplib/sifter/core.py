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
from sotrplib.source_catalog.solar_system_object_catalog import SolarSystemObjectCatalog
from sotrplib.sources.sources import MeasuredSource


@dataclass
class SifterResult:
    """Container for the three output lists from a sifting pass."""

    source_candidates: list[MeasuredSource]
    transient_candidates: list[MeasuredSource]
    noise_candidates: list[MeasuredSource]


class SiftingProvider(ABC):
    """Abstract base for source-sifting providers.

    Subclasses implement ``sift`` to classify detections as known sources,
    transients, or noise.
    """

    @abstractmethod
    def sift(
        self,
        sources: list[MeasuredSource],
        catalogs: list[SourceCatalog | SolarSystemObjectCatalog],
        input_map: ProcessableMap,
    ) -> SifterResult:
        """Classify a list of source detections.

        Parameters
        ----------
        sources : list of MeasuredSource
            Extracted source detections to classify.
        catalogs : list of SourceCatalog or SolarSystemObjectCatalog
            Reference catalogs used for crossmatching.
        input_map : ProcessableMap
            The map from which the sources were extracted.

        Returns
        -------
        SifterResult
            Detections partitioned into source, transient, and noise lists.
        """
        raise NotImplementedError


class EmptySifter(SiftingProvider):
    """No-op sifter that returns empty candidate lists for every input."""

    def sift(
        self,
        sources: list[MeasuredSource],
        catalogs: list[SourceCatalog | SolarSystemObjectCatalog],
        input_map: ProcessableMap,
    ) -> SifterResult:
        return SifterResult(
            source_candidates=[], transient_candidates=[], noise_candidates=[]
        )


class SimpleCatalogSifter(SiftingProvider):
    """Sifter that classifies detections by fixed-radius catalog crossmatch.

    Parameters
    ----------
    radius : Quantity[arcmin], optional
        Search radius for catalog crossmatching (default 2 arcmin).
    method : {"all", "closest"}, optional
        Whether to return all matches or only the closest (default ``"all"``).
    log : FilteringBoundLogger, optional
        Structured logger.
    """

    radius: u.Quantity
    log: FilteringBoundLogger
    method: Literal["closest", "all"]

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
        catalogs: list[SourceCatalog | SolarSystemObjectCatalog],
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

        result = SifterResult(
            source_candidates=source_candidates,
            transient_candidates=transient_candidates,
            noise_candidates=[],
        )

        return result


class DefaultSifter(SiftingProvider):
    """Sifter that uses a flux-scaled matching radius and quality cuts.

    Parameters
    ----------
    radius_1Jy : Quantity[arcmin], optional
        Crossmatch radius for a 1 Jy source (default 30 arcmin).
    min_match_radius : Quantity[arcmin], optional
        Minimum crossmatch radius for faint sources (default 1.5 arcmin).
    ra_jitter : Quantity[arcmin], optional
        Additional RA position uncertainty added in quadrature (default 0 arcmin).
    dec_jitter : Quantity[arcmin], optional
        Additional Dec position uncertainty added in quadrature (default 0 arcmin).
    cuts : dict, optional
        Quality cuts as ``{field: [min, max]}``.  Defaults to FWHM 0.5–2.5×,
        SNR ≥ 5, and finite observation time.
    crossmatch_with_gaia : bool, optional
        Query Gaia for unmatched sources (default ``True``).
    crossmatch_with_million_quasar : bool, optional
        Crossmatch with the million-quasar catalog (default ``True``).
    additional_catalogs : dict, optional
        Extra named catalogs, e.g. ``{"million_quasar": mq_catalog}``.
    log : FilteringBoundLogger, optional
        Structured logger.
    debug : bool, optional
        Enable verbose debug logging (default ``False``).
    """

    radius_1Jy: AstroPydanticQuantity[u.Jy]
    min_match_radius: AstroPydanticQuantity[u.arcmin]
    ra_jitter: AstroPydanticQuantity[u.arcmin]
    dec_jitter: AstroPydanticQuantity[u.arcmin]
    cuts: dict[str, list[float]]
    crossmatch_with_gaia: bool
    crossmatch_with_million_quasar: bool
    additional_catalogs: dict[str, Any]
    log: FilteringBoundLogger
    debug: bool

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
            "fwhm": [0.5, 2.5],
            "snr": [5.0, np.inf],
            "observation_mean_time": [1, np.inf],
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
        catalogs: list[SourceCatalog | SolarSystemObjectCatalog],
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

        return SifterResult(
            source_candidates=source_candidates,
            transient_candidates=transient_candidates,
            noise_candidates=noise_candidates,
        )
