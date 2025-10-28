from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

from astropy import units as u
from astropydantic import AstroPydanticQuantity
from pydantic import BaseModel
from structlog.types import FilteringBoundLogger

from sotrplib.maps.preprocessor import (
    KappaRhoCleaner,
    MapPreprocessor,
    MapShifter,
    MatchedFilter,
)


class PreprocessorConfig(BaseModel, ABC):
    preprocessor_type: str

    @abstractmethod
    def to_preprocessor(
        self, log: FilteringBoundLogger | None = None
    ) -> MapPreprocessor:
        return


class KappaRhoCleanerConfig(PreprocessorConfig):
    preprocessor_type: Literal["kappa_rho"] = "kappa_rho"
    cut_on: Literal["median", "percentile", "max"] = "median"
    fraction: float = 0.05

    def to_preprocessor(
        self, log: FilteringBoundLogger | None = None
    ) -> KappaRhoCleaner:
        return KappaRhoCleaner(cut_on=self.cut_on, fraction=self.fraction)


class MatchedFilterConfig(PreprocessorConfig):
    preprocessor_type: Literal["matched_filter"] = "matched_filter"
    infofile: Path | None = None
    maskfile: Path | None = None
    beam1d: Path | None = None
    shrink_holes: AstroPydanticQuantity = 20 * u.arcmin
    apod_edge: AstroPydanticQuantity = 10 * u.arcmin
    apod_holes: AstroPydanticQuantity = 5 * u.arcmin
    noisemask_lim: float | None = None
    highpass: bool = False
    band_height: AstroPydanticQuantity = 0 * u.degree
    shift: float = 0
    simple: bool = False
    simple_lknee: float = 1000
    simple_alpha: float = -3.5
    lres: tuple = (70, 100)
    pixwin: str = "nn"

    def to_preprocessor(
        self, log: FilteringBoundLogger | None = None
    ) -> MapPreprocessor:
        return MatchedFilter(
            infofile=self.infofile,
            maskfile=self.maskfile,
            beam1d=self.beam1d,
            shrink_holes=self.shrink_holes,
            apod_edge=self.apod_edge,
            apod_holes=self.apod_holes,
            noisemask_lim=self.noisemask_lim,
            highpass=self.highpass,
            band_height=self.band_height,
            shift=self.shift,
            simple=self.simple,
            simple_lknee=self.simple_lknee,
            simple_alpha=self.simple_alpha,
            lres=self.lres,
            pixwin=self.pixwin,
            log=log,
        )


class MapShifterConfig(PreprocessorConfig):
    preprocessor_type: Literal["map_shifter"] = "map_shifter"
    shift_ra: AstroPydanticQuantity[u.arcmin]
    shift_dec: AstroPydanticQuantity[u.arcmin]

    def to_preprocessor(
        self, log: FilteringBoundLogger | None = None
    ) -> MapPreprocessor:
        return MapShifter(shift_ra=self.shift_ra, shift_dec=self.shift_dec, log=log)


AllPreprocessorConfigTypes = (
    KappaRhoCleanerConfig | MatchedFilterConfig | MapShifterConfig
)
