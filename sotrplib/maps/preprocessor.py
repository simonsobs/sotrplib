"""
Dependencies for pre-processing (e.g. map cleaning). Affects
maps in-place. These operations happen in between 'build' and
'finalize' for the map creatino and i/o process.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

from astropy import units as u
from astropydantic import AstroPydanticQuantity
from structlog.types import FilteringBoundLogger

from sotrplib.filters.filters import matched_filter_depth1_map
from sotrplib.maps.core import ProcessableMap
from sotrplib.maps.maps import clean_map, kappa_clean, shift_map_radec
from sotrplib.utils.utils import get_frequency, get_fwhm


class MapPreprocessor(ABC):
    @abstractmethod
    def preprocess(self, input_map: ProcessableMap) -> ProcessableMap:
        """
        Provides preprocessing for the input map
        """
        return input_map


class KappaRhoCleaner(MapPreprocessor):
    cut_on: Literal["median", "percentile", "max"] = "median"
    fraction: float = 0.05

    def __init__(
        self,
        cut_on: Literal["median", "percentile", "max"] = "median",
        fraction: float = 0.05,
    ):
        self.cut_on = cut_on
        self.fraction = fraction

    def preprocess(self, input_map: ProcessableMap) -> ProcessableMap:
        # TODO: Figure out what warnings this raises
        input_map.kappa = kappa_clean(kappa=input_map.kappa, rho=input_map.rho)
        input_map.rho = clean_map(
            imap=input_map.rho,
            inverse_variance=input_map.kappa,
            cut_on=self.cut_on,
            fraction=self.fraction,
        )

        return input_map


class MatchedFilter(MapPreprocessor):
    def __init__(
        self,
        infofile: Path | None = None,
        maskfile: Path | None = None,
        beam1d: Path | None = None,
        shrink_holes: AstroPydanticQuantity = 20 * u.arcmin,
        apod_edge: AstroPydanticQuantity = 10 * u.arcmin,
        apod_holes: AstroPydanticQuantity = 5 * u.arcmin,
        noisemask_lim: float | None = None,
        highpass: bool = False,
        band_height: AstroPydanticQuantity = 0 * u.degree,
        shift: float = 0,
        simple: bool = False,
        simple_lknee: float = 1000,
        simple_alpha: float = -3.5,
        lres=(70, 100),
        pixwin: str = "nn",
        log: FilteringBoundLogger | None = None,
    ):
        self.infofile = infofile
        self.maskfile = maskfile
        self.beam1d = beam1d
        self.shrink_holes = shrink_holes
        self.apod_edge = apod_edge
        self.apod_holes = apod_holes
        self.noisemask_lim = noisemask_lim
        self.highpass = highpass
        self.band_height = band_height
        self.shift = shift
        self.simple = simple
        self.simple_lknee = simple_lknee
        self.simple_alpha = simple_alpha
        self.lres = lres
        self.pixwin = pixwin
        self.log = log

    def preprocess(self, input_map: ProcessableMap) -> ProcessableMap:
        rho, kappa = matched_filter_depth1_map(
            imap=input_map.intensity / input_map.intensity_units.to(u.uK),
            ivarmap=input_map.inverse_variance
            * input_map.intensity_units.to(u.uK) ** 2,
            band_center=get_frequency(input_map.frequency),
            infofile=self.infofile,
            maskfile=self.maskfile,
            beam_fwhm=get_fwhm(input_map.frequency),
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
        )
        input_map.matched_filtered = True
        input_map.intensity = rho
        input_map.inverse_variance = kappa
        input_map.snr = rho * kappa**-0.5
        input_map.flux = rho / kappa
        input_map.flux_units = u.mJy
        input_map.intensity_units = u.mJy
        return input_map


class MapShifter(MapPreprocessor):
    def __init__(
        self,
        shift_ra: AstroPydanticQuantity[u.arcmin],
        shift_dec: AstroPydanticQuantity[u.arcmin],
        log: FilteringBoundLogger | None = None,
    ):
        self.shift_ra = shift_ra
        self.shift_dec = shift_dec
        self.log = log

    def preprocess(self, input_map: ProcessableMap) -> ProcessableMap:
        return shift_map_radec(input_map, self.shift_ra, self.shift_dec, log=self.log)
