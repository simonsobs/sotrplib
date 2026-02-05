"""
Dependencies for pre-processing (e.g. map cleaning). Affects
maps in-place. These operations happen in between 'build' and
'finalize' for the map creatino and i/o process.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

import structlog
from astropy import units as u
from astropydantic import AstroPydanticQuantity
from structlog.types import FilteringBoundLogger

from sotrplib.filters.filters import matched_filter_depth1_map
from sotrplib.maps.core import (
    MatchedFilteredIntensityAndInverseVarianceMap,
    ProcessableMap,
)
from sotrplib.maps.maps import clean_map, kappa_clean
from sotrplib.maps.masks import mask_edge, mask_planets
from sotrplib.utils.utils import get_frequency, get_fwhm


class MapPreprocessor(ABC):
    @abstractmethod
    def preprocess(self, input_map: ProcessableMap) -> ProcessableMap:
        """
        Provides preprocessing for the input map
        """
        return input_map


class PlanetMasker(MapPreprocessor):
    mask_radius: AstroPydanticQuantity = 15 * u.arcmin

    def __init__(
        self,
        mask_radius: AstroPydanticQuantity = 15 * u.arcmin,
        log: FilteringBoundLogger | None = None,
    ):
        self.mask_radius = mask_radius
        self.log = log or structlog.get_logger()

    def preprocess(self, input_map: ProcessableMap) -> ProcessableMap:
        log = self.log.bind(func="PlanetMasker.preprocess")
        log.info(
            "PlanetMasker.preprocess.masking_planets",
            mask_radius=self.mask_radius,
            time_range=[input_map.observation_start, input_map.observation_end],
        )
        planet_mask = mask_planets(
            input_map.time_mean,
            [input_map.observation_start, input_map.observation_end],
            mask_radius=self.mask_radius,
            log=log,
        )
        input_map.mask = (
            input_map.mask * planet_mask if input_map.mask is not None else planet_mask
        )
        log.info("PlanetMasker.preprocess.completed")
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
        noisemask_radius: AstroPydanticQuantity = 10 * u.arcmin,
        highpass: bool = False,
        band_height: AstroPydanticQuantity = 1 * u.degree,
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
        self.noisemask_radius = noisemask_radius
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
            imap=input_map.intensity * input_map.intensity_units.to(u.K),
            ivarmap=input_map.inverse_variance / input_map.intensity_units.to(u.K) ** 2,
            band_center=get_frequency(
                input_map.frequency, instrument=input_map.instrument
            ),
            infofile=self.infofile,
            maskfile=self.maskfile,
            source_mask=input_map.mask,
            beam_fwhm=get_fwhm(input_map.frequency, instrument=input_map.instrument),
            beam1d=self.beam1d,
            shrink_holes=self.shrink_holes,
            apod_edge=self.apod_edge,
            apod_holes=self.apod_holes,
            noisemask_lim=self.noisemask_lim,
            noisemask_radius=self.noisemask_radius,
            highpass=self.highpass,
            band_height=self.band_height,
            shift=self.shift,
            simple=self.simple,
            simple_lknee=self.simple_lknee,
            simple_alpha=self.simple_alpha,
            lres=self.lres,
            pixwin=self.pixwin,
        )
        return MatchedFilteredIntensityAndInverseVarianceMap(
            rho=rho, kappa=kappa, prefiltered_map=input_map, flux_units=u.mJy
        )


class EdgeMask(MapPreprocessor):
    edge_width: AstroPydanticQuantity = AstroPydanticQuantity(10.0 * u.arcmin)
    mask_on: Literal["rho", "kappa", "flux", "snr", "inverse_variance", "intensity"] = (
        "kappa"
    )

    def __init__(
        self,
        edge_width: u.Quantity = u.Quantity(10.0, "arcmin"),
        mask_on: Literal[
            "rho", "kappa", "flux", "snr", "inverse_variance", "intensity"
        ] = "kappa",
        log: FilteringBoundLogger | None = None,
    ):
        self.edge_width = edge_width
        self.mask_on = mask_on
        self.log = log or structlog.get_logger()

    def preprocess(self, input_map: ProcessableMap) -> ProcessableMap:
        self.log = self.log.bind(func="EdgeMask.preprocess")
        number_of_pixels = int(
            self.edge_width.to_value(u.arcmin)
            / input_map.map_resolution.to_value(u.arcmin)
        )
        self.log = self.log.bind(number_of_pixels=number_of_pixels)
        mask_map = (
            getattr(input_map, self.mask_on)
            if hasattr(input_map, self.mask_on)
            else None
        )
        if mask_map is None:
            self.log.error(
                "EdgeMask.preprocess.get_mask_map.no_attribute", mask_on=self.mask_on
            )
            return input_map

        edge_mask = mask_edge(imap=mask_map, pix_num=number_of_pixels)
        input_map.mask = edge_mask
        self.log.info("EdgeMask.preprocess completed")

        return input_map
