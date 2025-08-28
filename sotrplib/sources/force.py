from typing import List, Literal

import numpy as np
from astropy import units as u
from astropydantic import AstroPydanticQuantity
from structlog import get_logger
from structlog.types import FilteringBoundLogger

from sotrplib.maps.core import ProcessableMap
from sotrplib.sources.core import ForcedPhotometryProvider
from sotrplib.sources.forced_photometry import (
    curve_fit_2d_gaussian,
    photutils_2D_gauss_fit,
)
from sotrplib.sources.sources import ForcedPhotometrySource, RegisteredSource
from sotrplib.utils.utils import get_fwhm


class EmptyForcedPhotometry(ForcedPhotometryProvider):
    """
    Just return empty lists
    """

    def force(
        self,
        input_map: ProcessableMap,
        sources: List[RegisteredSource],
    ) -> list[ForcedPhotometrySource]:
        return []


class SimpleForcedPhotometry(ForcedPhotometryProvider):
    mode: Literal["spline", "nn"]

    def __init__(
        self,
        mode: Literal["spline", "nn"],
    ):
        self.mode = mode

    def force(
        self,
        input_map: ProcessableMap,
        sources: List[RegisteredSource],
    ):
        # Implement the single pixel forced photometry ("nn" is much faster than "spline")
        ## see https://github.com/simonsobs/pixell/blob/master/pixell/utils.py#L542

        for source in sources:
            flux = input_map.flux.at(
                [source.dec.to_value(u.rad), source.ra.to_value(u.rad)], mode=self.mode
            )
            snr = input_map.snr.at(
                [source.dec.to_value(u.rad), source.ra.to_value(u.rad)], mode=self.mode
            )
            source.flux = AstroPydanticQuantity(u.Quantity(flux, input_map.flux_units))
            source.err_flux = AstroPydanticQuantity(
                u.Quantity(flux / snr, input_map.flux_units)
            )
            source.fit_method = "nearest_neighbor" if self.mode == "nn" else "spline"

        return sources


class Scipy2DGaussianFitter(ForcedPhotometryProvider):
    sources: list[RegisteredSource]
    flux_limit_centroid: u.Quantity  ## this should probably not exist within the function; i.e. be decided before handing the list to the provider.
    plot: bool
    reproject_thumbnails: bool
    log: FilteringBoundLogger

    def __init__(
        self,
        sources: list[RegisteredSource],
        flux_limit_centroid: u.Quantity = u.Quantity(0.3, "Jy"),
        plot: bool = False,
        reproject_thumbnails: bool = False,
        log: FilteringBoundLogger | None = None,
    ):
        self.sources = sources
        self.flux_limit_centroid = flux_limit_centroid
        self.plot = plot
        self.reproject_thumbnails = reproject_thumbnails
        self.log = log or get_logger()

    def force(
        self, input_map: ProcessableMap, sources: list[RegisteredSource]
    ) -> list[ForcedPhotometrySource]:
        # TODO: refactor get_fwhm as part of the ProcessableMap
        fwhm = u.Quantity(
            get_fwhm(freq=input_map.frequency, arr=input_map.array), "arcmin"
        )

        size_deg = fwhm.to_value("degree")
        fwhm_arcmin = fwhm.to_value("arcmin")

        sources, thumbnails = curve_fit_2d_gaussian(
            flux_map=input_map.flux,
            snr_map=input_map.snr,
            source_catalog=self.sources + sources,
            flux_lim_fit_centroid=self.flux_limit_centroid.to_value("Jy"),
            size_deg=size_deg,
            fwhm_arcmin=fwhm_arcmin,
            PLOT=self.plot,
            reproject_thumb=self.reproject_thumbnails,
            return_thumbnails=self.return_thumbnails,
            debug=True,
            log=self.log,
        )
        ## workaround until photutils_2D_gauss_fit reutrns forcedphotometry sources.
        ## convert the sourcecatalog to list of forcedphotometry sources
        rs: list[ForcedPhotometrySource] = []
        for i in range(len(sources["fluxJy"])):
            rsi = ForcedPhotometrySource(
                ra=sources["RADeg"][i] * u.deg,
                dec=sources["decDeg"][i] * u.deg,
                flux=sources["fluxJy"][i] * u.Jy,
            )
            if thumbnails:
                rsi.thumbnail = np.asarray(thumbnails[i])
                rsi.thumbnail_res = input_map.map_resolution
                rsi.thumbnail_unit = input_map.flux_units

            rs.append(rsi)
        return rs


class PhotutilsGaussianFitter(ForcedPhotometryProvider):
    # TODO: Figure out what type sources are
    sources: list
    flux_limit_centroid: u.Quantity
    plot: bool
    reproject_thumbnails: bool
    return_thumbnails: bool
    log: FilteringBoundLogger

    def __init__(
        self,
        sources: list,
        flux_limit_centroid: u.Quantity = u.Quantity(0.3, "Jy"),
        plot: bool = False,
        reproject_thumbnails: bool = False,
        return_thumbnails: bool = False,
        log: FilteringBoundLogger | None = None,
    ):
        self.sources = sources
        self.flux_limit_centroid = flux_limit_centroid
        self.plot = plot
        self.reproject_thumbnails = reproject_thumbnails
        self.return_thumbnails = return_thumbnails
        self.log = log or get_logger()

    def force(
        self, input_map: ProcessableMap, sources: list[RegisteredSource]
    ) -> list[ForcedPhotometrySource]:
        # TODO: refactor get_fwhm as part of the ProcessableMap
        fwhm = u.Quantity(
            get_fwhm(freq=input_map.frequency, arr=input_map.array), "arcmin"
        )

        size_deg = fwhm.to_value("degree")
        fwhm_arcmin = fwhm.to_value("arcmin")

        sources, thumbnails = photutils_2D_gauss_fit(
            flux_map=input_map.flux,
            snr_map=input_map.snr,
            source_catalog=self.sources + sources,
            flux_lim_fit_centroid=self.flux_limit_centroid.to_value("Jy"),
            size_deg=size_deg,
            fwhm_arcmin=fwhm_arcmin,
            PLOT=self.plot,
            reproject_thumb=self.reproject_thumbnails,
            return_thumbnails=self.return_thumbnails,
            debug=True,
            log=self.log,
        )
        ## workaround until photutils_2D_gauss_fit reutrns forcedphotometry sources.
        ## convert the sourcecatalog to list of forcedphotometry sources
        rs: list[ForcedPhotometrySource] = []
        for i in range(len(sources["fluxJy"])):
            rsi = ForcedPhotometrySource(
                ra=sources["RADeg"][i] * u.deg,
                dec=sources["decDeg"][i] * u.deg,
                flux=sources["fluxJy"][i] * u.Jy,
            )
            if thumbnails:
                rsi.thumbnail = np.asarray(thumbnails[i])
                rsi.thumbnail_res = input_map.map_resolution
                rsi.thumbnail_unit = input_map.flux_units

            rs.append(rsi)
        return rs
