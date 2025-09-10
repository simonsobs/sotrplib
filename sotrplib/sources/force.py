from typing import List, Literal

import numpy as np
from astropy import units as u
from structlog import get_logger
from structlog.types import FilteringBoundLogger

from sotrplib.maps.core import ProcessableMap
from sotrplib.sources.core import ForcedPhotometryProvider
from sotrplib.sources.forced_photometry import (
    photutils_2D_gauss_fit,
    scipy_2d_gaussian_fit,
)
from sotrplib.sources.sources import MeasuredSource, RegisteredSource
from sotrplib.utils.utils import get_fwhm


class EmptyForcedPhotometry(ForcedPhotometryProvider):
    """
    Just return empty lists
    """

    def force(
        self,
        input_map: ProcessableMap,
        sources: List[RegisteredSource],
    ) -> list[MeasuredSource]:
        return []


class SimpleForcedPhotometry(ForcedPhotometryProvider):
    mode: Literal["spline", "nn"]
    sources = list[RegisteredSource]

    def __init__(
        self,
        mode: Literal["spline", "nn"],
        sources: List[RegisteredSource] = [],
        log: FilteringBoundLogger | None = None,
    ):
        self.mode = mode
        self.sources = sources
        self.log = log or get_logger()

    def force(
        self,
        input_map: ProcessableMap,
        sources: List[RegisteredSource],
    ):
        # Implement the single pixel forced photometry ("nn" is much faster than "spline")
        ## see https://github.com/simonsobs/pixell/blob/master/pixell/utils.py#L542
        out_sources = []
        for source in self.sources + sources:
            self.log.debug(
                "SimpleForcedPhotometry.source",
                ra=source.ra.to(u.deg),
                dec=source.dec.to(u.deg),
                crossmatch=source.crossmatches,
            )
            source_pos = np.array(
                [[source.dec.to_value(u.rad)], [source.ra.to_value(u.rad)]]
            )  # [[dec],[ra]] in radians
            flux = input_map.flux.at(source_pos, mode=self.mode)[0]
            snr = input_map.snr.at(source_pos, mode=self.mode)[0]
            out_source = MeasuredSource(
                **source.model_dump(),
                fit_method="nearest_neighbor" if self.mode == "nn" else "spline",
            )
            out_source.flux = flux * input_map.flux_units
            out_source.err_flux = flux / snr * input_map.flux_units
            out_sources.append(out_source)
            self.log.info(
                "SimpleForcedPhotometry.source_measured",
                ra=source.ra,
                dec=source.dec,
                flux=out_source.flux,
                err_flux=out_source.err_flux,
                snr=snr,
            )
        return out_sources


class Scipy2DGaussianFitter(ForcedPhotometryProvider):
    sources: list[RegisteredSource]
    flux_limit_centroid: u.Quantity  ## this should probably not exist within the function; i.e. be decided before handing the list to the provider.
    reproject_thumbnails: bool
    log: FilteringBoundLogger

    def __init__(
        self,
        sources: list[RegisteredSource] = [],
        flux_limit_centroid: u.Quantity = u.Quantity(0.3, "Jy"),
        reproject_thumbnails: bool = False,
        thumbnail_half_width: u.Quantity = u.Quantity(0.25, "deg"),
        log: FilteringBoundLogger | None = None,
    ):
        self.sources = sources
        self.flux_limit_centroid = flux_limit_centroid
        self.reproject_thumbnails = reproject_thumbnails
        self.thumbnail_half_width = thumbnail_half_width
        self.log = log or get_logger()

    def force(
        self, input_map: ProcessableMap, sources: list[RegisteredSource]
    ) -> list[MeasuredSource]:
        # TODO: refactor get_fwhm as part of the ProcessableMap
        fwhm = u.Quantity(
            get_fwhm(freq=input_map.frequency, arr=input_map.array), "arcmin"
        )

        fit_sources = scipy_2d_gaussian_fit(
            input_map,
            self.sources + sources,
            flux_lim_fit_centroid=self.flux_limit_centroid,
            thumbnail_half_width=self.thumbnail_half_width,
            fwhm=fwhm,
            reproject_thumb=self.reproject_thumbnails,
            log=self.log,
        )

        return fit_sources


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
    ) -> list[MeasuredSource]:
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
        rs: list[MeasuredSource] = []
        for i in range(len(sources["fluxJy"])):
            rsi = MeasuredSource(
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
