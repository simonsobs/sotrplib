import itertools
from typing import List, Literal

import numpy as np
from astropy import units as u
from structlog import get_logger
from structlog.types import FilteringBoundLogger

from sotrplib.maps.core import ProcessableMap
from sotrplib.maps.pointing import MapPointingOffset
from sotrplib.source_catalog.core import SourceCatalog
from sotrplib.sources.core import ForcedPhotometryProvider
from sotrplib.sources.forced_photometry import (
    scipy_2d_gaussian_fit,
)
from sotrplib.sources.sources import MeasuredSource, RegisteredSource
from sotrplib.utils.utils import get_fwhm


class EmptyForcedPhotometry(ForcedPhotometryProvider):
    """
    Just return empty lists
    """

    def __init__(
        self,
        log: FilteringBoundLogger | None = None,
    ):
        self.log = log or get_logger()

    def force(
        self,
        input_map: ProcessableMap,
        catalogs: List[SourceCatalog],
        pointing_residuals: MapPointingOffset | None = None,
    ) -> list[MeasuredSource]:
        return []


class SimpleForcedPhotometry(ForcedPhotometryProvider):
    mode: Literal["spline", "nn"]

    def __init__(
        self,
        mode: Literal["spline", "nn"],
        sources: List[RegisteredSource] = [],
        log: FilteringBoundLogger | None = None,
    ):
        self.mode = mode
        self.log = log or get_logger()

    def force(
        self,
        input_map: ProcessableMap,
        catalogs: List[SourceCatalog],
        pointing_residuals: MapPointingOffset | None = None,
    ):
        # Implement the single pixel forced photometry ("nn" is much faster than "spline")
        ## see https://github.com/simonsobs/pixell/blob/master/pixell/utils.py#L542
        out_sources = []
        source_list = list(
            itertools.chain(*[c.forced_photometry_sources(input_map) for c in catalogs])
        )
        for source in source_list:
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
                instrument=input_map.instrument,
                array=input_map.array,
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
    flux_limit_centroid: u.Quantity  ## this should probably not exist within the function; i.e. be decided before handing the list to the provider.
    reproject_thumbnails: bool
    log: FilteringBoundLogger

    def __init__(
        self,
        flux_limit_centroid: u.Quantity = u.Quantity(0.3, "Jy"),
        reproject_thumbnails: bool = False,
        thumbnail_half_width: u.Quantity = u.Quantity(0.1, "deg"),
        log: FilteringBoundLogger | None = None,
    ):
        self.flux_limit_centroid = flux_limit_centroid
        self.reproject_thumbnails = reproject_thumbnails
        self.thumbnail_half_width = thumbnail_half_width
        self.log = log or get_logger()

    def force(
        self,
        input_map: ProcessableMap,
        catalogs: list[SourceCatalog],
        pointing_residuals: MapPointingOffset | None = None,
    ) -> list[MeasuredSource]:
        fwhm = get_fwhm(freq=input_map.frequency, arr=input_map.array)
        source_list = list(
            itertools.chain(*[c.forced_photometry_sources(input_map) for c in catalogs])
        )
        fit_sources = scipy_2d_gaussian_fit(
            input_map,
            source_list=source_list,
            flux_lim_fit_centroid=self.flux_limit_centroid,
            thumbnail_half_width=self.thumbnail_half_width,
            fwhm=fwhm,
            reproject_thumb=self.reproject_thumbnails,
            pointing_residuals=pointing_residuals,
            log=self.log,
        )

        return fit_sources


class Scipy2DGaussianPointingFitter(ForcedPhotometryProvider):
    min_flux: u.Quantity
    reproject_thumbnails: bool
    log: FilteringBoundLogger

    def __init__(
        self,
        min_flux: u.Quantity = u.Quantity(0.3, "Jy"),
        reproject_thumbnails: bool = False,
        thumbnail_half_width: u.Quantity = u.Quantity(0.1, "deg"),
        log: FilteringBoundLogger | None = None,
    ):
        self.min_flux = min_flux
        self.reproject_thumbnails = reproject_thumbnails
        self.thumbnail_half_width = thumbnail_half_width
        self.log = log or get_logger()

    def force(
        self, input_map: ProcessableMap, catalogs: list[SourceCatalog]
    ) -> list[MeasuredSource]:
        fwhm = get_fwhm(freq=input_map.frequency, arr=input_map.array)
        source_list = list(
            itertools.chain(*[c.forced_photometry_sources(input_map) for c in catalogs])
        )
        pointing_source_list = [
            source
            for source in source_list
            if (source.flux is not None) and (source.flux > self.min_flux)
        ]
        self.log.info(
            "Scipy2DGaussianPointingFitter.force",
            n_sources=len(pointing_source_list),
            min_flux=self.min_flux,
            thumbnail_half_width=self.thumbnail_half_width,
            fwhm=fwhm,
            reproject_thumbnails=self.reproject_thumbnails,
        )

        fit_sources = scipy_2d_gaussian_fit(
            input_map,
            source_list=pointing_source_list,
            flux_lim_fit_centroid=self.min_flux,
            thumbnail_half_width=self.thumbnail_half_width,
            fwhm=fwhm,
            reproject_thumb=self.reproject_thumbnails,
            log=self.log,
        )

        return fit_sources
