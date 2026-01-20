import itertools
from typing import List, Literal

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
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
from sotrplib.utils.utils import angular_separation, get_fwhm


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
    allowable_center_offset: u.Quantity
    thumbnail_half_width: u.Quantity
    near_source_rel_flux_limit: float | None
    log: FilteringBoundLogger

    def __init__(
        self,
        flux_limit_centroid: u.Quantity = u.Quantity(0.3, "Jy"),
        reproject_thumbnails: bool = False,
        thumbnail_half_width: u.Quantity = u.Quantity(0.1, "deg"),
        near_source_rel_flux_limit: float | None = None,
        allowable_center_offset: u.Quantity = u.Quantity(1.0, "arcmin"),
        log: FilteringBoundLogger | None = None,
    ):
        """
        Initialize the Scipy2DGaussianFitter.
        Parameters
        ----------
        flux_limit_centroid : astropy.units.Quantity, optional
            Minimum flux required to attempt centroid fitting (default: 0.3 Jy).
        reproject_thumbnails : bool, optional
            Whether to reproject thumbnails before fitting (default: False).
            If False, just cuts out square in pixel space.
        thumbnail_half_width : astropy.units.Quantity, optional
            Half-width of the thumbnail cutout used for fitting (default: 0.1 deg).
        allowable_center_offset : astropy.units.Quantity, optional
            Maximum allowed offset from the initial guess position during Gaussian fitting (default: 1.0 arcmin).
        near_source_rel_flux_limit : float | None, optional
            Relative flux limit to exclude nearby sources (default: None).
            If another source within the thumbnail_half_width has a flux
            greater than near_source_rel_flux_limit * source.flux,
            flag the source as fitting failed.
            If None, ignore.
        log : FilteringBoundLogger or None, optional
            Logger instance to use (default: None).
        """
        self.flux_limit_centroid = flux_limit_centroid
        self.reproject_thumbnails = reproject_thumbnails
        self.thumbnail_half_width = thumbnail_half_width
        self.allowable_center_offset = allowable_center_offset
        self.near_source_rel_flux_limit = near_source_rel_flux_limit
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
        self.log.info(
            "Scipy2DGaussianFitter.force",
            n_sources=len(source_list),
            thumbnail_half_width=self.thumbnail_half_width,
            fwhm=fwhm,
            reproject_thumbnails=self.reproject_thumbnails,
            allowable_center_offset=self.allowable_center_offset,
        )
        if len(source_list) == 0:
            return []
        ## check if there are any pointing sources with a source within the thumbnail_half_width
        ## and above near_source_rel_flux_limit * pointing_source.flux
        has_nearby_sources = [False] * len(source_list)
        if self.near_source_rel_flux_limit is not None and len(source_list) > 1:
            source_positions = SkyCoord(
                ra=[s.ra for s in source_list], dec=[s.dec for s in source_list]
            )
            source_fluxes = (
                np.array([s.flux.to_value(u.Jy) for s in source_list]) * u.Jy
            )
            for i, fp_source in enumerate(source_list):
                c = SkyCoord(ra=fp_source.ra, dec=fp_source.dec)
                neighboridx, neighbordist, _ = c.match_to_catalog_sky(
                    source_positions, nthneighbor=2
                )
                nearby = neighbordist <= self.thumbnail_half_width * (2**-0.5)
                brighter = (
                    source_fluxes[neighboridx]
                    >= fp_source.flux * self.near_source_rel_flux_limit
                )
                if nearby & brighter:  # more than itself
                    has_nearby_sources[i] = True

        self.log.info(
            "Scipy2DGaussianFitter.force",
            flagged_nearby_sources=sum(has_nearby_sources),
        )

        fit_sources = scipy_2d_gaussian_fit(
            input_map,
            source_list=source_list,
            flux_lim_fit_centroid=self.flux_limit_centroid,
            thumbnail_half_width=self.thumbnail_half_width,
            fwhm=fwhm,
            reproject_thumb=self.reproject_thumbnails,
            pointing_residuals=pointing_residuals,
            allowable_center_offset=self.allowable_center_offset,
            flags={"nearby_source": has_nearby_sources},
            log=self.log,
        )

        return fit_sources


class Scipy2DGaussianPointingFitter(ForcedPhotometryProvider):
    min_flux: u.Quantity
    reproject_thumbnails: bool
    thumbnail_half_width: u.Quantity
    allowable_center_offset: u.Quantity
    near_source_rel_flux_limit: float
    log: FilteringBoundLogger

    def __init__(
        self,
        min_flux: u.Quantity = u.Quantity(0.3, "Jy"),
        reproject_thumbnails: bool = False,
        thumbnail_half_width: u.Quantity = u.Quantity(0.1, "deg"),
        near_source_rel_flux_limit: float = 0.3,
        allowable_center_offset: u.Quantity = u.Quantity(3.0, "arcmin"),
        log: FilteringBoundLogger | None = None,
    ):
        """
        Initialize the Scipy2DGaussianFitter.
        Parameters
        ----------
        min_flux : astropy.units.Quantity, optional
            Minimum flux required to attempt pointing fit (default: 0.3 Jy).
        reproject_thumbnails : bool, optional
            Whether to reproject thumbnails before fitting (default: False).
            If False, just cuts out square in pixel space.
        thumbnail_half_width : astropy.units.Quantity, optional
            Half-width of the thumbnail cutout used for fitting (default: 0.1 deg).
        near_source_rel_flux_limit : float, optional
            Relative flux limit to exclude nearby sources (default: 0.3).
            If another source within the thumbnail_half_width has a flux
            greater than near_source_rel_flux_limit * pointing_source.flux,
            the pointing source is excluded from fitting.
        allowable_center_offset : astropy.units.Quantity, optional
            Maximum allowed offset from the initial guess position during Gaussian fitting (default: 1.0 arcmin).
        log : FilteringBoundLogger or None, optional
            Logger instance to use (default: None).
        """
        self.min_flux = min_flux
        self.reproject_thumbnails = reproject_thumbnails
        self.thumbnail_half_width = thumbnail_half_width
        self.near_source_rel_flux_limit = near_source_rel_flux_limit
        self.allowable_center_offset = allowable_center_offset
        self.log = log or get_logger()

    def force(
        self, input_map: ProcessableMap, catalogs: list[SourceCatalog]
    ) -> list[MeasuredSource]:
        fwhm = get_fwhm(freq=input_map.frequency, arr=input_map.array)
        ## get all forced photometry sources from catalogs
        source_list = list(
            itertools.chain(*[c.forced_photometry_sources(input_map) for c in catalogs])
        )
        ## select only those above min_flux for pointing fitting
        pointing_source_list = [
            source
            for source in source_list
            if (source.flux is not None) and (source.flux > self.min_flux)
        ]
        ## check if there are any pointing sources with a source within the thumbnail_half_width
        ## and above near_source_rel_flux_limit * pointing_source.flux
        has_nearby_sources = [False] * len(pointing_source_list)
        for i, pointing_source in enumerate(pointing_source_list):
            for init_source in source_list:
                if pointing_source == init_source:
                    continue
                sep = angular_separation(
                    pointing_source.ra,
                    pointing_source.dec,
                    init_source.ra,
                    init_source.dec,
                )
                if (
                    sep < self.thumbnail_half_width * (2**0.5)
                    and init_source.flux is not None
                ):
                    if (
                        init_source.flux
                        > pointing_source.flux * self.near_source_rel_flux_limit
                    ):
                        has_nearby_sources[i] = True
                        break
        pointing_source_list = [
            src
            for src, has_nearby in zip(pointing_source_list, has_nearby_sources)
            if not has_nearby
        ]
        self.log.info(
            "Scipy2DGaussianPointingFitter.force",
            n_sources=len(pointing_source_list),
            min_flux=self.min_flux,
            thumbnail_half_width=self.thumbnail_half_width,
            fwhm=fwhm,
            reproject_thumbnails=self.reproject_thumbnails,
            allowable_center_offset=self.allowable_center_offset,
            removed_nearby_sources=sum(has_nearby_sources),
        )
        if len(pointing_source_list) == 0:
            return []
        fit_sources = scipy_2d_gaussian_fit(
            input_map,
            source_list=pointing_source_list,
            flux_lim_fit_centroid=self.min_flux,
            thumbnail_half_width=self.thumbnail_half_width,
            fwhm=fwhm,
            reproject_thumb=self.reproject_thumbnails,
            allowable_center_offset=self.allowable_center_offset,
            log=self.log,
        )

        return fit_sources
