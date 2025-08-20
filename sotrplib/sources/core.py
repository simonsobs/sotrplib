"""
Core searches for known sources in a map (forced photometry) and
unknown sources in a map (blind search)
"""

from abc import ABC, abstractmethod

import structlog
from astropy import units as u
from pixell import enmap
from structlog.types import FilteringBoundLogger

from sotrplib.maps.core import ProcessableMap
from sotrplib.sources.forced_photometry import photutils_2D_gauss_fit
from sotrplib.sources.sources import SourceCandidate
from sotrplib.utils.utils import get_fwhm


class ForcedPhotometryProvider(ABC):
    # TODO: consdier just putting the thumbnail in the SourceCandidate if we think it is actually useful
    # and we will likely want to send that up to the lightcurve server anyway.
    @abstractmethod
    def force(
        self, input_map: ProcessableMap
    ) -> tuple[list[SourceCandidate], list[enmap.ndmap]]:
        return


class EmtpyForcedPhotometry(ForcedPhotometryProvider):
    def force(self, input_map):
        return [], []


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
        self.log = log or structlog.get_logger()

    def force(
        self, input_map: ProcessableMap
    ) -> tuple[list[SourceCandidate], list[enmap.ndmap]]:
        # TODO: refactor get_fwhm as part of the ProcessableMap
        fwhm = u.Quantity(
            get_fwhm(freq=input_map.frequency, arr=input_map.array), "arcmin"
        )

        size_deg = fwhm.to_value("degree")
        fwhm_arcmin = fwhm.to_value("arcmin")

        sources, thumbnails = photutils_2D_gauss_fit(
            flux_map=input_map.flux,
            snr_map=input_map.snr,
            source_catalog=self.sources,
            flux_lim_fit_centroid=self.flux_limit_centroid.to_value("Jy"),
            size_deg=size_deg,
            fwhm_arcmin=fwhm_arcmin,
            PLOT=self.plot,
            reproject_thumb=self.reproject_thumbnails,
            return_thumbnails=self.return_thumbnails,
            debug=True,
            log=self.log,
        )

        return sources, thumbnails


class BlindSearchProvider(ABC):
    @abstractmethod
    def search(
        self, input_map: ProcessableMap
    ) -> tuple[list[SourceCandidate], list[enmap.ndmap]]:
        return


class EmptyBlindSearch(BlindSearchProvider):
    def search(
        self, input_map: ProcessableMap
    ) -> tuple[list[SourceCandidate], list[enmap.ndmap]]:
        return [], []


class SigmaClipBlindSearch(BlindSearchProvider):
    def search(
        self, input_map: ProcessableMap
    ) -> tuple[list[SourceCandidate], list[enmap.ndmap]]:
        raise NotImplementedError
