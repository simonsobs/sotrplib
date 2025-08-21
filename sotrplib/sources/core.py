"""
Core searches for known sources in a map (forced photometry) and
unknown sources in a map (blind search)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import structlog
from astropy import units as u
from pixell import enmap
from structlog.types import FilteringBoundLogger

from sotrplib.maps.core import ProcessableMap
from sotrplib.sources.finding import extract_sources
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


class EmptyForcedPhotometry(ForcedPhotometryProvider):
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
        self,
        input_map: ProcessableMap,
    ) -> tuple[list[SourceCandidate], list[enmap.ndmap]]:
        return


class EmptyBlindSearch(BlindSearchProvider):
    def search(
        self, input_map: ProcessableMap
    ) -> tuple[list[SourceCandidate], list[enmap.ndmap]]:
        return [], []


@dataclass
class BlindSearchParameters:
    """
    Parameters for blind source searching using photutils.
    """

    sigma_threshold: float = 5.0
    minimum_separation: list[u.Quantity] = field(
        default_factory=lambda: [u.Quantity(0.5, "arcmin")]
    )
    sigma_threshold_for_minimum_separation: list[float] = field(
        default_factory=lambda: [3.0]
    )


class SigmaClipBlindSearch(BlindSearchProvider):
    def __init__(
        self,
        input_map: ProcessableMap,
        log: FilteringBoundLogger | None = None,
        parameters: BlindSearchParameters | None = None,
        pixel_mask: enmap.ndmap | None = None,
    ):
        self.log = log or structlog.get_logger()
        self.input_map = input_map
        if not self.input_map.finalized:
            raise ValueError(
                "Input map must be finalized before searching for sources."
            )
        if self.input_map.res is None:
            raise ValueError("Flux map must have a resolution.")
        self.parameters = parameters or BlindSearchParameters()
        self.pixel_mask = pixel_mask
        self.log.info("SigmaClipBlindSearch.initialized", parameters=self.parameters)

    def search(
        self,
    ) -> tuple[list[SourceCandidate], list[enmap.ndmap]]:
        res_arcmin = self.input_map.res * self.input_map.map_resolution_units
        res_arcmin = res_arcmin.to(u.arcmin)
        self.log.info(
            "SigmaClipBlindSearch.searching",
            res_arcmin=res_arcmin,
            nsigma=self.parameters.sigma_threshold,
            minrad=[ms.to(u.arcmin).value for ms in self.parameters.minimum_separation],
            sigma_thresh_for_minrad=self.parameters.sigma_threshold_for_minimum_separation,
        )
        extracted_sources = extract_sources(
            self.input_map.flux,
            timemap=self.input_map.time_mean,
            maprms=abs(self.input_map.flux / self.input_map.snr),
            nsigma=self.parameters.sigma_threshold,
            minrad=[ms.to(u.arcmin).value for ms in self.parameters.minimum_separation],
            sigma_thresh_for_minrad=self.parameters.sigma_threshold_for_minimum_separation,
            res=res_arcmin.value,
            log=self.log,
        )
        return extracted_sources, []
