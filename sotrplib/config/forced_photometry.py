from abc import ABC, abstractmethod
from typing import Literal

from astropy import units as u
from astropydantic import AstroPydanticQuantity
from pydantic import BaseModel
from structlog.types import FilteringBoundLogger

from sotrplib.sources.force import (
    EmptyForcedPhotometry,
    ForcedPhotometryProvider,
    SimpleForcedPhotometry,
    TwoDGaussianFitter,
    TwoDGaussianPointingFitter,
)


class ForcedPhotometryConfig(BaseModel, ABC):
    photometry_type: str

    @abstractmethod
    def to_forced_photometry(
        self, log: FilteringBoundLogger | None = None
    ) -> ForcedPhotometryProvider:
        return


class EmptyPhotometryConfig(ForcedPhotometryConfig):
    photometry_type: Literal["empty"] = "empty"

    def to_forced_photometry(
        self, log: FilteringBoundLogger | None = None
    ) -> EmptyForcedPhotometry:
        return EmptyForcedPhotometry()


class SimpleForcedPhotometryConfig(ForcedPhotometryConfig):
    photometry_type: Literal["simple"] = "simple"
    mode: Literal["spline", "nn"]

    def to_forced_photometry(self, log=None) -> SimpleForcedPhotometry:
        return SimpleForcedPhotometry(mode=self.mode, log=log)


class LmfitGaussianFitterConfig(ForcedPhotometryConfig):
    photometry_type: Literal["lmfit"] = "lmfit"
    flux_limit_centroid: AstroPydanticQuantity[u.Jy] = u.Quantity(0.3, "Jy")
    reproject_thumbnails: bool = False
    thumbnail_half_width: AstroPydanticQuantity[u.deg] = u.Quantity(0.1, "deg")
    allowable_center_offset: AstroPydanticQuantity[u.arcmin] = u.Quantity(1.0, "arcmin")
    near_source_rel_flux_limit: float = 1.0
    min_flux: AstroPydanticQuantity[u.Jy] | None = None
    goodness_of_fit_threshold: float | None = None

    def to_forced_photometry(
        self, log: FilteringBoundLogger | None = None
    ) -> TwoDGaussianFitter:
        return TwoDGaussianFitter(
            mode="lmfit",
            flux_limit_centroid=self.flux_limit_centroid,
            reproject_thumbnails=self.reproject_thumbnails,
            thumbnail_half_width=self.thumbnail_half_width,
            allowable_center_offset=self.allowable_center_offset,
            near_source_rel_flux_limit=self.near_source_rel_flux_limit,
            goodness_of_fit_threshold=self.goodness_of_fit_threshold,
            min_flux=self.min_flux,
            log=log,
        )


class Lmfit2DGaussianPointingConfig(ForcedPhotometryConfig):
    photometry_type: Literal["lmfit_pointing"] = "lmfit_pointing"
    min_flux: AstroPydanticQuantity[u.Jy] = u.Quantity(0.3, "Jy")
    reproject_thumbnails: bool = False
    thumbnail_half_width: AstroPydanticQuantity[u.deg] = u.Quantity(0.1, "deg")
    allowable_center_offset: AstroPydanticQuantity[u.arcmin] = u.Quantity(3.0, "arcmin")
    # Isolation cut: a pointing source is used only if no catalog source
    # within near_source_radius is brighter than 0.2 times its flux (it is
    # 5 times brighter than its neighbours). This removes most pointing
    # sources in crowded Galactic-plane regions.
    near_source_rel_flux_limit: float = 0.2
    near_source_radius: AstroPydanticQuantity[u.arcmin] | None = u.Quantity(
        20.0, "arcmin"
    )
    # Point-source cut: a fitted pointing source is removed if its larger
    # fitted FWHM is more than max_fwhm_ratio times the reference FWHM of
    # the band. This removes extended sources (Galactic dust, mostly at
    # f220 and f280). The reference values are the median fitted FWHM of
    # bright high-latitude sources in the phase-2 wide depth-1 maps
    # (matched filtered). The nominal beams (get_fwhm) are up to 23% smaller.
    max_fwhm_ratio: float | None = 1.5
    reference_fwhm: dict[str, AstroPydanticQuantity[u.arcmin]] | None = {
        "f090": u.Quantity(2.19, "arcmin"),
        "f150": u.Quantity(1.53, "arcmin"),
        "f220": u.Quantity(1.19, "arcmin"),
        "f280": u.Quantity(1.11, "arcmin"),
    }
    goodness_of_fit_threshold: float | None = None

    def to_forced_photometry(
        self, log: FilteringBoundLogger | None = None
    ) -> TwoDGaussianPointingFitter:
        return TwoDGaussianPointingFitter(
            mode="lmfit",
            min_flux=self.min_flux,
            reproject_thumbnails=self.reproject_thumbnails,
            thumbnail_half_width=self.thumbnail_half_width,
            allowable_center_offset=self.allowable_center_offset,
            near_source_rel_flux_limit=self.near_source_rel_flux_limit,
            near_source_radius=self.near_source_radius,
            max_fwhm_ratio=self.max_fwhm_ratio,
            reference_fwhm=self.reference_fwhm,
            goodness_of_fit_threshold=self.goodness_of_fit_threshold,
            log=log,
        )


AllForcedPhotometryConfigTypes = (
    EmptyPhotometryConfig
    | SimpleForcedPhotometryConfig
    | LmfitGaussianFitterConfig
    | Lmfit2DGaussianPointingConfig
)
