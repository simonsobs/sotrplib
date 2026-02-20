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


class ScipyGaussianFitterConfig(ForcedPhotometryConfig):
    photometry_type: Literal["scipy"] = "scipy"
    flux_limit_centroid: AstroPydanticQuantity[u.Jy] = u.Quantity(0.3, "Jy")
    reproject_thumbnails: bool = False
    thumbnail_half_width: AstroPydanticQuantity[u.deg] = u.Quantity(0.1, "deg")
    allowable_center_offset: AstroPydanticQuantity[u.arcmin] = u.Quantity(1.0, "arcmin")
    near_source_rel_flux_limit: float = 1.0
    goodness_of_fit_threshold: float | None = None

    def to_forced_photometry(
        self, log: FilteringBoundLogger | None = None
    ) -> TwoDGaussianFitter:
        return TwoDGaussianFitter(
            mode="scipy",
            flux_limit_centroid=self.flux_limit_centroid,
            reproject_thumbnails=self.reproject_thumbnails,
            thumbnail_half_width=self.thumbnail_half_width,
            allowable_center_offset=self.allowable_center_offset,
            near_source_rel_flux_limit=self.near_source_rel_flux_limit,
            goodness_of_fit_threshold=self.goodness_of_fit_threshold,
            log=log,
        )


class LmfitGaussianFitterConfig(ForcedPhotometryConfig):
    photometry_type: Literal["lmfit"] = "lmfit"
    flux_limit_centroid: AstroPydanticQuantity[u.Jy] = u.Quantity(0.3, "Jy")
    reproject_thumbnails: bool = False
    thumbnail_half_width: AstroPydanticQuantity[u.deg] = u.Quantity(0.1, "deg")
    allowable_center_offset: AstroPydanticQuantity[u.arcmin] = u.Quantity(1.0, "arcmin")
    near_source_rel_flux_limit: float = 1.0
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
            log=log,
        )


class Scipy2DGaussianPointingConfig(ForcedPhotometryConfig):
    photometry_type: Literal["scipy_pointing"] = "scipy_pointing"
    min_flux: AstroPydanticQuantity[u.Jy] = u.Quantity(0.3, "Jy")
    reproject_thumbnails: bool = False
    thumbnail_half_width: AstroPydanticQuantity[u.deg] = u.Quantity(0.1, "deg")
    allowable_center_offset: AstroPydanticQuantity[u.arcmin] = u.Quantity(3.0, "arcmin")
    near_source_rel_flux_limit: float = 0.3
    goodness_of_fit_threshold: float | None = None

    def to_forced_photometry(
        self, log: FilteringBoundLogger | None = None
    ) -> TwoDGaussianFitter:
        return TwoDGaussianPointingFitter(
            mode="scipy",
            flux_limit_centroid=self.min_flux,
            reproject_thumbnails=self.reproject_thumbnails,
            thumbnail_half_width=self.thumbnail_half_width,
            allowable_center_offset=self.allowable_center_offset,
            near_source_rel_flux_limit=self.near_source_rel_flux_limit,
            goodness_of_fit_threshold=self.goodness_of_fit_threshold,
            log=log,
        )


class Lmfit2DGaussianPointingConfig(ForcedPhotometryConfig):
    photometry_type: Literal["lmfit_pointing"] = "lmfit_pointing"
    min_flux: AstroPydanticQuantity[u.Jy] = u.Quantity(0.3, "Jy")
    reproject_thumbnails: bool = False
    thumbnail_half_width: AstroPydanticQuantity[u.deg] = u.Quantity(0.1, "deg")
    allowable_center_offset: AstroPydanticQuantity[u.arcmin] = u.Quantity(3.0, "arcmin")
    near_source_rel_flux_limit: float = 0.3
    goodness_of_fit_threshold: float | None = None

    def to_forced_photometry(
        self, log: FilteringBoundLogger | None = None
    ) -> TwoDGaussianPointingFitter:
        return TwoDGaussianPointingFitter(
            mode="lmfit",
            flux_limit_centroid=self.min_flux,
            reproject_thumbnails=self.reproject_thumbnails,
            thumbnail_half_width=self.thumbnail_half_width,
            allowable_center_offset=self.allowable_center_offset,
            near_source_rel_flux_limit=self.near_source_rel_flux_limit,
            goodness_of_fit_threshold=self.goodness_of_fit_threshold,
            log=log,
        )


AllForcedPhotometryConfigTypes = (
    EmptyPhotometryConfig
    | ScipyGaussianFitterConfig
    | SimpleForcedPhotometryConfig
    | Scipy2DGaussianPointingConfig
    | LmfitGaussianFitterConfig
    | Lmfit2DGaussianPointingConfig
)
