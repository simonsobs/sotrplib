from abc import ABC, abstractmethod
from typing import Literal

from astropy import units as u
from astropydantic import AstroPydanticQuantity
from pydantic import BaseModel
from structlog.types import FilteringBoundLogger

from sotrplib.sources.force import (
    EmptyForcedPhotometry,
    ForcedPhotometryProvider,
    Lmfit2DGaussianFitter,
    Lmfit2DGaussianPointingFitter,
    Scipy2DGaussianFitter,
    Scipy2DGaussianPointingFitter,
    SimpleForcedPhotometry,
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

    def to_forced_photometry(
        self, log: FilteringBoundLogger | None = None
    ) -> Scipy2DGaussianFitter:
        return Scipy2DGaussianFitter(
            flux_limit_centroid=self.flux_limit_centroid,
            reproject_thumbnails=self.reproject_thumbnails,
            thumbnail_half_width=self.thumbnail_half_width,
            allowable_center_offset=self.allowable_center_offset,
            near_source_rel_flux_limit=self.near_source_rel_flux_limit,
            log=log,
        )


class LmfitGaussianFitterConfig(ForcedPhotometryConfig):
    photometry_type: Literal["lmfit"] = "lmfit"
    flux_limit_centroid: AstroPydanticQuantity[u.Jy] = u.Quantity(0.3, "Jy")
    reproject_thumbnails: bool = False
    thumbnail_half_width: AstroPydanticQuantity[u.deg] = u.Quantity(0.1, "deg")
    allowable_center_offset: AstroPydanticQuantity[u.arcmin] = u.Quantity(1.0, "arcmin")
    near_source_rel_flux_limit: float = 1.0

    def to_forced_photometry(
        self, log: FilteringBoundLogger | None = None
    ) -> Lmfit2DGaussianFitter:
        return Lmfit2DGaussianFitter(
            flux_limit_centroid=self.flux_limit_centroid,
            reproject_thumbnails=self.reproject_thumbnails,
            thumbnail_half_width=self.thumbnail_half_width,
            allowable_center_offset=self.allowable_center_offset,
            near_source_rel_flux_limit=self.near_source_rel_flux_limit,
            log=log,
        )


class Scipy2DGaussianPointingConfig(ForcedPhotometryConfig):
    photometry_type: Literal["scipy_pointing"] = "scipy_pointing"
    min_flux: AstroPydanticQuantity[u.Jy] = u.Quantity(0.3, "Jy")
    reproject_thumbnails: bool = False
    thumbnail_half_width: AstroPydanticQuantity[u.deg] = u.Quantity(0.1, "deg")
    allowable_center_offset: AstroPydanticQuantity[u.arcmin] = u.Quantity(3.0, "arcmin")
    near_source_rel_flux_limit: float = 0.3

    def to_forced_photometry(
        self, log: FilteringBoundLogger | None = None
    ) -> Scipy2DGaussianPointingFitter:
        return Scipy2DGaussianPointingFitter(
            min_flux=self.min_flux,
            reproject_thumbnails=self.reproject_thumbnails,
            thumbnail_half_width=self.thumbnail_half_width,
            allowable_center_offset=self.allowable_center_offset,
            near_source_rel_flux_limit=self.near_source_rel_flux_limit,
            log=log,
        )


class Lmfit2DGaussianPointingConfig(ForcedPhotometryConfig):
    photometry_type: Literal["lmfit_pointing"] = "lmfit_pointing"
    min_flux: AstroPydanticQuantity[u.Jy] = u.Quantity(0.3, "Jy")
    reproject_thumbnails: bool = False
    thumbnail_half_width: AstroPydanticQuantity[u.deg] = u.Quantity(0.1, "deg")
    allowable_center_offset: AstroPydanticQuantity[u.arcmin] = u.Quantity(3.0, "arcmin")
    near_source_rel_flux_limit: float = 0.3

    def to_forced_photometry(
        self, log: FilteringBoundLogger | None = None
    ) -> Lmfit2DGaussianPointingFitter:
        return Lmfit2DGaussianPointingFitter(
            min_flux=self.min_flux,
            reproject_thumbnails=self.reproject_thumbnails,
            thumbnail_half_width=self.thumbnail_half_width,
            allowable_center_offset=self.allowable_center_offset,
            near_source_rel_flux_limit=self.near_source_rel_flux_limit,
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
