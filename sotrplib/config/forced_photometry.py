import warnings
from abc import ABC, abstractmethod
from typing import Any, Literal

from astropy import units as u
from astropydantic import AstroPydanticQuantity
from pydantic import BaseModel, model_validator
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
    reproject_thumbnails: bool = False
    thumbnail_half_width: AstroPydanticQuantity[u.deg] = u.Quantity(0.1, "deg")
    near_source_rel_flux_limit: float = 1.0
    min_flux: AstroPydanticQuantity[u.Jy] | None = None
    goodness_of_fit_threshold: float | None = None

    @model_validator(mode="before")
    @classmethod
    def ignore_centroid_options(cls, data: Any) -> Any:
        # Forced photometry does not fit centroids now. Older configs can
        # have these options: ignore them, with a warning.
        if isinstance(data, dict):
            for key in ("flux_limit_centroid", "allowable_center_offset"):
                if key in data:
                    warnings.warn(
                        f"forced_photometry.{key} is not used: forced photometry "
                        "fits each source at its position. Remove it from the config.",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                    data = {k: v for k, v in data.items() if k != key}
        return data

    def to_forced_photometry(
        self, log: FilteringBoundLogger | None = None
    ) -> TwoDGaussianFitter:
        return TwoDGaussianFitter(
            mode="lmfit",
            reproject_thumbnails=self.reproject_thumbnails,
            thumbnail_half_width=self.thumbnail_half_width,
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
    near_source_rel_flux_limit: float = 0.3
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
            goodness_of_fit_threshold=self.goodness_of_fit_threshold,
            log=log,
        )


AllForcedPhotometryConfigTypes = (
    EmptyPhotometryConfig
    | SimpleForcedPhotometryConfig
    | LmfitGaussianFitterConfig
    | Lmfit2DGaussianPointingConfig
)
