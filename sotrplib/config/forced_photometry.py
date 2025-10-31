from abc import ABC, abstractmethod
from typing import Literal

from astropy import units as u
from astropydantic import AstroPydanticQuantity
from pydantic import BaseModel
from structlog.types import FilteringBoundLogger

from sotrplib.sources.force import (
    EmptyForcedPhotometry,
    ForcedPhotometryProvider,
    Scipy2DGaussianFitter,
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

    def to_forced_photometry(
        self, log: FilteringBoundLogger | None = None
    ) -> Scipy2DGaussianFitter:
        return Scipy2DGaussianFitter(
            flux_limit_centroid=self.flux_limit_centroid,
            reproject_thumbnails=self.reproject_thumbnails,
            thumbnail_half_width=self.thumbnail_half_width,
            log=log,
        )


AllForcedPhotometryConfigTypes = (
    EmptyPhotometryConfig | ScipyGaussianFitterConfig | SimpleForcedPhotometryConfig
)
