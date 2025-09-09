from abc import ABC, abstractmethod
from typing import Literal

from astropy import units as u
from astropydantic import AstroPydanticQuantity
from pydantic import BaseModel
from structlog.types import FilteringBoundLogger

from sotrplib.sources.force import (
    EmptyForcedPhotometry,
    ForcedPhotometryProvider,
    PhotutilsGaussianFitter,
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


class PhotutilsGaussianFitterConfig(ForcedPhotometryConfig):
    photometry_type: Literal["photutils"] = "photutils"
    # TODO actually support passing sources here!
    flux_limit_centroid: AstroPydanticQuantity[u.Jy] = u.Quantity(0.3, "Jy")

    def to_forced_photometry(
        self, log: FilteringBoundLogger | None = None
    ) -> PhotutilsGaussianFitter:
        return PhotutilsGaussianFitter(
            # TODO: Support non-simulated sources
            sources=[],
            flux_limit_centroid=self.flux_limit_centroid,
        )


AllForcedPhotometryConfigTypes = (
    EmptyPhotometryConfig | PhotutilsGaussianFitterConfig | SimpleForcedPhotometryConfig
)
