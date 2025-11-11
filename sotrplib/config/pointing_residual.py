from abc import ABC, abstractmethod
from typing import Literal

from pydantic import BaseModel
from structlog.types import FilteringBoundLogger

from sotrplib.maps.pointing import (
    ConstantPointingOffset,
    EmptyPointingOffset,
    MapPointingOffset,
)


class PointingResidualConfig(BaseModel, ABC):
    pointing_residual_type: str

    @abstractmethod
    def to_residuals(
        self, log: FilteringBoundLogger | None = None
    ) -> MapPointingOffset:
        return


class EmptyPointingResidualConfig(PointingResidualConfig):
    pointing_residual_type: Literal["empty"] = "empty"

    def to_residuals(
        self, log: FilteringBoundLogger | None = None
    ) -> MapPointingOffset:
        return EmptyPointingOffset()


class ConstantResidualConfig(PointingResidualConfig):
    pointing_residual_type: Literal["mean", "median"] = "mean"
    min_snr: float = 5.0
    min_sources: int = 5
    sigma_clip_level: float = 3.0

    def to_residuals(
        self, log: FilteringBoundLogger | None = None
    ) -> MapPointingOffset:
        return ConstantPointingOffset(
            min_snr=self.min_snr,
            min_num=self.min_sources,
            avg_method="mean",
            sigma_clip_level=self.sigma_clip_level,
            log=log,
        )


AllPointingResidualConfigTypes = EmptyPointingResidualConfig | ConstantResidualConfig
