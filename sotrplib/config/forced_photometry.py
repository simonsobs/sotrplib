from abc import ABC, abstractmethod
from typing import Literal

from pydantic import BaseModel
from structlog.types import FilteringBoundLogger

from sotrplib.sources.core import EmptyForcedPhotometry, ForcedPhotometryProvider


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


AllForcedPhotometryConfigTypes = EmptyPhotometryConfig
