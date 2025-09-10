from abc import ABC, abstractmethod
from typing import Literal

from pydantic import BaseModel
from structlog.types import FilteringBoundLogger

from sotrplib.sources.subtractor import (
    EmptySourceSubtractor,
    PhotutilsSourceSubtractor,
    SourceSubtractor,
)


class SourceSubtractorConfig(BaseModel, ABC):
    subtractor_type: str

    @abstractmethod
    def to_source_subtractor(
        self, log: FilteringBoundLogger | None = None
    ) -> SourceSubtractor:
        return


class EmptySourceSubtractorConfig(SourceSubtractorConfig):
    subtractor_type: Literal["empty"] = "empty"

    def to_source_subtractor(
        self, log: FilteringBoundLogger | None = None
    ) -> EmptySourceSubtractor:
        return EmptySourceSubtractor()


class PhotutilsSourceSubtractorConfig(SourceSubtractorConfig):
    subtractor_type: Literal["photutils"] = "photutils"

    def to_source_subtractor(
        self, log: FilteringBoundLogger | None = None
    ) -> SourceSubtractor:
        return PhotutilsSourceSubtractor()


AllSourceSubtractorConfigTypes = (
    EmptySourceSubtractorConfig | PhotutilsSourceSubtractorConfig
)
