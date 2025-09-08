from abc import ABC, abstractmethod
from typing import Literal

from numpydantic import NDArray
from pydantic import BaseModel
from structlog.types import FilteringBoundLogger

from sotrplib.sources.blind import (
    BlindSearchParameters,
    BlindSearchProvider,
    EmptyBlindSearch,
    SigmaClipBlindSearch,
)


class BlindSearchConfig(BaseModel, ABC):
    search_type: str

    @abstractmethod
    def to_search_provider(
        self, log: FilteringBoundLogger | None = None
    ) -> BlindSearchProvider:
        return


class EmptyBlindSearchConfig(BlindSearchConfig):
    search_type: Literal["empty"] = "empty"

    def to_search_provider(
        self, log: FilteringBoundLogger | None = None
    ) -> EmptyBlindSearch:
        return EmptyBlindSearch()


class PhotutilsBlindSearchConfig(BlindSearchConfig):
    search_type: Literal["photutils"] = "photutils"
    parameters: BlindSearchParameters | None = None
    pixel_mask: NDArray | None = None

    def to_search_provider(
        self, log: FilteringBoundLogger | None = None
    ) -> BlindSearchProvider:
        return SigmaClipBlindSearch(
            parameters=self.parameters,
            pixel_mask=self.pixel_mask,
            log=log,
        )


AllBlindSearchConfigTypes = (
    EmptyBlindSearchConfig,
    PhotutilsBlindSearchConfig,
)
