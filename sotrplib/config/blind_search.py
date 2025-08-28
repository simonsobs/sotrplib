from abc import ABC, abstractmethod
from typing import Literal

from pydantic import BaseModel
from structlog.types import FilteringBoundLogger

from sotrplib.sources.blind import (
    BlindSearchProvider,
    EmptyBlindSearch,
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


AllBlindSearchConfigTypes = EmptyBlindSearchConfig
