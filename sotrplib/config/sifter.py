from abc import ABC, abstractmethod
from typing import Literal

from pydantic import BaseModel
from structlog.types import FilteringBoundLogger

from sotrplib.sifter.core import DefaultSifter, EmptySifter, SiftingProvider
from sotrplib.sources.sources import RegisteredSource


class SifterConfig(BaseModel, ABC):
    sifter_type: str

    @abstractmethod
    def to_sifter(self, log: FilteringBoundLogger | None = None) -> SiftingProvider:
        return


class EmptySifterConfig(SifterConfig):
    sifter_type: Literal["empty"] = "empty"

    def to_sifter(self, log: FilteringBoundLogger | None = None) -> EmptySifter:
        return EmptySifter()


class DefaultSifterConfig(SifterConfig):
    sifter_type: Literal["default"] = "default"
    catalog_sources: list[RegisteredSource] | None = None

    def to_sifter(self, log: FilteringBoundLogger | None = None) -> SiftingProvider:
        return DefaultSifter(
            catalog_sources=self.catalog_sources,
            log=log,
        )


AllSifterConfigTypes = EmptySifterConfig | DefaultSifterConfig
