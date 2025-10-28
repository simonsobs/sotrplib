from abc import ABC, abstractmethod
from typing import Literal

from astropy import units as u
from astropydantic import AstroPydanticQuantity
from pydantic import BaseModel
from structlog.types import FilteringBoundLogger

from sotrplib.sifter.core import (
    DefaultSifter,
    EmptySifter,
    SiftingProvider,
    SimpleCatalogSifter,
)


class SifterConfig(BaseModel, ABC):
    sifter_type: str

    @abstractmethod
    def to_sifter(self, log: FilteringBoundLogger | None = None) -> SiftingProvider:
        return


class EmptySifterConfig(SifterConfig):
    sifter_type: Literal["empty"] = "empty"

    def to_sifter(self, log: FilteringBoundLogger | None = None) -> EmptySifter:
        return EmptySifter()


class SimpleCatalogSifterConfig(SifterConfig):
    sifter_type: Literal["simple"] = "simple"
    radius: AstroPydanticQuantity[u.arcmin] = 1.0 * u.arcmin
    method: Literal["closest", "all"] = "closest"

    def to_sifter(self, log: FilteringBoundLogger | None = None) -> SimpleCatalogSifter:
        return SimpleCatalogSifter(
            radius=self.radius,
            method=self.method,
            log=log,
        )


class DefaultSifterConfig(SifterConfig):
    sifter_type: Literal["default"] = "default"

    def to_sifter(self, log: FilteringBoundLogger | None = None) -> SiftingProvider:
        return DefaultSifter(
            log=log,
        )


AllSifterConfigTypes = (
    EmptySifterConfig | DefaultSifterConfig | SimpleCatalogSifterConfig
)
