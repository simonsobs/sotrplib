from abc import ABC, abstractmethod
from typing import Literal

from astropy import units as u
from astropydantic import AstroPydanticQuantity
from pydantic import BaseModel
from structlog.types import FilteringBoundLogger

from sotrplib.sifter.core import (
    DEFAULT_SIFTER_CUTS,
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
    min_match_radius: AstroPydanticQuantity[u.arcmin] = 1.5 * u.arcmin
    cuts: dict[str, list[float]] | None = None
    'Cuts to change, {name: [min, max]}, for example {"snr": [3.0, inf]}. '
    "The other cuts keep their values in DEFAULT_SIFTER_CUTS. A candidate "
    "outside a cut becomes a noise candidate."

    def to_sifter(self, log: FilteringBoundLogger | None = None) -> SiftingProvider:
        return DefaultSifter(
            min_match_radius=self.min_match_radius,
            cuts={**DEFAULT_SIFTER_CUTS, **(self.cuts or {})},
            log=log,
        )


AllSifterConfigTypes = (
    EmptySifterConfig | DefaultSifterConfig | SimpleCatalogSifterConfig
)
