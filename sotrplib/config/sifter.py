from abc import ABC, abstractmethod
from typing import Literal

from pydantic import BaseModel
from structlog.types import FilteringBoundLogger

from sotrplib.sifter.core import EmptySifter, SiftingProvider


class SifterConfig(BaseModel, ABC):
    sifter_type: str

    @abstractmethod
    def to_sifter(self, log: FilteringBoundLogger | None = None) -> SiftingProvider:
        return


class EmptySifterConfig(SifterConfig):
    sifter_type: Literal["empty"] = "empty"

    def to_sifter(self, log: FilteringBoundLogger | None = None) -> EmptySifter:
        return EmptySifter()


AllSifterConfigTypes = EmptySifterConfig
