"""
Map Coadding configuration
"""

from abc import ABC, abstractmethod
from typing import Literal

from pydantic import BaseModel
from structlog.types import FilteringBoundLogger

from sotrplib.maps.map_coadding import EmptyMapCoadder, MapCoadder, RhoKappaMapCoadder


class MapCoadderConfig(BaseModel, ABC):
    coadd_type: str

    @abstractmethod
    def to_coadder(
        self,
    ) -> MapCoadder:
        return


class EmptyMapCoadderConfig(MapCoadderConfig):
    coadd_type: Literal["empty"] = "empty"

    def to_coadder(self) -> MapCoadder:
        return EmptyMapCoadder()


class RhoKappaMapCoadderConfig(MapCoadderConfig):
    coadd_type: Literal["rhokappa_coadd"] = "rhokappa_coadd"

    frequencies: list[str] | None = None
    arrays: list[str] | None = None
    instrument: str | None = None

    def to_coadder(
        self,
        log: FilteringBoundLogger | None = None,
    ) -> MapCoadder:
        return RhoKappaMapCoadder(
            frequencies=self.frequencies,
            arrays=self.arrays,
            instrument=self.instrument,
            log=log,
        )


AllMapCoadderConfigTypes = RhoKappaMapCoadderConfig | EmptyMapCoadderConfig
