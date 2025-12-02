"""
Map Coadding configuration
"""

from abc import ABC, abstractmethod
from typing import Literal

from pydantic import BaseModel
from structlog.types import FilteringBoundLogger

from sotrplib.maps.core import (
    CoaddedRhoKappaMap,
    ProcessableMap,
)


class MapCoadderConfig(BaseModel, ABC):
    coadd_type: str

    @abstractmethod
    def coadd(self, log: FilteringBoundLogger | None = None) -> ProcessableMap:
        return


class RhoKappaMapCoadderConfig(MapCoadderConfig):
    coadd_type: Literal["rhokappa_coadd"] = "rhokappa_coadd"

    def coadd(
        self,
        input_maps: list[ProcessableMap],
        frequency: str,
        array: str | None = None,
        log: FilteringBoundLogger | None = None,
    ) -> ProcessableMap:
        coadd_map = CoaddedRhoKappaMap(
            input_maps=input_maps,
            frequency=frequency,
            array=array,
            log=log,
        )
        coadd_map.build()
        return coadd_map


AllMapCoadderConfigTypes = RhoKappaMapCoadderConfig
