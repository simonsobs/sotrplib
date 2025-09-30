"""
Core searches for known sources in a map (forced photometry) and
unknown sources in a map (blind search)
"""

from abc import ABC, abstractmethod

from pixell import enmap

from sotrplib.maps.core import ProcessableMap
from sotrplib.source_catalog.core import SourceCatalog
from sotrplib.sources.sources import (
    MeasuredSource,
)


class ForcedPhotometryProvider(ABC):
    @abstractmethod
    def force(
        self, input_map: ProcessableMap, catalogs: list[SourceCatalog]
    ) -> list[MeasuredSource]:
        return []


class BlindSearchProvider(ABC):
    @abstractmethod
    def search(
        self,
        input_map: ProcessableMap,
    ) -> tuple[list[MeasuredSource], list[enmap.ndmap]]:
        return [], []
