"""
Core searches for known sources in a map (forced photometry) and
unknown sources in a map (blind search)
"""

from abc import ABC, abstractmethod

from pixell import enmap

from sotrplib.maps.core import ProcessableMap
from sotrplib.sources.sources import (
    BlindSearchSource,
    ForcedPhotometrySource,
    RegisteredSource,
)


class ForcedPhotometryProvider(ABC):
    # thumbnail now attribute of ForcedPhotometrySource
    @abstractmethod
    def force(
        self, input_map: ProcessableMap, sources: list[RegisteredSource]
    ) -> list[ForcedPhotometrySource]:
        return []


class BlindSearchProvider(ABC):
    @abstractmethod
    def search(
        self,
        input_map: ProcessableMap,
    ) -> tuple[list[BlindSearchSource], list[enmap.ndmap]]:
        return [], []
