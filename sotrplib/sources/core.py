"""
Core searches for known sources in a map (forced photometry) and
unknown sources in a map (blind search)
"""

from abc import ABC, abstractmethod
from typing import List

from pixell import enmap

from sotrplib.maps.core import ProcessableMap
from sotrplib.sources.sources import (
    ForcedPhotometrySource,
    RegisteredSource,
    SourceCandidate,
)


class ForcedPhotometryProvider(ABC):
    # thumbnail now attribute of ForcedPhotometrySource
    @abstractmethod
    def force(
        self, input_map: ProcessableMap, sources: List[RegisteredSource]
    ) -> list[ForcedPhotometrySource]:
        return []


class BlindSearchProvider(ABC):
    @abstractmethod
    def search(
        self,
        input_map: ProcessableMap,
    ) -> tuple[list[SourceCandidate], list[enmap.ndmap]]:
        return [], []
