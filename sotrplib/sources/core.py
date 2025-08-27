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
    # TODO: consdier just putting the thumbnail in the SourceCandidate if we think it is actually useful
    # and we will likely want to send that up to the lightcurve server anyway.
    @abstractmethod
    def force(
        self, input_map: ProcessableMap, sources: List[RegisteredSource]
    ) -> tuple[list[ForcedPhotometrySource], list[enmap.ndmap]]:
        return [], []


class BlindSearchProvider(ABC):
    @abstractmethod
    def search(
        self,
        input_map: ProcessableMap,
    ) -> tuple[list[SourceCandidate], list[enmap.ndmap]]:
        return
