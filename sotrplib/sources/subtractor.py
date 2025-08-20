"""
Source subtraction steps.
"""

from abc import ABC, abstractmethod

from sotrplib.maps.core import ProcessableMap
from sotrplib.sources.sources import SourceCandidate


class SourceSubtractedMap(ProcessableMap):
    def __init__(self):
        return


class SourceSubtractor(ABC):
    @abstractmethod
    def subtract(
        self, sources: list[SourceCandidate], input_map: ProcessableMap
    ) -> SourceSubtractedMap:
        return


class EmptySourceSubtractor(SourceSubtractor):
    """
    No-op for source subtraction (don't actually subtract the sources)
    """

    def subtract(
        self, sources: list[SourceCandidate], input_map: ProcessableMap
    ) -> ProcessableMap:
        return input_map
