"""
Source subtraction steps.
"""

from abc import ABC, abstractmethod

from sotrplib.maps.core import ProcessableMap
from sotrplib.sources.sources import MeasuredSource


class SourceSubtractedMap(ProcessableMap):
    def __init__(self):
        return


class SourceSubtractor(ABC):
    @abstractmethod
    def subtract(
        self,
        sources: list[MeasuredSource],
        input_map: ProcessableMap,
    ) -> SourceSubtractedMap:
        return


class EmptySourceSubtractor(SourceSubtractor):
    """
    No-op for source subtraction (don't actually subtract the sources)
    """

    def subtract(
        self,
        sources: list[MeasuredSource],
        input_map: ProcessableMap,
    ) -> ProcessableMap:
        return input_map


class PhotutilsSourceSubtractor(SourceSubtractor):
    """
    Subtract sources using photutils.
    """

    def subtract(
        self,
        sources: list[MeasuredSource],
        input_map: ProcessableMap,
    ) -> ProcessableMap:
        from sotrplib.maps.maps import subtract_sources

        subtract_sources(input_map, sources, log=None)
        return input_map
