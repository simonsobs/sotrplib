"""
Source subtraction steps.
"""

from abc import ABC, abstractmethod

from sotrplib.maps.core import ProcessableMap
from sotrplib.maps.maps import subtract_sources
from sotrplib.sources.sources import MeasuredSource


class SourceSubtractedMap(ProcessableMap):
    def __init__(self):
        return

    def _compute_hits(self):
        return


class SourceSubtractor(ABC):
    """Abstract base for source-subtraction pipeline steps."""

    @abstractmethod
    def subtract(
        self,
        sources: list[MeasuredSource],
        input_map: ProcessableMap,
    ) -> SourceSubtractedMap:
        """Subtract sources from the input map.

        Parameters
        ----------
        sources : list of MeasuredSource
            Sources to subtract.
        input_map : ProcessableMap
            Map from which sources will be subtracted.

        Returns
        -------
        SourceSubtractedMap
        """
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
        subtract_sources(input_map, sources, log=None)
        return input_map
