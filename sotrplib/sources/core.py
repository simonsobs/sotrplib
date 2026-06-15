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
    """Abstract base for forced-photometry pipeline steps."""

    @abstractmethod
    def force(
        self, input_map: ProcessableMap, catalogs: list[SourceCatalog]
    ) -> list[MeasuredSource]:
        """Run forced photometry on a map using catalog positions.

        Parameters
        ----------
        input_map : ProcessableMap
            Finalized map on which to measure fluxes.
        catalogs : list of SourceCatalog
            Source catalogs supplying positions for forced photometry.

        Returns
        -------
        list of MeasuredSource
        """
        return []


class BlindSearchProvider(ABC):
    """Abstract base for blind source-search pipeline steps."""

    @abstractmethod
    def search(
        self,
        input_map: ProcessableMap,
    ) -> tuple[list[MeasuredSource], list[enmap.ndmap]]:
        """Search a map for sources without prior position information.

        Parameters
        ----------
        input_map : ProcessableMap
            Finalized map to search.

        Returns
        -------
        tuple of (list of MeasuredSource, list of enmap.ndmap)
            Detected sources and associated diagnostic maps.
        """
        return [], []
