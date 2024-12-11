"""
Forced photometry processing module.
"""

import abc

from ..data.sources import PotentialSource
from ..data.catalog import CatalogSource
from ..data.maps import MapBase

class AbstractForcer(abc.ABC):
    """
    Abstract class for forced photometry.
    """

    @abc.abstractmethod
    def prepare(self, maps: list[MapBase]):
        """
        Prepares the forcer for processing; this is called before running the
        ``force`` method.

        Parameters
        ----------
        maps : list[MapBase]
            The maps to extract sources from.
        """
        raise NotImplementedError
    
    
    @abc.abstractmethod
    def force(self, catalog: list[CatalogSource]) -> list[PotentialSource]:
        """
        Forces photometry on sources.

        Parameters
        ----------
        catalogs: list[CatalogSource]
            The catalog sources to force photometry on.

        Returns
        -------
        list[PotentialSource]
            The potential sources with forced photometry.
        """
        raise NotImplementedError



