"""
Blind source detection and extraction.
"""

import abc

from ..data.maps import MapBase
from ..data.sources import PotentialSource

class AbstractBlind(abc.ABC):
    """
    Abstract class for blind photometry.
    """

    @abc.abstractmethod
    def prepare(self, maps: list[MapBase]):
        """
        Prepares the blind photometry for processing; this is called before running the
        ``blind`` method.

        Parameters
        ----------
        maps : list[MapBase]
            The maps to extract sources from.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def blind(self) -> list[PotentialSource]:
        """
        Blindly extracts sources from maps.

        Returns
        -------
        list[PotentialSource]
            The potential sources extracted.
        """
        raise NotImplementedError