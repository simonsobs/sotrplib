"""
Sifter components.
"""

import abc

from ..data.sources import PotentialSource, SiftedSource, RemovedSource


class AbstractSifter(abc.ABC):
    """
    Abstract class for sifters.
    """

    @abc.abstractmethod
    def sift(self, potential_sources: list[PotentialSource]) -> list[SiftedSource | RemovedSource]:
        """
        Sifts potential sources.

        Parameters
        ----------
        potential_sources : list[PotentialSource]
            The potential sources to sift.
        
        Returns
        -------
        list[SiftedSource | RemovedSource]
            The sifted sources, classified into SiftedSource (valid) and
            RemovedSource (invalid).
        """

        raise NotImplementedError