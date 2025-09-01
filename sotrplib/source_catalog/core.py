"""
Source catalog dependencies.
"""

from abc import ABC, abstractmethod
from typing import Literal

import astropy.units as u
import numpy as np
from numpy.typing import NDArray
from pixell import utils as pixell_utils

from sotrplib.sources.sources import RegisteredSource


class SourceCatalog(ABC):
    @abstractmethod
    def source_by_id(self, id) -> RegisteredSource:
        """
        Get the information about a source by its internal ID.
        """
        return

    @abstractmethod
    def crossmatch(
        self,
        ra: u.Quantity,
        dec: u.Quantity,
        radius: u.Quantity,
        method: Literal["closest", "all"],
    ) -> list[RegisteredSource]:
        return


class RegisteredSourceCatalog(SourceCatalog):
    """
    A source catalog generated purely from a list of 'registered sources',
    usually those that are generated as part of the simulation process.
    """

    sources: list[RegisteredSource]
    ra_dec_array: NDArray

    def __init__(self, sources: list[RegisteredSource]):
        self.sources = sources

        # Generate the RA, Dec array for crossmatches.
        self.ra_dec_array = np.asarray([(x.ra, x.dec) for x in self.sources])

    def source_by_id(self, id: int):
        return self.sources[id]

    def crossmatch(
        self,
        ra: u.Quantity,
        dec: u.Quantity,
        radius: u.Quantity,
        method: Literal["closest", "all"],
    ) -> list[RegisteredSource]:
        """
        Get sources within radius of the catalog.
        """

        matches = pixell_utils.crossmatch(
            pos1=[[ra.to_value("deg"), dec.to_value("deg")]],
            pos2=self.ra_dec_array,
            rmax=radius.to_value("arcmin"),
            mode=method,
        )

        return [self.sources[y] for _, y in matches]
