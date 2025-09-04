"""
Source catalog dependencies.
"""

from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
from astropy import units as u
from astropydantic import AstroPydanticQuantity
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
        ra: AstroPydanticQuantity[u.deg],
        dec: AstroPydanticQuantity[u.deg],
        radius: AstroPydanticQuantity[u.arcmin],
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
        self.ra_dec_array = np.asarray(
            [(x.ra.to("deg").value, x.dec.to("deg").value) for x in self.sources]
        )

    def source_by_id(self, id: int):
        return self.sources[id]

    def crossmatch(
        self,
        ra: AstroPydanticQuantity[u.deg],
        dec: AstroPydanticQuantity[u.deg],
        radius: AstroPydanticQuantity[u.arcmin],
        method: Literal["closest", "all"],
    ) -> list[RegisteredSource]:
        """
        Get sources within radius of the catalog.
        """

        matches = pixell_utils.crossmatch(
            pos1=[[ra.to("deg").value, dec.to("deg").value]],
            pos2=self.ra_dec_array,
            rmax=radius.to("arcmin").value,
            mode=method,
        )

        return [self.sources[y] for _, y in matches]
