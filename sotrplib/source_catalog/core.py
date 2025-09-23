"""
Source catalog dependencies.
"""

from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from numpy.typing import NDArray
from pixell import utils as pixell_utils

from sotrplib.sources.sources import CrossMatch, RegisteredSource
from sotrplib.utils.utils import angular_separation


class SourceCatalog(ABC):
    @abstractmethod
    def add_sources(self, sources: list[RegisteredSource]):
        """
        Add sources to the main internal list for searching.
        """
        return

    @abstractmethod
    def sources_in_box(
        self, box: list[SkyCoord] | None = None
    ) -> list[RegisteredSource]:
        """
        Get sources that live in a specific box on the sky.
        """
        return

    @abstractmethod
    def forced_photometry_sources(
        self, box: list[SkyCoord] | None = None
    ) -> list[RegisteredSource]:
        """
        Get the list of sources to be used for forced photometry
        """
        return

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
    ) -> list[CrossMatch]:
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

    def add_sources(self, sources: list[RegisteredSource]):
        self.sources.extend(sources)

    def sources_in_box(
        self, box: list[SkyCoord] | None = None
    ) -> list[RegisteredSource]:
        if box is None:
            return self.sources

        left = box[0].ra
        right = box[1].ra
        bottom = box[0].dec
        top = box[1].dec

        return [
            x for x in self.sources if (left < x.ra < right) and (bottom < x.dec < top)
        ]

    def forced_photometry_sources(
        self, box: list[SkyCoord] | None = None
    ) -> list[RegisteredSource]:
        return self.sources_in_box(box=box)

    def source_by_id(self, id: int):
        return self.sources[id]

    def crossmatch(
        self,
        ra: u.Quantity,
        dec: u.Quantity,
        radius: u.Quantity,
        method: Literal["closest", "all"],
    ) -> list[CrossMatch]:
        """
        Get sources within radius of the catalog.
        """

        matches = pixell_utils.crossmatch(
            pos1=[[ra.to("deg").value, dec.to("deg").value]],
            pos2=self.ra_dec_array,
            rmax=radius.to("arcmin").value,
            mode=method,
        )

        sources = [self.sources[y] for _, y in matches]

        return [
            CrossMatch(
                source_id=str(s.source_id),
                probability=1.0 / len(sources),
                distance=angular_separation(s.ra, ra, s.dec, dec),
                flux=s.flux,
                err_flux=s.err_flux,
                frequency=s.frequency,
                catalog_name="registered",
                catalog_idx=y[1],
                alternate_names=[],
            )
            for s, y in zip(sources, matches)
        ]
