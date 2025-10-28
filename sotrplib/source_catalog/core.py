"""
Source catalog dependencies.
"""

from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
from astropy import units as u
from astropy.coordinates import Angle, SkyCoord
from numpy.typing import NDArray
from pixell import utils as pixell_utils

from sotrplib.sources.sources import CrossMatch, RegisteredSource
from sotrplib.utils.utils import angular_separation, normalize_ra


class SourceCatalog(ABC):
    @abstractmethod
    def add_sources(self, sources: list[RegisteredSource]):
        """
        Add sources to the main internal list for searching.
        """
        return

    @abstractmethod
    def get_sources_in_box(
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

    def get_sources_in_box(
        self, box: list[SkyCoord] | None = None
    ) -> list[RegisteredSource]:
        if box is None:
            return self.sources
        ## Get sky coord limits and whether ra is -180,180 or 0,360
        ra_corners, box_ra_wrap = normalize_ra(Angle([box[0].ra, box[1].ra]))
        source_ra_wrap = (
            "-180-180" if any(s.ra < 0 * u.deg for s in self.sources) else "0-360"
        )
        source_ras = np.asarray([s.ra.to_value(u.deg) for s in self.sources])
        if source_ra_wrap != box_ra_wrap:
            ## shift source RAs to match box system
            if box_ra_wrap == "0-360":
                source_ras[source_ras < 0] += 360
            else:
                source_ras[source_ras > 180] -= 360

        ## Check for RA wrap-around
        ra_min, ra_max = np.min(Angle(ra_corners)), np.max(Angle(ra_corners))

        ra_span = ra_max - ra_min
        if ra_span > 180 * u.deg:
            wrap_warning = True
        else:
            wrap_warning = False

        ## If wrapped, shift RA system so box is continuous
        if wrap_warning:
            # shift RA so that region is continuous
            shift = 180 * u.deg - ra_max
            ra_corners = (ra_corners + shift + 360 * u.deg) % (360 * u.deg)
            source_ras = (source_ras + shift + 360) % 360
            ra_min, ra_max = np.min(Angle(ra_corners)), np.max(Angle(ra_corners))
        dec_min = np.min(Angle([box[0].dec, box[1].dec]))
        dec_max = np.max(Angle([box[0].dec, box[1].dec]))

        return [
            x
            for idx, x in enumerate(self.sources)
            if (ra_min < source_ras[idx] * u.deg < ra_max)
            and (dec_min < x.dec < dec_max)
        ]

    def forced_photometry_sources(
        self, box: list[SkyCoord] | None = None
    ) -> list[RegisteredSource]:
        return self.get_sources_in_box(box=box)

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
