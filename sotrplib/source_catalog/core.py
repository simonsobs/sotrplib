"""
Source catalog dependencies.
"""

from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
import structlog
import uuid7 as uuid
from astropy import units as u
from astropy.coordinates import Angle, SkyCoord
from astropy.time import Time
from pixell import utils as pixell_utils
from structlog.types import FilteringBoundLogger

from sotrplib.maps.core import ProcessableMap
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
        self, box: tuple[SkyCoord, SkyCoord] | None = None, t: Time | None = None
    ) -> list[RegisteredSource]:
        """
        Get sources that live in a specific sky box.
        Sky box is defined by two corner coordinates, (lower-left, upper-right).
        Time required for moving sources in order to determine if they are within the box.
        """
        return

    @abstractmethod
    def get_sources_in_map(self, input_map: ProcessableMap) -> list[RegisteredSource]:
        """
        Get sources that are within the valid region of a given map.
        """
        return

    @abstractmethod
    def forced_photometry_sources(
        self,
        mask_map: ProcessableMap | None = None,
        t_min: Time | None = None,
        t_max: Time | None = None,
        flux_lower_limit: u.Quantity | None = None,
    ) -> list[RegisteredSource]:
        """
        Get the list of sources to be used for forced photometry.
        Will use mask_map to filter sources.
        flux_lower_limit to filter by flux.
        t_min and t_max can be used to filter by time.
        """
        return

    @abstractmethod
    def source_by_id(self, id: str | uuid.UUID) -> RegisteredSource:
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
        t: Time | None = None,
    ) -> list[CrossMatch]:
        """
        Crossmatch a set of coordinates with the sources in the catalog.
        Optionally include a time for moving sources.
        """
        return


class RegisteredSourceCatalog(SourceCatalog):
    """
    A source catalog generated purely from a list of 'registered sources',
    usually those that are generated as part of the simulation process.
    """

    sources: list[RegisteredSource]

    def __init__(self, sources: list[RegisteredSource]):
        self.sources = sources

    def add_sources(self, sources: list[RegisteredSource]):
        self.sources.extend(sources)

    def get_sources_in_map(
        self,
        input_map: ProcessableMap,
        log: FilteringBoundLogger | None = None,
    ) -> list[RegisteredSource]:
        """
        Get sources that are within the valid region of a given map.
        """
        log = log or structlog.get_logger()
        if len(self.sources) == 0:
            return []
        coords = SkyCoord(
            ra=[s.ra for s in self.sources], dec=[s.dec for s in self.sources]
        )
        inside, _ = input_map.filter_sources(coords)
        if not np.any(inside):
            log.warning(
                "get_sources_in_map.no_sources_in_map",
                map_id=input_map.map_id,
                wcs=input_map.flux.wcs,
            )
            return []

        return [self.sources[i] for i in np.where(inside)[0]]

    def get_sources_in_box(
        self, box: tuple[SkyCoord, SkyCoord] | None = None
    ) -> list[RegisteredSource]:
        """
        Get sources that are within a given sky box.
        If no box is provided, all sources are returned.
        """
        if box is None:
            return self.sources
        ## Get sky coord limits and whether ra is -180,180 or 0,360
        ra_corners, box_ra_wrap = normalize_ra(Angle([box[0].ra, box[1].ra]))
        source_ra_wrap = (
            "-180-180" if any(s.ra < 0.0 * u.deg for s in self.sources) else "0-360"
        )
        source_ras = np.asarray([s.ra.to_value(u.deg) for s in self.sources]) * u.deg
        if source_ra_wrap != box_ra_wrap:
            ## shift source RAs to match box system
            if box_ra_wrap == "0-360":
                source_ras[source_ras < 0.0 * u.deg] += 360.0 * u.deg
            else:
                source_ras[source_ras > 180.0 * u.deg] -= 360.0 * u.deg

        ## Check for RA wrap-around
        ra_min, ra_max = np.min(Angle(ra_corners)), np.max(Angle(ra_corners))

        ra_span = ra_max - ra_min
        if ra_span > 180.0 * u.deg:
            wrap_warning = True
        else:
            wrap_warning = False

        ## If wrapped, shift RA system so box is continuous
        if wrap_warning:
            # shift RA so that region is continuous
            shift = 180.0 * u.deg - ra_max
            ra_corners = (ra_corners + shift + 360.0 * u.deg) % (360.0 * u.deg)
            source_ras = (source_ras + shift + 360.0 * u.deg) % (360.0 * u.deg)
            ra_min, ra_max = np.min(Angle(ra_corners)), np.max(Angle(ra_corners))
        dec_min = np.min(Angle([box[0].dec, box[1].dec]))
        dec_max = np.max(Angle([box[0].dec, box[1].dec]))

        return [
            x
            for idx, x in enumerate(self.sources)
            if (ra_min < source_ras[idx] < ra_max) and (dec_min < x.dec < dec_max)
        ]

    def forced_photometry_sources(
        self, mask_map: ProcessableMap, flux_lower_limit: u.Quantity | None = None
    ) -> list[RegisteredSource]:
        """
        Get sources that are within the valid region of a given map.
        Optionally filter by a lower limit on the source flux.
        """
        sources = self.get_sources_in_map(mask_map)
        if flux_lower_limit is not None:
            sources = [s for s in sources if s.flux >= flux_lower_limit]
        return sources

    def source_by_id(self, id: str | uuid.UUID) -> RegisteredSource:
        source_ids = [s.source_id for s in self.sources]
        if id not in source_ids:
            return None
        idx = source_ids.index(id)
        return self.sources[idx]

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
        if len(self.sources) == 0:
            return []

        catalog_positions = np.array(
            [(s.ra.to("radian").value, s.dec.to("radian").value) for s in self.sources]
        )
        matches = pixell_utils.crossmatch(
            pos1=[[ra.to("radian").value, dec.to("radian").value]],
            pos2=catalog_positions,
            rmax=radius.to("radian").value,
            mode=method,
            coords="radec",
        )

        sources = [self.sources[y] for _, y in matches]

        return [
            CrossMatch(
                ra=s.ra,
                dec=s.dec,
                source_id=s.source_id,
                probability=1.0 / len(sources),
                angular_separation=angular_separation(
                    ra1=s.ra, dec1=s.dec, ra2=ra, dec2=dec
                ),
                flux=s.flux,
                err_flux=s.err_flux,
                frequency=s.frequency,
                catalog_name="registered",
                catalog_idx=s.source_id,
                alternate_names=[],
            )
            for s, _ in zip(sources, matches)
        ]
