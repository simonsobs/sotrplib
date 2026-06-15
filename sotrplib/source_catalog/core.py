"""
Source catalog dependencies.
"""

from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
import structlog
from astropy import units as u
from astropy.coordinates import Angle, SkyCoord
from numpy.typing import NDArray
from pixell import utils as pixell_utils
from structlog.types import FilteringBoundLogger

from sotrplib.maps.core import ProcessableMap
from sotrplib.sources.sources import CrossMatch, RegisteredSource
from sotrplib.utils.utils import angular_separation, normalize_ra


class SourceCatalog(ABC):
    """Abstract base for all source catalogs used in the pipeline."""

    @abstractmethod
    def add_sources(self, sources: list[RegisteredSource]):
        """Add sources to the catalog's internal list.

        Parameters
        ----------
        sources : list of RegisteredSource
            Sources to add.
        """
        return

    @abstractmethod
    def get_sources_in_box(
        self, box: list[SkyCoord] | None = None
    ) -> list[RegisteredSource]:
        """Return sources within a sky bounding box.

        Parameters
        ----------
        box : list of SkyCoord, optional
            ``[lower_left, upper_right]`` sky-coordinate corners.  If ``None``,
            return all sources.

        Returns
        -------
        list of RegisteredSource
        """
        return

    @abstractmethod
    def get_sources_in_map(
        self, input_map: ProcessableMap | None = None
    ) -> list[RegisteredSource]:
        """Return sources within the valid (non-zero) region of a map.

        Parameters
        ----------
        input_map : ProcessableMap, optional
            Map used to determine the valid sky footprint.

        Returns
        -------
        list of RegisteredSource
        """
        return

    @abstractmethod
    def forced_photometry_sources(
        self, mask_map: ProcessableMap | None = None
    ) -> list[RegisteredSource]:
        """Return the subset of sources intended for forced photometry.

        Parameters
        ----------
        mask_map : ProcessableMap, optional
            Map used to restrict the source list to the observed region.

        Returns
        -------
        list of RegisteredSource
        """
        return

    @abstractmethod
    def source_by_id(self, id) -> RegisteredSource:
        """Return the source with the given internal ID.

        Parameters
        ----------
        id : int or str
            Internal source identifier.

        Returns
        -------
        RegisteredSource
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
        """Return catalog crossmatches within ``radius`` of a position.

        Parameters
        ----------
        ra : Quantity
            Right ascension of the query position.
        dec : Quantity
            Declination of the query position.
        radius : Quantity
            Search radius.
        method : {"closest", "all"}
            Return only the closest match or all matches within the radius.

        Returns
        -------
        list of CrossMatch
        """
        return


class RegisteredSourceCatalog(SourceCatalog):
    """Source catalog backed by an in-memory list of ``RegisteredSource`` objects.

    Typically used for simulation-injected sources.

    Parameters
    ----------
    sources : list of RegisteredSource
        Initial source list.
    """

    sources: list[RegisteredSource]
    ra_dec_array: NDArray

    def __init__(self, sources: list[RegisteredSource]):
        self.sources = sources
        # Generate the RA, Dec array for crossmatches.
        self.ra_dec_array = np.asarray(
            [(x.ra.to("radian").value, x.dec.to("radian").value) for x in self.sources]
        )

    def add_sources(self, sources: list[RegisteredSource]):
        self.sources.extend(sources)
        self.ra_dec_array = np.asarray(
            [(x.ra.to("radian").value, x.dec.to("radian").value) for x in self.sources]
        )

    def get_sources_in_map(
        self,
        input_map: ProcessableMap | None = None,
        log: FilteringBoundLogger | None = None,
    ) -> list[RegisteredSource]:
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
        self, box: list[SkyCoord] | None = None
    ) -> list[RegisteredSource]:
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
        self, mask_map: ProcessableMap | None = None
    ) -> list[RegisteredSource]:
        return self.get_sources_in_map(mask_map)

    def source_by_id(self, id: int):
        return self.sources[id]

    def crossmatch(
        self,
        ra: u.Quantity,
        dec: u.Quantity,
        radius: u.Quantity,
        method: Literal["closest", "all"],
    ) -> list[CrossMatch]:
        """Return crossmatches within ``radius`` of a position.

        Parameters
        ----------
        ra : Quantity
            Query RA.
        dec : Quantity
            Query Dec.
        radius : Quantity
            Search radius.
        method : {"closest", "all"}
            Return only the closest match or all matches.

        Returns
        -------
        list of CrossMatch
        """

        if len(self.ra_dec_array) == 0:
            return []

        matches = pixell_utils.crossmatch(
            pos1=[[ra.to("radian").value, dec.to("radian").value]],
            pos2=self.ra_dec_array,
            rmax=radius.to("radian").value,
            mode=method,
            coords="radec",
        )

        sources = [self.sources[y] for _, y in matches]

        return [
            CrossMatch(
                ra=s.ra,
                dec=s.dec,
                source_id=str(s.source_id),
                probability=1.0 / len(sources),
                angular_separation=angular_separation(
                    ra1=s.ra, dec1=s.dec, ra2=ra, dec2=dec
                ),
                flux=s.flux,
                err_flux=s.err_flux,
                frequency=s.frequency,
                catalog_name="registered",
                catalog_idx=y[1],
                alternate_names=[],
            )
            for s, y in zip(sources, matches)
        ]
