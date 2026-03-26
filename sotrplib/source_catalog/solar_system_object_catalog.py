"""
Source catalog dependencies.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Literal

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from numpy.typing import NDArray
from pixell import utils as pixell_utils
from pydantic import AwareDatetime
from skyfield.toposlib import GeographicPosition
from structlog import get_logger
from structlog.types import FilteringBoundLogger

from sotrplib.maps.core import ProcessableMap
from sotrplib.solar_system.solar_system import (
    get_sso_ephem_in_map,
    load_jpl_ephem_database,
)
from sotrplib.sources.sources import CrossMatch, RegisteredSource
from sotrplib.utils.utils import angular_separation


class SolarSystemObjectCatalog(ABC):
    @abstractmethod
    def get_sources_in_map(self, input_map: ProcessableMap) -> list[RegisteredSource]:
        """
        Get sources that may be present in a specific map.
        """
        return

    def forced_photometry_sources(
        self, input_map: ProcessableMap
    ) -> list[RegisteredSource]:
        """
        Wrapper for forced photometry sources.
        """
        return

    @abstractmethod
    def source_by_id(self, id) -> RegisteredSource:
        """
        Get the information about a source by its internal ID.
        """
        return

    @abstractmethod
    def get_positions(
        self,
        times: list[datetime],
        source_ids: list | None = None,
    ) -> dict[str, SkyCoord]:
        """
        Get the positions of sources at specific times.

        Parameters
        ----------
        times
            Array of MJD times to get positions for.
        source_ids
            List of source IDs to get positions for. If None, get positions for all sources.

        Returns
        -------
        dict[str, SkyCoord]
            Dictionary mapping source IDs to their SkyCoord positions at the given times.
        """
        return


class SSOCat(SolarSystemObjectCatalog):
    db_path: str
    sso_ephems: dict
    catalog: list[RegisteredSource]
    observer: GeographicPosition
    db: any
    radec_array: NDArray
    start_time: AwareDatetime | None
    stop_time: AwareDatetime | None

    def __init__(
        self,
        db_path: str,
        observer: GeographicPosition,
        start_time: AwareDatetime | None = None,
        stop_time: AwareDatetime | None = None,
        log: FilteringBoundLogger | None = None,
    ):
        self.db_path = db_path
        self.observer = observer
        self.start_time = start_time
        self.stop_time = stop_time
        self.db = load_jpl_ephem_database(
            self.db_path, start_time=self.start_time, stop_time=self.stop_time
        )
        self.log = log or get_logger()
        self.sso_ephems = {}
        self.catalog = []

    def get_sources_in_map(self, input_map: ProcessableMap) -> list[RegisteredSource]:
        # Implementation to get sources in the map
        sso_in_map = get_sso_ephem_in_map(
            input_map,
            ephem_df=self.db,
            observer=self.observer,
        )
        if len(sso_in_map) == 0:
            return []
        out_sources = []
        for sso in sso_in_map:
            if not isinstance(sso_in_map[sso]["pos"], SkyCoord):
                self.log.error(
                    f"Expected position to be a SkyCoord, but got {type(sso_in_map[sso]['pos'])} for source {sso}.",
                    position=sso_in_map[sso]["pos"],
                )
                continue
            sso_ra = np.atleast_1d(sso_in_map[sso]["pos"].ra)
            sso_dec = np.atleast_1d(sso_in_map[sso]["pos"].dec)
            sso_time = np.atleast_1d(sso_in_map[sso]["time"])
            for i in range(len(sso_ra)):
                out_sources.append(
                    RegisteredSource(
                        source_id=sso,
                        source_type="sso",
                        ra=sso_ra[i],
                        dec=sso_dec[i],
                        flux=0.0 * u.mJy,  ## todo: supply estimated fluxes
                        observation_mean_time=sso_time[i],
                        catalog_name="solar_system_catalog",
                        crossmatches=[
                            CrossMatch(
                                ra=sso_ra[i],
                                dec=sso_dec[i],
                                flux=0.0 * u.mJy,  ## todo: supply estimated fluxes
                                source_type="sso",
                                catalog_name="solar_system_catalog",
                                source_id=sso,
                            )
                        ],
                    )
                )
        self.sso_ephems = sso_in_map
        self.catalog = out_sources
        self.ra_dec_array = np.asarray(
            [(x.ra.to("deg").value, x.dec.to("deg").value) for x in out_sources]
        )
        return out_sources

    def forced_photometry_sources(self, input_map) -> list[RegisteredSource]:
        return (
            self.catalog if self.catalog != [] else self.get_sources_in_map(input_map)
        )

    def source_by_id(self, id) -> RegisteredSource:
        # Implementation to get source by ID
        sso_row = self.db[self.db["designation"] == id]
        if sso_row.empty:
            raise ValueError(f"Source ID {id} not found in SSO catalog.")
        sso = sso_row.iloc[0]
        source = RegisteredSource(
            source_id=sso["designation"],
            source_type="sso",
            flux=0.0 * u.mJy,  ## todo: supply estimated fluxes
            catalog_name="solar_system_catalog",
        )
        return source

    def get_positions(
        self,
        times: list[datetime],
        source_ids: list | None = None,
    ) -> dict[str, SkyCoord]:
        # Implementation to get positions of sources at specific times
        from sotrplib.solar_system.solar_system import get_sso_ephems_at_time

        if source_ids is not None:
            ephem_df = self.db[self.db["designation"].isin(source_ids)]
        else:
            ephem_df = self.db

        sso_ephems = get_sso_ephems_at_time(
            ephem_df,
            sample_times=times,
            observer=self.observer,
        )

        positions = {}
        for designation, eph_data in sso_ephems.items():
            positions[designation] = eph_data["pos"]

        return positions

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
                angular_separation=angular_separation(
                    ra1=s.ra, dec1=s.dec, ra2=ra, dec2=dec
                ),
                flux=s.flux,
                err_flux=s.err_flux,
                frequency=s.frequency,
                catalog_name="solar_system_catalog",
                catalog_idx=y[1],
                alternate_names=[],
            )
            for s, y in zip(sources, matches)
        ]
