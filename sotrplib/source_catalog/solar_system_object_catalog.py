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
from skyfield.api import wgs84

from sotrplib.maps.core import ProcessableMap
from sotrplib.solar_system.solar_system import get_sso_in_map, load_mpc_orbital_database
from sotrplib.sources.sources import CrossMatch, RegisteredSource
from sotrplib.utils.utils import angular_separation, datetime_mean


class SolarSystemObjectCatalog(ABC):
    @abstractmethod
    def get_sources_in_map(self, input_map: ProcessableMap) -> list[RegisteredSource]:
        """
        Get sources that may be present in a specific map.
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
    observer: wgs84.latlon
    db: any
    radec_array: NDArray

    def __init__(
        self,
        db_path: str,
        observer: wgs84.latlon,
    ):
        self.db_path = db_path
        self.db = load_mpc_orbital_database(self.db_path)
        self.observer = observer
        self.sso_ephems = {}
        self.catalog = []

    def get_sources_in_map(self, input_map: ProcessableMap) -> list[RegisteredSource]:
        # Implementation to get sources in the map
        sso_in_map = get_sso_in_map(
            input_map,
            orbital_df=self.db,
            observer=self.observer,
        )
        if len(sso_in_map) == 0:
            return []
        out_sources = []
        for sso in sso_in_map:
            out_sources.append(
                RegisteredSource(
                    source_id=sso,
                    source_type="sso",
                    ra=np.mean(sso_in_map[sso]["pos"].ra),
                    dec=np.mean(sso_in_map[sso]["pos"].dec),
                    observation_mean_time=datetime_mean(sso_in_map[sso]["time"]),
                    crossmatches=[
                        CrossMatch(
                            ra=np.mean(sso_in_map[sso]["pos"].ra),
                            dec=np.mean(sso_in_map[sso]["pos"].dec),
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

    def source_by_id(self, id) -> RegisteredSource:
        # Implementation to get source by ID
        sso_row = self.db[self.db["designation"] == id]
        if sso_row.empty:
            raise ValueError(f"Source ID {id} not found in SSO catalog.")
        sso = sso_row.iloc[0]
        source = RegisteredSource(
            source_id=sso["designation"],
            source_type="sso",
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
            orbital_df = self.db[self.db["designation"].isin(source_ids)]
        else:
            orbital_df = self.db

        sso_ephems = get_sso_ephems_at_time(
            orbital_df,
            time=times,
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
                catalog_name="sso",
                catalog_idx=y[1],
                alternate_names=[],
            )
            for s, y in zip(sources, matches)
        ]
