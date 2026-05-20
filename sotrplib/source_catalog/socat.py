"""
Use a SOCat database as a source catalog.
"""

from typing import Literal

import numpy as np
import structlog
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from pixell import utils as pixell_utils
from socat.client.settings import SOCatClientSettings
from socat.database import RegisteredMovingSourceTable
from sqlalchemy import select
from structlog.types import FilteringBoundLogger

from sotrplib.maps.core import ProcessableMap
from sotrplib.sources.sources import CrossMatch, RegisteredSource
from sotrplib.utils.utils import angular_separation

from .core import SourceCatalog


class SOCat(SourceCatalog):
    """
    A catalog implementation backed by the configured SOCat database.

    Queries both fixed and solar-system-object (moving) sources. SSO queries
    require t_min and t_max so that ephemeris points in the time window are
    returned alongside static sources.
    """

    def __init__(
        self,
        t_min: Time | None = None,
        t_max: Time | None = None,
        flux_lower_limit: u.Quantity = 0.03 * u.Jy,
        additional_positions: list[SkyCoord] | None = None,
        additional_fluxes: list[u.Quantity] | None = None,
        additional_source_ids: list[str] | None = None,
        log: FilteringBoundLogger | None = None,
    ):
        self.log = log or structlog.get_logger()
        self.catalog = SOCatClientSettings().client
        self.t_min = t_min
        self.t_max = t_max
        self.flux_lower_limit = flux_lower_limit
        self.log.info("socat.initialized")

        if additional_positions is not None:
            sources = []
            for i, pos in enumerate(additional_positions):
                f = additional_fluxes[i] if additional_fluxes is not None else None
                sid = (
                    additional_source_ids[i]
                    if additional_source_ids is not None
                    else None
                )
                sources.append(
                    RegisteredSource(ra=pos.ra, dec=pos.dec, flux=f, source_id=sid)
                )
            self.add_sources(sources)

    def _fixed_to_registered(self, source) -> RegisteredSource:
        return RegisteredSource(
            ra=source.position.ra,
            dec=source.position.dec,
            flux=source.flux if source.flux is not None else None,
            source_id=source.name,
            crossmatches=[
                CrossMatch(
                    source_id=source.name,
                    probability=1.0,
                    angular_separation=0.0 * u.deg,
                    frequency=90.0 * u.GHz,
                    catalog_name="socat",
                    catalog_idx=source.source_id,
                )
            ],
        )

    def _ephem_to_registered(self, ephem) -> RegisteredSource:
        return RegisteredSource(
            ra=ephem.position.ra,
            dec=ephem.position.dec,
            flux=ephem.flux,
            source_id=ephem.name,
            source_type="sso",
            observation_mean_time=ephem.time,
            crossmatches=[
                CrossMatch(
                    source_id=ephem.name,
                    source_type="sso",
                    probability=1.0,
                    angular_separation=0.0 * u.deg,
                    catalog_name="socat",
                    catalog_idx=ephem.ephem_id,
                    observation_time=ephem.time,
                )
            ],
        )

    def _get_sso_in_box(
        self, lower_left: SkyCoord, upper_right: SkyCoord
    ) -> list[RegisteredSource]:
        if self.t_min is None or self.t_max is None:
            return []
        with self.catalog.ephem._get_session() as session:
            rows = (
                session.execute(
                    select(RegisteredMovingSourceTable).where(
                        self.t_min.datetime <= RegisteredMovingSourceTable.time,
                        RegisteredMovingSourceTable.time <= self.t_max.datetime,
                        float(lower_left.ra.to_value("deg"))
                        <= RegisteredMovingSourceTable.ra_deg,
                        RegisteredMovingSourceTable.ra_deg
                        <= float(upper_right.ra.to_value("deg")),
                        float(lower_left.dec.to_value("deg"))
                        <= RegisteredMovingSourceTable.dec_deg,
                        RegisteredMovingSourceTable.dec_deg
                        <= float(upper_right.dec.to_value("deg")),
                    )
                )
                .scalars()
                .all()
            )
        return [self._ephem_to_registered(r.to_model()) for r in rows]

    def _get_sso_at_time(
        self, obs_time: Time, mask_map: ProcessableMap | None = None
    ) -> list[RegisteredSource]:
        """
        For each SSO with a defined flux, return the ephemeris point closest to
        obs_time within the catalog's t_min/t_max window.

        If mask_map has a time_mean map, the initial position estimate is used to
        look up the per-pixel observation time at that location, and the ephemeris
        is re-evaluated at that more accurate time.
        """
        if self.t_min is None or self.t_max is None:
            return []
        with self.catalog.ephem._get_session() as session:
            rows = (
                session.execute(
                    select(RegisteredMovingSourceTable).where(
                        self.t_min.datetime <= RegisteredMovingSourceTable.time,
                        RegisteredMovingSourceTable.time <= self.t_max.datetime,
                        RegisteredMovingSourceTable.flux_mJy.isnot(None),
                    )
                )
                .scalars()
                .all()
            )
        if not rows:
            return []

        by_sso: dict[int, list] = {}
        for row in rows:
            by_sso.setdefault(row.sso_id, []).append(row)

        def closest_row(sso_rows, t: Time):
            dt = t.datetime
            return min(sso_rows, key=lambda r: abs((r.time - dt).total_seconds()))

        if mask_map is None or mask_map.time_mean is None:
            return [
                self._ephem_to_registered(closest_row(rows, obs_time).to_model())
                for rows in by_sso.values()
            ]

        refined = []
        for sso_rows in by_sso.values():
            initial = closest_row(sso_rows, obs_time)
            t_unix = mask_map.time_mean.at(
                (np.deg2rad(initial.dec_deg), np.deg2rad(initial.ra_deg)),
                mode="nn",
            )
            if np.isfinite(t_unix) and t_unix > 0:
                pixel_time = Time(float(t_unix), format="unix")
                best = closest_row(sso_rows, pixel_time)
            else:
                best = initial
            refined.append(best)

        return [self._ephem_to_registered(r.to_model()) for r in refined]

    def add_sources(self, sources: list[RegisteredSource]):
        for source in sources:
            self.catalog.create_source(
                position=SkyCoord(source.ra, source.dec),
                flux=source.flux,
                name=source.source_id,
            )

    ## TODO: need to update for boxes which need to be broken in two.
    def get_sources_in_box(
        self, box: list[SkyCoord] | None = None
    ) -> list[RegisteredSource]:
        if box is None:
            box = [
                SkyCoord(ra=0.0 * u.deg, dec=-90.0 * u.deg),
                SkyCoord(ra=359.999 * u.deg, dec=90.0 * u.deg),
            ]
        if box[0].ra > box[1].ra and box[0].dec < box[1].dec:
            box = [box[1], box[0]]

        fixed = [
            self._fixed_to_registered(s)
            for s in self.catalog.get_box_fixed(lower_left=box[0], upper_right=box[1])
        ]
        ssos = self._get_sso_in_box(lower_left=box[0], upper_right=box[1])

        self.log.debug(
            "socat.get_sources_in_box",
            box=box,
            n_fixed=len(fixed),
            n_sso=len(ssos),
        )
        return fixed + ssos

    def get_sources_in_map(self, input_map: ProcessableMap) -> list[RegisteredSource]:
        all_sources = self.get_sources_in_box(box=None)
        if not all_sources:
            return []
        coords = SkyCoord(
            ra=[s.ra for s in all_sources], dec=[s.dec for s in all_sources]
        )
        inside, _ = input_map.filter_sources(coords)
        if not np.any(inside):
            self.log.warning(
                "socat.get_sources_in_map.no_sources_in_map",
                map_id=input_map.map_id,
                wcs=input_map.flux.wcs,
            )
            return []
        return [all_sources[i] for i in range(len(all_sources)) if inside[i]]

    def get_all_sources(self) -> list[RegisteredSource]:
        return self.get_sources_in_box(box=None)

    def forced_photometry_sources(
        self, mask_map: ProcessableMap
    ) -> list[RegisteredSource]:
        obs_time = Time(mask_map.observation_time)
        fixed = [
            self._fixed_to_registered(s)
            for s in self.catalog.get_forced_photometry_sources(
                minimum_flux=self.flux_lower_limit
            )
        ]
        ssos = self._get_sso_at_time(obs_time, mask_map=mask_map)
        print(ssos)
        sources = fixed + ssos
        self.log.info(
            "socat.forced_photometry_sources",
            n_fixed=len(fixed),
            n_ssos=len(ssos),
            n_total=len(sources),
        )
        if not sources:
            return []
        coords = SkyCoord(ra=[s.ra for s in sources], dec=[s.dec for s in sources])
        inside, _ = mask_map.filter_sources(coords)
        if not np.any(inside):
            self.log.warning(
                "socat.forced_photometry_sources.no_sources_in_map",
                n_sources=len(sources),
                map_id=mask_map.map_id,
                wcs=mask_map.flux.wcs,
            )
        return [sources[i] for i in range(len(sources)) if inside[i]]

    def source_by_id(self, id) -> RegisteredSource:
        return self._fixed_to_registered(self.catalog.get_source(source_id=id))

    def crossmatch(
        self,
        ra: u.Quantity,
        dec: u.Quantity,
        radius: u.Quantity,
        method: Literal["closest", "all"],
    ) -> list[CrossMatch]:
        close_sources = self.get_sources_in_box(
            [
                SkyCoord(ra=ra - 2.0 * radius, dec=dec - 2.0 * radius),
                SkyCoord(ra=ra + 2.0 * radius, dec=dec + 2.0 * radius),
            ]
        )
        ra_dec_array = np.asarray(
            [(x.ra.to("radian").value, x.dec.to("radian").value) for x in close_sources]
        )
        if len(ra_dec_array) == 0:
            return []
        matches = pixell_utils.crossmatch(
            pos1=[[ra.to("radian").value, dec.to("radian").value]],
            pos2=ra_dec_array,
            rmax=radius.to("radian").value,
            mode=method,
            coords="radec",
        )
        sources = [close_sources[y] for _, y in matches]
        return [
            CrossMatch(
                source_id=s.source_id,
                source_type=s.source_type,
                probability=1.0 / len(sources),
                angular_separation=angular_separation(s.ra, s.dec, ra, dec),
                flux=s.flux,
                err_flux=s.err_flux,
                frequency=s.frequency,
                catalog_name=s.catalog_name,
                catalog_idx=y,
                alternate_names=s.alternate_names,
                ra=s.ra,
                dec=s.dec,
                observation_time=s.observation_mean_time,
            )
            for y, s in enumerate(sources)
        ]
