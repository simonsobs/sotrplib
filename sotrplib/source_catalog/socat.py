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
from socat.database import SolarSystemObject
from structlog.types import FilteringBoundLogger

from sotrplib.maps.core import ProcessableMap
from sotrplib.sources.sources import CrossMatch, RegisteredSource
from sotrplib.utils.utils import angular_separation

from .core import SourceCatalog


class SOCat(SourceCatalog):
    """
    A catalog backed by the configured SOCat database.

    Queries both fixed and solar-system-object (moving) sources. SSO queries
    require t_min and t_max so that ephemeris interpolation is bounded.
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
        self.log.info(
            "socat.initialized",
            t_min=self.t_min,
            t_max=self.t_max,
            flux_lower_limit=self.flux_lower_limit,
        )

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
            flux=source.flux,
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

    def _get_obs_time(
        self, position: SkyCoord, input_map: ProcessableMap
    ) -> Time | None:
        """Return the per-pixel mean observation time at a sky position, or None on failure."""
        try:
            pix = input_map.hits.sky2pix(
                np.array([position.dec.to_value("rad"), position.ra.to_value("rad")])
            )
            _, t_pixel, _ = input_map.get_pixel_times(pix)
            if isinstance(t_pixel, (int, float, np.floating)):
                return Time(float(t_pixel), format="unix") if t_pixel > 0 else None
            return Time(t_pixel)
        except (IndexError, ValueError):
            return None

    def _sg_to_registered_with_refinement(
        self, sg, t_initial: Time, input_map: ProcessableMap
    ) -> RegisteredSource | None:
        """
        Two-step SSO position: initial position query at t_initial, then refined at per-pixel time.

        For a moving source, we query the position at the mid point of the observation,
        then use that position to find the time at which that pixel was observed, then
        query the position again using that updated time.

        Main belt asteroids have apparent motion of up to several arcmin per hour.
        For a depth1 map that spans 12-24 hours, we therefore might expect up to a degree or two.
        The time refinement in depth1 map over 1-2 degrees is something like 10-20 minutes for an OT.
        This reduces the positional uncertainty to <~1 arcmin.
        """
        initial = self._sg_to_registered(sg, t_initial)
        if initial is None or not isinstance(sg.source, SolarSystemObject):
            return initial
        t = self._get_obs_time(SkyCoord(ra=initial.ra, dec=initial.dec), input_map)
        if t is None:
            return initial
        refined = self._sg_to_registered(sg, t)
        return refined if refined is not None else initial

    def _sg_to_registered(self, sg, t: Time) -> RegisteredSource | None:
        try:
            sg.init_interp(ephem_cat=self.catalog.ephem)
            position, flux = sg.at_time(t=t)
        except (AttributeError, ValueError):
            return None
        is_sso = isinstance(sg.source, SolarSystemObject)
        if is_sso:
            source_type = "sso"
        elif hasattr(sg.source, "source_type"):
            source_type = sg.source.source_type
        else:
            source_type = "unknown"
        catalog_idx = sg.source.sso_id if is_sso else sg.source.source_id
        return RegisteredSource(
            ra=position.ra,
            dec=position.dec,
            flux=flux,
            source_id=sg.source.name,
            source_type=source_type,
            observation_mean_time=t,
            crossmatches=[
                CrossMatch(
                    source_id=sg.source.name,
                    source_type=source_type,
                    probability=1.0,
                    angular_separation=0.0 * u.deg,
                    catalog_name="socat",
                    catalog_idx=catalog_idx,
                    observation_time=t,
                )
            ],
        )

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

        if self.t_min is None or self.t_max is None:
            sources = [
                self._fixed_to_registered(s)
                for s in self.catalog.get_box_fixed(
                    lower_left=box[0], upper_right=box[1]
                )
            ]
        else:
            t_eval = self.t_min + (self.t_max - self.t_min) / 2
            sources = [
                s
                for s in (
                    self._sg_to_registered(sg, t_eval)
                    for sg in self.catalog.sso.get_box(
                        lower_left=box[0],
                        upper_right=box[1],
                        t_min=self.t_min,
                        t_max=self.t_max,
                        source_cat=self.catalog,
                        ephem_cat=self.catalog.ephem,
                    )
                )
                if s is not None
            ]

        self.log.debug("socat.get_sources_in_box", box=box, n_sources=len(sources))
        return sources

    def get_sources_in_map(self, input_map: ProcessableMap) -> list[RegisteredSource]:
        ## for now, just get all the sources and filter them.
        sky_box = [
            SkyCoord(ra=0.0 * u.deg, dec=-90.0 * u.deg),
            SkyCoord(ra=359.999 * u.deg, dec=90.0 * u.deg),
        ]
        if self.t_min is not None and self.t_max is not None:
            t_eval = self.t_min + (self.t_max - self.t_min) / 2
            all_sources = [
                s
                for s in (
                    self._sg_to_registered_with_refinement(sg, t_eval, input_map)
                    for sg in self.catalog.sso.get_box(
                        lower_left=sky_box[0],
                        upper_right=sky_box[1],
                        t_min=self.t_min,
                        t_max=self.t_max,
                        source_cat=self.catalog,
                        ephem_cat=self.catalog.ephem,
                    )
                )
                if s is not None
            ]
        else:
            self.log.warning(
                "socat.get_sources_in_map.no_time_range",
                map_id=input_map.map_id,
                observation_time=input_map.observation_time,
            )
            all_sources = self.get_sources_in_box(box=None)

        if not all_sources:
            self.log.warning(
                "socat.get_sources_in_map.no_sources_in_box",
                map_id=input_map.map_id,
                sky_box=sky_box,
                wcs=input_map.flux.wcs,
            )
            return []
        coords = SkyCoord(
            ra=[s.ra for s in all_sources], dec=[s.dec for s in all_sources]
        )
        self.log.info(
            "socat.get_sources_in_map.sources_in_bounding_box",
            bbox=sky_box,
            map_id=input_map.map_id,
            n_sources=len(all_sources),
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
        self, input_map: ProcessableMap, use_sso: bool = True
    ) -> list[RegisteredSource]:
        obs_time = Time(input_map.observation_time)
        self.log.bind(
            func="socat.forced_photometry_sources",
            obs_time=obs_time,
            t_min=self.t_min,
            t_max=self.t_max,
            use_sso=use_sso,
        )
        if self.t_min is None or self.t_max is None or not use_sso:
            self.log.info(
                "socat.forced_photometry_sources.retrieving_sources_fixed_only"
            )
            sources = [
                self._fixed_to_registered(s)
                for s in self.catalog.get_forced_photometry_sources(
                    minimum_flux=self.flux_lower_limit
                )
            ]
        else:
            self.log.info("socat.forced_photometry_sources.retrieving_sources_with_sso")
            candidates = self.get_sources_in_map(input_map)
            sources = [
                s
                for s in candidates
                if s is not None
                and s.flux is not None
                and s.flux >= self.flux_lower_limit
            ]

        self.log.info("socat.forced_photometry_sources", n_sources=len(sources))
        if not sources:
            return []
        coords = SkyCoord(ra=[s.ra for s in sources], dec=[s.dec for s in sources])
        inside, _ = input_map.filter_sources(coords)
        if not np.any(inside):
            self.log.warning(
                "socat.forced_photometry_sources.no_sources_in_map",
                n_sources=len(sources),
                map_id=input_map.map_id,
                wcs=input_map.flux.wcs,
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
