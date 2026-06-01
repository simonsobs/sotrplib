"""
Use a SOCat database as a source catalog.
"""

from typing import Literal

import numpy as np
import structlog
import tqdm
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time, TimeDelta
from pixell import utils as pixell_utils
from socat.client.settings import SOCatClientSettings
from socat.database import SolarSystemObject
from structlog.types import FilteringBoundLogger

from sotrplib.maps.core import ProcessableMap
from sotrplib.sources.sources import CrossMatch, RegisteredSource
from sotrplib.utils.utils import angular_separation

from .core import SourceCatalog

_TIME_RANGE_PADDING = TimeDelta(1, format="jd")


class SOCat(SourceCatalog):
    """
    A catalog backed by the configured SOCat database.

    Queries both fixed and solar-system-object (moving) sources.
    SSO queries require t_min and t_max so that ephemeris interpolation
    is bounded.

    t_min and t_max can be set manually or automatically generated from
    the maps.

    flux_lower_limit can be set to filter sources for forced photometry,
    but if None, socat will return all monitored sources

    """

    def __init__(
        self,
        t_min: Time | None = None,
        t_max: Time | None = None,
        flux_lower_limit: u.Quantity | None = None,
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

        self._interp_cache: dict = {}

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

    def set_time_range(
        self, t_min: Time, t_max: Time, time_padding: TimeDelta = _TIME_RANGE_PADDING
    ) -> None:
        """
        Set the time range from the map database and clear the interpolation cache.
        The time range is expanded by time_padding on either side to avoid edge effects.
        """
        self._refresh(t_min=t_min - time_padding, t_max=t_max + time_padding)

    def _check_and_expand_interp_range(self, input_map: ProcessableMap) -> None:
        """
        If the map's observation window falls outside [t_min, t_max], expand and refresh.
        """
        obs_start = Time(input_map.observation_start)
        obs_end = Time(input_map.observation_end)
        new_t_min = self.t_min
        new_t_max = self.t_max
        needs_refresh = False
        if self.t_min is None or obs_start < self.t_min:
            new_t_min = obs_start - _TIME_RANGE_PADDING
            needs_refresh = True
        if self.t_max is None or obs_end > self.t_max:
            new_t_max = obs_end + _TIME_RANGE_PADDING
            needs_refresh = True
        if needs_refresh:
            self.log.info(
                "socat.expanding_time_range",
                map_id=input_map.map_id,
                old_t_min=self.t_min,
                old_t_max=self.t_max,
                new_t_min=new_t_min,
                new_t_max=new_t_max,
            )
            self._refresh(t_min=new_t_min, t_max=new_t_max)

    def _initialize_generators(self) -> None:
        """Fetch all SSO generators for the current time range and pre-initialize interpolation."""
        sky_box = [
            SkyCoord(ra=0.0 * u.deg, dec=-90.0 * u.deg),
            SkyCoord(ra=359.999 * u.deg, dec=90.0 * u.deg),
        ]
        for sg in tqdm.tqdm(
            self.catalog.sso.get_box(
                lower_left=sky_box[0],
                upper_right=sky_box[1],
                t_min=self.t_min,
                t_max=self.t_max,
                source_cat=self.catalog,
                ephem_cat=self.catalog.ephem,
            ),
            desc="socat: initializing SSO interpolation",
        ):
            is_sso = isinstance(sg.source, SolarSystemObject)
            cache_key = (
                f"sso:{sg.source.sso_id}" if is_sso else f"fixed:{sg.source.source_id}"
            )
            sg.init_interp(ephem_cat=self.catalog.ephem)
            self._interp_cache[cache_key] = sg
        self.log.info(
            "socat.generators_initialized",
            n_generators=len(self._interp_cache),
            t_min=self.t_min,
            t_max=self.t_max,
        )

    def _refresh(self, t_min: Time | None = None, t_max: Time | None = None):
        """Clear the interpolation cache, update the time range, and re-initialize generators."""
        self._interp_cache.clear()
        if t_min is not None:
            self.t_min = t_min
        if t_max is not None:
            self.t_max = t_max
        self.log.info(
            "socat.refreshed",
            t_min=self.t_min,
            t_max=self.t_max,
        )
        if self.t_min is not None and self.t_max is not None:
            self._initialize_generators()

    def _sg_to_registered(self, sg, t: Time) -> RegisteredSource | None:
        """
        Convert an socat SourceGenerator output to a RegisteredSource, querying at time t.

        sets the source type to "sso" if the source is a SolarSystemObject,
        otherwise uses the source_type attribute if it exists, otherwise "unknown".
        """
        try:
            is_sso = isinstance(sg.source, SolarSystemObject)
            cache_key = (
                f"sso:{sg.source.sso_id}" if is_sso else f"fixed:{sg.source.source_id}"
            )
            if cache_key not in self._interp_cache:
                sg.init_interp(ephem_cat=self.catalog.ephem)
                self._interp_cache[cache_key] = sg
            else:
                sg = self._interp_cache[cache_key]
            position, flux = sg.at_time(t=t)
        except (AttributeError, ValueError):
            return None
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

    def add_sources(self, sources: list[RegisteredSource], no_monitor: bool = False):
        for source in sources:
            self.catalog.create_source(
                position=SkyCoord(source.ra, source.dec),
                flux=source.flux,
                name=source.source_id,
                monitored=not no_monitor,
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

    def get_sources_in_map(
        self, input_map: ProcessableMap, monitored: bool = False
    ) -> list[RegisteredSource]:
        """
        Get sources within the given map.

        Parameters
        ----------
        input_map : ProcessableMap
            The map for which to get sources.
        monitored : bool
            If True, only return sources that are listed as `monitored` in the catalog.

        Returns
        -------
        list[RegisteredSource]
            List of sources within the map.
        """
        self._check_and_expand_interp_range(input_map)
        ## for now, just get all the sources and filter them.
        sky_box = [
            SkyCoord(ra=0.0 * u.deg, dec=-90.0 * u.deg),
            SkyCoord(ra=359.999 * u.deg, dec=90.0 * u.deg),
        ]
        if input_map.observation_time is not None:
            t_mid = Time(input_map.observation_time)
        elif self.t_min is not None and self.t_max is not None:
            t_mid = self.t_min + (self.t_max - self.t_min) / 2
        else:
            t_mid = None

        if self._interp_cache:
            source_generators = self._interp_cache.values()
        else:
            source_generators = self.catalog.sso.get_box(
                lower_left=sky_box[0],
                upper_right=sky_box[1],
                t_min=self.t_min,
                t_max=self.t_max,
                source_cat=self.catalog,
                ephem_cat=self.catalog.ephem,
            )
        if monitored:
            source_generators = [sg for sg in source_generators if sg.source.monitored]

        all_sources = [
            s
            for s in (
                self._sg_to_registered_with_refinement(sg, t_mid, input_map)
                for sg in tqdm.tqdm(source_generators)
            )
            if s is not None
        ]

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

        inside, _ = input_map.filter_sources(coords)
        if not np.any(inside):
            self.log.warning(
                "socat.get_sources_in_map.no_sources_in_map",
                map_id=input_map.map_id,
                wcs=input_map.flux.wcs,
                n_sources_in_bbox=len(all_sources),
            )
            return []

        return [all_sources[i] for i in range(len(all_sources)) if inside[i]]

    def get_all_sources(self) -> list[RegisteredSource]:
        return self.get_sources_in_box(box=None)

    def forced_photometry_sources(
        self, input_map: ProcessableMap, use_sso: bool = True
    ) -> list[RegisteredSource]:
        """
        Wrapper for socat's get_forced_photometry because it doesn't work with
        ssos yet.
        If times arent provided (or use_sso is False) then it just grabs the fixed sources.
        Otherwise, grab all sources and filter them.
        """
        obs_time = Time(input_map.observation_time)
        log = self.log.bind(
            func="socat.forced_photometry_sources",
            obs_time=obs_time,
            t_min=self.t_min,
            t_max=self.t_max,
            use_sso=use_sso,
        )

        if not use_sso:
            log.info("socat.forced_photometry_sources.fixed_sources_only")
            sources = [
                self._fixed_to_registered(s)
                for s in self.catalog.get_forced_photometry_sources(
                    minimum_flux=self.flux_lower_limit
                )
            ]
        elif self.t_min is None or self.t_max is None:
            log.warning(
                "socat.forced_photometry_sources.invalid_time_range",
                map_id=input_map.map_id,
            )
            sources = [
                self._fixed_to_registered(s)
                for s in self.catalog.get_forced_photometry_sources(
                    minimum_flux=self.flux_lower_limit
                )
            ]
        else:
            log.info("socat.forced_photometry_sources.retrieving_sources_with_sso")
            candidates = self.get_sources_in_map(input_map, monitored=True)
            sources = [
                s
                for s in candidates
                if s is not None
                and (s.flux is not None or self.flux_lower_limit is None)
                and (self.flux_lower_limit is None or s.flux >= self.flux_lower_limit)
            ]

        log.info("socat.forced_photometry_sources", n_sources=len(sources))
        if not sources:
            return []

        coords = SkyCoord(ra=[s.ra for s in sources], dec=[s.dec for s in sources])
        inside, _ = input_map.filter_sources(coords)
        if not np.any(inside):
            log.warning(
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
