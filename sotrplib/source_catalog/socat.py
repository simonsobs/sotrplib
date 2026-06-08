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
from socat.core import SourceGenerator
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

        self._source_generators: list[SourceGenerator] | None = None

    def _sg_to_registered(
        self, sg: SourceGenerator, t: Time
    ) -> RegisteredSource | None:
        """
        Convert an socat SourceGenerator output to a RegisteredSource, querying at time t.

        sets the source type to "sso" if the source is a SolarSystemObject,
        otherwise uses the source_type attribute if it exists, otherwise "unknown".
        """
        try:
            is_sso = isinstance(sg.source, SolarSystemObject)
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

    def _sg_to_registered_with_refinement(
        self,
        sg: SourceGenerator,
        t_initial: Time,
        input_map: ProcessableMap,
        position_tolerance: u.Quantity = 1.0 * u.arcmin,
        max_iterations: int = 10,
    ) -> RegisteredSource | None:
        """
        Convert a SourceGenerator to a RegisteredSource, iteratively refining the
        SSO position using the map's per-pixel observation time until the change in
        position between iterations is smaller than position_tolerance.

        For a moving source, we query the position at t_initial, look up the
        per-pixel observation time at that position, re-query the position at the
        new time, and repeat until convergence.  Non-SSO sources are returned
        immediately without refinement.

        Main belt asteroids have apparent motion of up to several arcmin per hour.
        For a depth1 map that spans 12-24 hours, we therefore might expect up to a
        degree or two of offset.  Iterative refinement converges to positional errors
        well below position_tolerance.
        """
        current = self._sg_to_registered(sg, t_initial)
        if current is None or not isinstance(sg.source, SolarSystemObject):
            return current
        for _ in range(max_iterations):
            t = input_map.get_obs_time(SkyCoord(ra=current.ra, dec=current.dec))
            if t is None:
                return current
            refined = self._sg_to_registered(sg, t)
            if refined is None:
                return current
            sep = SkyCoord(ra=current.ra, dec=current.dec).separation(
                SkyCoord(ra=refined.ra, dec=refined.dec)
            )
            current = refined
            if sep < position_tolerance:
                break
        return current

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

    def _initialize_generators(self, box: list[SkyCoord] | None = None) -> None:
        """Fetch SourceGenerators for the full sky at the current time range and cache them."""
        self._source_generators = list(
            self.catalog.get_box(
                lower_left=box[0]
                if box is not None
                else SkyCoord(ra=0.0 * u.deg, dec=-90.0 * u.deg),
                upper_right=box[1]
                if box is not None
                else SkyCoord(ra=359.999 * u.deg, dec=90.0 * u.deg),
                t_min=self.t_min,
                t_max=self.t_max,
            )
        )
        self.log.info(
            "socat.generators_initialized",
            n_generators=len(self._source_generators),
            t_min=self.t_min,
            t_max=self.t_max,
        )

    def _get_source_generators(
        self, box: list[SkyCoord] | None
    ) -> list[SourceGenerator]:
        """Return cached SourceGenerators, fetching from DB if the cache is stale."""
        if self._source_generators is None:
            if box is None:
                box = [
                    SkyCoord(ra=0.0 * u.deg, dec=-90.0 * u.deg),
                    SkyCoord(ra=359.999 * u.deg, dec=90.0 * u.deg),
                ]
            if self.t_min is None or self.t_max is None:
                self.log.warning("socat.get_source_generators.no_time_range")
                return []
            self._initialize_generators(box=box)
        return self._source_generators

    def _refresh(self, t_min: Time | None = None, t_max: Time | None = None):
        """Update the time range and invalidate the SourceGenerator cache."""
        if t_min is not None:
            self.t_min = t_min
        if t_max is not None:
            self.t_max = t_max
        self._source_generators = None
        self._get_source_generators(
            box=None
        )  # pre-fetch generators for the new time range
        self.log.info(
            "socat.refreshed",
            t_min=self.t_min,
            t_max=self.t_max,
        )

    def add_sources(self, sources: list[RegisteredSource], monitor: bool = True):
        for source in sources:
            self.catalog.create_source(
                position=SkyCoord(source.ra, source.dec),
                flux=source.flux,
                name=source.source_id,
                flags={"monitored": monitor},
            )

    def get_sources_in_box(
        self, box: list[SkyCoord] | None = None
    ) -> list[RegisteredSource]:
        """Get sources within the given sky box and time range.
        If box is None, get all sources in the time range.
        """
        if self.t_min is None or self.t_max is None:
            self.log.error("socat.get_sources_in_box.no_time_range")
            return []
        t_eval = self.t_min + (self.t_max - self.t_min) / 2
        source_generators = self._get_source_generators(box=box)
        sources = [
            s
            for s in (self._sg_to_registered(sg, t_eval) for sg in source_generators)
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
        if input_map.observation_time is not None:
            t_mid = Time(input_map.observation_time)
        elif self.t_min is not None and self.t_max is not None:
            t_mid = self.t_min + (self.t_max - self.t_min) / 2
        else:
            t_mid = None

        source_generators = self._get_source_generators(box=input_map.bbox)
        if monitored:
            source_generators = [sg for sg in source_generators if sg.source.monitored]

        all_sources = [
            s
            for s in (
                self._sg_to_registered_with_refinement(
                    sg,
                    t_mid,
                    input_map,
                    position_tolerance=input_map.map_resolution / 2,
                )
                for sg in tqdm.tqdm(source_generators)
            )
            if s is not None
        ]

        if not all_sources:
            self.log.warning(
                "socat.get_sources_in_map.no_sources_in_box",
                map_id=input_map.map_id,
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
        self, input_map: ProcessableMap
    ) -> list[RegisteredSource]:
        """
        Wrapper for socat's get_monitored_sources.
        If times arent provided (or use_sso is False) then it just grabs the fixed sources.
        Otherwise, grab all sources and filter them.
        """
        obs_time = Time(input_map.observation_time)
        log = self.log.bind(
            func="socat.forced_photometry_sources",
            obs_time=obs_time,
            t_min=self.t_min,
            t_max=self.t_max,
        )

        if self.t_min is None or self.t_max is None:
            log.warning(
                "socat.forced_photometry_sources.invalid_time_range",
                map_id=input_map.map_id,
            )
            sources = [
                self.sg_to_registered(sg, obs_time)
                for sg in self.catalog.get_box_fixed(minimum_flux=self.flux_lower_limit)
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
                catalog_idx=s.source_id,
                alternate_names=s.alternate_names,
                ra=s.ra,
                dec=s.dec,
                observation_time=s.observation_mean_time,
            )
            for s in sources
        ]
