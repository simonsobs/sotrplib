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

# Specific padding for the time range used to construct
# interpolation objects. Add this by default to all time ranges.
_TIME_RANGE_PADDING = TimeDelta(1, format="jd")


class SOCat(SourceCatalog):
    """
    A catalog backed by the configured SOCat database.

    Queries both fixed and solar-system-object (moving) sources.
    SSO queries require t_min and t_max so that ephemeris interpolation
    is bounded.

    """

    def __init__(
        self,
        log: FilteringBoundLogger | None = None,
    ):
        self.log = log or structlog.get_logger()
        self.catalog = SOCatClientSettings().client
        self.log.info(
            "socat.initialized",
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
        self, sky_box: tuple[SkyCoord, SkyCoord] | None = None, t: Time | None = None
    ) -> list[RegisteredSource]:
        """
        Get sources within the given sky box.
        If sky_box is None, get all sources.
        If t is provided, get sources at that time by interpolating using +/- padding time,
        then querying source at that the time, t.
        """
        if sky_box is None:
            sky_box = (
                SkyCoord(0, -89.999, unit="deg"),
                SkyCoord(359.999, 89.999, unit="deg"),
            )

        if t is None:
            self.log.warning("socat.get_sources_in_box.no_eval_time")
            source_generators = self.catalog.get_box_fixed(
                lower_left=sky_box[0], upper_right=sky_box[1]
            )
        else:
            source_generators = self.catalog.get_box(
                lower_left=sky_box[0],
                upper_right=sky_box[1],
                t_min=t - _TIME_RANGE_PADDING,
                t_max=t + _TIME_RANGE_PADDING,
            )

        sources = [
            s
            for s in (self._sg_to_registered(sg, t) for sg in source_generators)
            if s is not None
        ]
        self.log.debug(
            "socat.get_sources_in_box",
            sky_box=sky_box,
            n_sources=len(sources),
            t_eval=t,
        )
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
        if input_map.observation_time is not None:
            t_mid = Time(input_map.observation_time)
            t_min = (
                Time(input_map.observation_start) - _TIME_RANGE_PADDING
                if input_map.observation_start is not None
                else None
            )
            t_max = (
                Time(input_map.observation_end) + _TIME_RANGE_PADDING
                if input_map.observation_end is not None
                else None
            )
        else:
            self.log.warning(
                "socat.get_sources_in_map.no_observation_time",
                map_id=input_map.map_id,
            )
            t_mid = None
            t_min = None
            t_max = None

        # Get sources within the map's sky box and a padded time range around the observation.
        source_generators = self.catalog.get_box(
            lower_left=input_map.sky_box[0],
            upper_right=input_map.sky_box[1],
            t_min=t_min,
            t_max=t_max,
        )
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

    def get_all_sources(self, t: Time | None = None) -> list[RegisteredSource]:
        """
        Just a helper function to get all the sources in the
        catalog by querying with no spatial constraints.
        If t is not provided, only fixed sources are returned.
        """
        return self.get_sources_in_box(sky_box=None, t=t)

    def forced_photometry_sources(
        self, input_map: ProcessableMap, flux_lower_limit: u.Quantity | None = None
    ) -> list[RegisteredSource]:
        """
        Wrapper for socat's get_monitored_sources.
        If observation start and end times are not in the input_map,
        then only fixed sources are returned.
        Grabs the sources and filter them to
        return only ones within the map region.
        """
        obs_time = (
            Time(input_map.observation_time)
            if input_map.observation_time is not None
            else None
        )
        t_min = (
            Time(input_map.observation_start) - _TIME_RANGE_PADDING
            if input_map.observation_start is not None
            else None
        )
        t_max = (
            Time(input_map.observation_end) + _TIME_RANGE_PADDING
            if input_map.observation_end is not None
            else None
        )
        log = self.log.bind(
            func="socat.forced_photometry_sources",
            obs_time=obs_time,
            t_min=t_min,
            t_max=t_max,
            flux_lower_limit=flux_lower_limit,
        )

        if t_min is None or t_max is None:
            log.warning(
                "socat.forced_photometry_sources.invalid_time_range",
                map_id=input_map.map_id,
            )
            sources = [
                self.sg_to_registered(sg, obs_time)
                for sg in self.catalog.get_box_fixed(minimum_flux=flux_lower_limit)
            ]
        else:
            log.info("socat.forced_photometry_sources.retrieving_sources_with_sso")
            candidates = self.get_sources_in_map(input_map, monitored=True)
            sources = [
                s
                for s in candidates
                if s is not None
                and (s.flux is not None or flux_lower_limit is None)
                and (flux_lower_limit is None or s.flux >= flux_lower_limit)
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
        t: Time | None = None,
    ) -> list[CrossMatch]:
        """
        Crossmatch the given coordinates with the sources in the catalog.
        This uses pixell's crossmatch function.

        If a time is provided, the catalog positions will be
        evaluated at that time.

        Makes use of `get_sources_in_box`.
        """
        close_sources = self.get_sources_in_box(
            [
                SkyCoord(ra=ra - 2.0 * radius, dec=dec - 2.0 * radius),
                SkyCoord(ra=ra + 2.0 * radius, dec=dec + 2.0 * radius),
            ],
            t=t,
        )
        catalog_positions = np.asarray(
            [(x.ra.to("radian").value, x.dec.to("radian").value) for x in close_sources]
        )
        if len(catalog_positions) == 0:
            return []
        matches = pixell_utils.crossmatch(
            pos1=[[ra.to("radian").value, dec.to("radian").value]],
            pos2=catalog_positions,
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
        ## couple lines to get source type.
        ## not yet included in socat
        if is_sso:
            source_type = "sso"
        elif hasattr(sg.source, "source_type"):
            source_type = sg.source.source_type
        else:
            source_type = "unknown"
        ## redundant for now.
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
