"""
Use a SOCat database as a source catalog.
"""

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import structlog
import tqdm
from astropy import units as u
from astropy.coordinates import ICRS, SkyCoord
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

if TYPE_CHECKING:
    from sotrplib.sifter.map_matching import MapMatchSummary

# Specific padding for the time range used to construct
# interpolation objects. Add this by default to all time ranges.
_TIME_RANGE_PADDING = TimeDelta(1, format="jd")


def transient_name(prefix: str, ra: u.Quantity, dec: u.Quantity) -> str:
    """
    Return a position name for a new source, for example
    "SO-T J231323.4+024034".
    """
    position = SkyCoord(ra=ra, dec=dec)
    ra_str = position.ra.to_string(unit=u.hourangle, sep="", precision=1, pad=True)
    dec_str = position.dec.to_string(sep="", precision=0, alwayssign=True, pad=True)
    return f"{prefix} J{ra_str}{dec_str}"


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
        add_confirmed_transients: bool = False,
        duplicate_radius: u.Quantity = 1.0 * u.arcmin,
        name_prefix: str = "SO-T",
        record_directory: Path | None = None,
        monitor_existing_sources: bool = True,
    ):
        """
        Parameters
        ----------
        add_confirmed_transients : bool
            If True, update() adds each confirmed map-match group to the
            catalog as a new monitored fixed source. The forced photometry
            of the next runs then measures it.
        duplicate_radius : u.Quantity
            update() does not add a group if a catalog source is within this
            distance of the group position.
        name_prefix : str
            update() names a new source "<name_prefix> J<hhmmss.s><+ddmmss>".
        record_directory : Path | None
            If not None, update() writes a JSON record of each update to
            this directory. The socat schema keeps only the position, the
            flux, the name and the flags. The record keeps the map-match
            data of each new source (group_id, significance, maps).
        monitor_existing_sources : bool
            If a confirmed group matches a fixed catalog source that is not
            monitored, update() sets the source to monitored (action
            "monitored"). For example, a known star that flares.
        """
        self.log = log or structlog.get_logger()
        self.catalog = SOCatClientSettings().client
        self.add_confirmed_transients = add_confirmed_transients
        self.duplicate_radius = duplicate_radius
        self.name_prefix = name_prefix
        self.record_directory = record_directory
        self.monitor_existing_sources = monitor_existing_sources
        self.log.info(
            "socat.initialized",
        )

    def __getstate__(self):
        # The socat client holds SQLAlchemy engines/sessionmakers, which can't
        # be pickled -- drop it so the catalog can be sent to process-pool
        # workers (e.g. the prefect runner), and reconnect from the same
        # environment settings on the other side.
        state = self.__dict__.copy()
        del state["catalog"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.catalog = SOCatClientSettings().client

    def add_sources(self, sources: list[RegisteredSource], monitored: bool = True):
        for source in sources:
            self.catalog.create_source(
                position=SkyCoord(source.ra, source.dec),
                flux=source.flux,
                name=source.source_id,
                flags={"monitored": monitored},
            )

    def update(self, summary: "MapMatchSummary") -> list[RegisteredSource]:
        """
        Add the confirmed map-match groups of a run to the catalog, as
        monitored fixed sources. Only does this if add_confirmed_transients
        is True.

        A group is not added if a catalog source (fixed, or a solar system
        object at the time of the group) is within duplicate_radius of the
        group position. Thus, a rerun of the same maps does not add the
        same event again. If that source is a fixed source that is not
        monitored, and monitor_existing_sources is True, the source is set
        to monitored.

        Each member candidate of an added group gets a crossmatch to the new
        source. Returns the new sources and the newly monitored sources.

        The socat id (a UUID7) of each source is in crossmatches[0].catalog_idx,
        as for the sources from get_sources_in_box. The source_id of a
        returned source is its socat name.
        """
        if not self.add_confirmed_transients:
            return []

        added = []
        records = []
        for group in summary.groups:
            if not group.confirmed:
                continue

            times = [
                c.observation_mean_time
                for _, c in group.members
                if c.observation_mean_time is not None
            ]
            t = min(times) if times else summary.observation_start
            group_record = {
                "group_id": str(group.group_id),
                "confirmed": group.confirmed,
                "rank": group.rank,
                "significance": group.significance,
                "max_snr": group.max_snr,
                "n_maps": group.n_maps,
                "n_arrays": group.n_arrays,
                "n_bands": group.n_bands,
                "ra_deg": group.ra.to_value(u.deg),
                "dec_deg": group.dec.to_value(u.deg),
                "first_detection": t.isot if t is not None else None,
                "maps": sorted({r.map_name for r, _ in group.members}),
            }

            existing = self._closest_source(group.ra, group.dec, t)
            if existing is not None:
                action = "already_in_catalog"
                socat_id: uuid.UUID = existing.crossmatches[0].catalog_idx
                if (
                    self.monitor_existing_sources
                    and existing.source_type != "sso"
                    and not self.catalog.get_source(source_id=socat_id).monitored
                ):
                    # A known source that is now a confirmed transient (for
                    # example, a flaring star). Monitor it from now on.
                    self.catalog.update_source(
                        source_id=socat_id, flags={"monitored": True}
                    )
                    action = "monitored"
                    added.append(existing)
                self.log.info(
                    f"socat.update.group_{action}",
                    group_id=str(group.group_id),
                    name=str(existing.source_id),
                    socat_id=socat_id,
                )
                records.append(
                    {
                        **group_record,
                        "action": action,
                        "socat_id": socat_id,
                        "name": str(existing.source_id),
                    }
                )
                continue

            fluxes = [c.flux.to(u.mJy) for _, c in group.members if c.flux is not None]
            flux = max(fluxes) if fluxes else None
            name = transient_name(self.name_prefix, group.ra, group.dec)
            created = self.catalog.create_source(
                position=ICRS(ra=group.ra, dec=group.dec),
                flux=flux,
                name=name,
                flags={"monitored": True},
            )
            socat_id = created.source_id

            crossmatch = CrossMatch(
                source_id=name,
                source_type="unknown",
                probability=1.0,
                catalog_name="socat",
                catalog_idx=socat_id,
                ra=group.ra,
                dec=group.dec,
                flux=flux,
            )
            for _, candidate in group.members:
                candidate.add_crossmatch(
                    crossmatch.model_copy(
                        update={
                            "angular_separation": angular_separation(
                                group.ra, group.dec, candidate.ra, candidate.dec
                            )
                        }
                    )
                )

            added.append(
                RegisteredSource(
                    ra=group.ra,
                    dec=group.dec,
                    flux=flux,
                    source_id=name,
                    source_type="unknown",
                    catalog_name="socat",
                    observation_mean_time=t,
                    crossmatches=[crossmatch],
                )
            )
            records.append(
                {
                    **group_record,
                    "action": "added",
                    "socat_id": socat_id,
                    "name": name,
                    "flux_mJy": flux.to_value(u.mJy) if flux is not None else None,
                }
            )
            self.log.info(
                "socat.update.added_transient",
                name=name,
                socat_id=socat_id,
                group_id=str(group.group_id),
                significance=round(group.significance, 2),
                ra=group.ra.to_value(u.deg),
                dec=group.dec.to_value(u.deg),
            )

        self.log.info(
            "socat.update.completed",
            n_confirmed_groups=sum(g.confirmed for g in summary.groups),
            n_added=len(added),
        )
        if self.record_directory is not None and records:
            self._write_update_record(summary, records)
        return added

    def _closest_source(
        self, ra: u.Quantity, dec: u.Quantity, t: Time | None
    ) -> RegisteredSource | None:
        """
        The closest catalog source (fixed, or a solar system object at time
        t) within duplicate_radius, or None. Its first crossmatch has the
        socat id (catalog_idx).
        """
        radius = self.duplicate_radius
        nearby = self.get_sources_in_box(
            [
                SkyCoord(ra=ra - 2.0 * radius, dec=dec - 2.0 * radius),
                SkyCoord(ra=ra + 2.0 * radius, dec=dec + 2.0 * radius),
            ],
            t=t,
        )
        best, best_sep = None, None
        for source in nearby:
            sep = angular_separation(source.ra, source.dec, ra, dec)
            if sep <= radius and (best_sep is None or sep < best_sep):
                best, best_sep = source, sep
        return best

    def _write_update_record(self, summary: "MapMatchSummary", records: list[dict]):
        self.record_directory.mkdir(parents=True, exist_ok=True)
        start = summary.observation_start
        stamp = start.strftime("%Y%m%dT%H%M%S") if start is not None else "unknown"
        # A rerun of the same maps writes a new file. It does not replace the
        # record of the first run, which has the sources that it added.
        written = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
        path = self.record_directory / f"socat_update_{stamp}_{written}.json"
        with open(path, "w") as f:
            # default=str writes the socat UUIDs as strings.
            json.dump(
                {"maps": list(summary.map_names), "groups": records},
                f,
                indent=2,
                default=str,
            )
        self.log.info("socat.update.record_written", path=str(path))

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
                mapcat_id=input_map.mapcat_id,
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
                mapcat_id=input_map.mapcat_id,
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
                mapcat_id=input_map.mapcat_id,
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
                mapcat_id=input_map.mapcat_id,
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
                mapcat_id=input_map.mapcat_id,
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
