"""
Read maps from the map tracking database.
"""

from datetime import datetime, timedelta, timezone
from typing import Literal

from astropy import units as u
from astropy.coordinates import SkyCoord

# Libraries are loaded here because of the external database
# connection; this only happens once, and we don't want it to
# affect the import time of this module (or need a database to
# be available just to import sotrplib).
from mapcat.database import DepthOneMapTable, TimeDomainProcessingTable
from mapcat.helper import settings as mapcat_settings
from pydantic import AwareDatetime
from sqlmodel import select
from structlog import get_logger
from structlog.types import FilteringBoundLogger

from .core import FluxAndSNRMap, IntensityAndInverseVarianceMap, RhoAndKappaMap


class MapCatDatabaseReader:
    """
    Reader for maps from the map tracking database. Note that the
    database connection is configured through the map catalog library
    itself.

    Default is to read all arrays and frequencies for the past 1 day
    by setting start_time to 1 day ago and end_time to now.

    """

    instrument: str | None = None
    frequency: str | None = None
    array: str | None = None
    number_to_read: int | None = None
    start_time: AwareDatetime | None = None
    end_time: AwareDatetime | None = None
    map_ids: list[int] | None = None
    box: tuple[SkyCoord, SkyCoord] | None = None
    map_type: Literal["intensity", "flux", "rhokappa"] = "intensity"
    map_units: u.Unit = u.Unit("K")
    rerun: bool = False
    stale_processing_time: timedelta = timedelta(hours=2)
    log: FilteringBoundLogger

    def __init__(
        self,
        number_to_read: int | None = None,
        start_time: AwareDatetime | None = None,
        end_time: AwareDatetime | None = None,
        map_ids: list[int] | None = None,
        frequency: str | None = None,
        array: str | None = None,
        instrument: str | None = None,
        box: tuple[SkyCoord, SkyCoord] | None = None,
        map_units: u.Unit = u.Unit("K"),
        map_type: Literal["intensity", "flux", "rhokappa"] = "intensity",
        rerun: bool = False,
        stale_processing_time: timedelta = timedelta(hours=2),
        log: FilteringBoundLogger | None = None,
    ):
        self.number_to_read = number_to_read
        self.start_time = (
            start_time
            if start_time is not None
            else datetime.now(timezone.utc) - timedelta(days=1)
        )
        self.end_time = end_time if end_time is not None else datetime.now(timezone.utc)
        self.map_ids = map_ids or []
        self.frequency = frequency
        self.array = array
        self.instrument = instrument
        self.map_units = map_units
        self.box = box
        self.rerun = rerun
        self.map_type = map_type
        if self.map_type == "flux":
            raise NotImplementedError(
                "Flux map reading is not yet implemented in mapcat depth one map table"
            )
        self._map_list = None
        self.stale_processing_time = stale_processing_time
        self.log = log or get_logger()

        self.validate_map_units()

    def validate_map_units(self):
        if self.map_type not in ("intensity", "flux", "rhokappa"):
            raise ValueError(
                f"map_type must be one of 'intensity','flux','rhokappa', got {self.map_type}"
            )
        if self.map_type == "intensity" and self.map_units.is_equivalent(u.K) is False:
            raise ValueError(
                f"map_units must be equivalent to Kelvin for intensity maps, got {self.map_units}"
            )
        elif self.map_type == "flux" and self.map_units.is_equivalent(u.Jy) is False:
            raise ValueError(
                f"map_units must be equivalent to Jansky for flux maps, got {self.map_units}"
            )
        elif (
            self.map_type == "rhokappa" and self.map_units.is_equivalent(u.Jy) is False
        ):
            raise ValueError(
                f"map_units must be equivalent to Jansky for rhokappa maps, got {self.map_units}"
            )

    def map_list(self):
        if self._map_list is not None:
            return self._map_list

        self.log.info(
            "MapCatDatabaseReader.connecting_to_db",
            db_url=mapcat_settings.database_name,
        )
        query = select(DepthOneMapTable)

        query = (
            query.where(DepthOneMapTable.frequency == self.frequency)
            if self.frequency
            else query
        )
        query = (
            query.where(DepthOneMapTable.tube_slot == self.array)
            if self.array
            else query
        )

        if self.start_time is not None:
            query = query.where(
                DepthOneMapTable.stop_time >= self.start_time.timestamp()
            )
        if self.end_time is not None:
            query = query.where(
                DepthOneMapTable.start_time <= self.end_time.timestamp()
            )

        if self.map_ids:
            query = query.where(DepthOneMapTable.map_id.in_(self.map_ids))

        maps = []
        with mapcat_settings.session() as session:
            results = session.execute(query).scalars().all()
            self.log.info("MapCatDatabaseReader.found_maps", number_found=len(results))
            if self.number_to_read is None:
                self.number_to_read = len(results)
            for result in results:
                if not self.rerun and check_if_processed(
                    result.map_id,
                    session=session,
                    stale_limit=self.stale_processing_time,
                ):
                    self.log.info(
                        "MapCatDatabaseReader.skipping_processed_map",
                        map_id=result.map_id,
                    )
                    continue
                ## workaround for failed mapcat generation
                if result.mean_time_path is None:
                    result.mean_time_path = (
                        str(result.ivar_path).split("_ivar.fits")[0] + "_time.fits"
                    )
                if self.map_type == "intensity":
                    maps.append(
                        IntensityAndInverseVarianceMap(
                            intensity_filename=mapcat_settings.depth_one_parent
                            / result.map_path,
                            inverse_variance_filename=mapcat_settings.depth_one_parent
                            / result.ivar_path,
                            time_filename=mapcat_settings.depth_one_parent
                            / result.mean_time_path,
                            start_time=datetime.fromtimestamp(
                                result.start_time, tz=timezone.utc
                            ),
                            end_time=datetime.fromtimestamp(
                                result.stop_time, tz=timezone.utc
                            ),
                            box=self.box,
                            intensity_units=self.map_units,
                            frequency=result.frequency,
                            array=result.tube_slot,
                            instrument=self.instrument,
                            log=self.log,
                        )
                    )
                elif self.map_type == "rhokappa":
                    maps.append(
                        RhoAndKappaMap(
                            rho_filename=mapcat_settings.depth_one_parent
                            / result.rho_path,
                            kappa_filename=mapcat_settings.depth_one_parent
                            / result.kappa_path,
                            time_filename=mapcat_settings.depth_one_parent
                            / result.mean_time_path,
                            start_time=datetime.fromtimestamp(
                                result.start_time, tz=timezone.utc
                            ),
                            end_time=datetime.fromtimestamp(
                                result.stop_time, tz=timezone.utc
                            ),
                            box=self.box,
                            flux_units=self.map_units,
                            frequency=result.frequency,
                            array=result.tube_slot,
                            instrument=self.instrument,
                            log=self.log,
                        )
                    )
                else:
                    maps.append(
                        FluxAndSNRMap(
                            flux_filename=mapcat_settings.depth_one_parent
                            / result.flux_path,
                            snr_filename=mapcat_settings.depth_one_parent
                            / result.snr_path,
                            time_filename=mapcat_settings.depth_one_parent
                            / result.mean_time_path,
                            start_time=datetime.fromtimestamp(
                                result.start_time, tz=timezone.utc
                            ),
                            end_time=datetime.fromtimestamp(
                                result.stop_time, tz=timezone.utc
                            ),
                            box=self.box,
                            flux_units=self.map_units,
                            frequency=result.frequency,
                            array=result.tube_slot,
                            instrument=self.instrument,
                            log=self.log,
                        )
                    )

                maps[-1].map_id = result.map_id
                maps[-1]._parent_database = mapcat_settings.database_name
                self.map_ids.append(result.map_id)
                set_processing_start(result.map_id, session=session)
                if len(maps) >= self.number_to_read:
                    break
        self._map_list = maps
        return maps

    def __iter__(self):
        return iter(self.map_list())


def check_if_processed(
    map_id: int,
    session=None,
    completed_status: str = "completed",
    processing_status: str = "processing",
    stale_limit=timedelta(hours=2),
) -> bool:
    ## session is mapcat_settings.session() whatever that is
    if session is None:
        session = mapcat_settings.session()
    query = select(TimeDomainProcessingTable).where(
        TimeDomainProcessingTable.map_id == map_id
    )
    session_results = session.execute(query).one_or_none()
    if session_results is None:
        return False
    for r in session_results:
        if (r.processing_status == completed_status) | (
            r.processing_status == processing_status
        ) & (
            (datetime.now(timezone.utc).timestamp() - r.processing_start)
            < stale_limit.total_seconds()
        ):
            return True
    return False


def set_processing_start(map_id: int, session=None):
    ## session is mapcat_settings.session() whatever that is
    if session is None:
        session = mapcat_settings.session()
    query = select(TimeDomainProcessingTable).where(
        TimeDomainProcessingTable.map_id == map_id
    )
    session_results = session.execute(query).one_or_none()
    if session_results is None:
        session_results = [
            TimeDomainProcessingTable(processing_status_id=map_id, map_id=map_id)
        ]
    for r in session_results:
        r.processing_start = datetime.now(timezone.utc).timestamp()
        r.processing_status = "processing"

        session.add(r)
        session.commit()
    return


def set_processing_end(map_id: int, session=None):
    ## session is mapcat_settings.session() whatever that is
    if session is None:
        session = mapcat_settings.session()
    query = select(TimeDomainProcessingTable).where(
        TimeDomainProcessingTable.processing_status_id == map_id
    )
    session_result = session.execute(query).one_or_none()
    if session_result is None:
        raise ValueError(
            f"No processing_start status found for map_id {map_id} when trying to set processing_end."
        )
    for r in session_result:
        r.processing_end = datetime.now(timezone.utc).timestamp()
        r.processing_status = "completed"
        session.add(r)
        session.commit()
    return
