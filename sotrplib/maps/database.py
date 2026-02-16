"""
Read maps from the map tracking database.
"""

from datetime import datetime, timezone

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

from .core import IntensityAndInverseVarianceMap


class MapCatDatabaseReader:
    """
    Reader for maps from the map tracking database. Note that the
    database connection is configured through the map catalog library
    itself.
    """

    instrument: str | None = None
    frequency: str | None = None
    array: str | None = None
    number_to_read: int | None = 1
    start_time: AwareDatetime | None = None
    end_time: AwareDatetime | None = None
    map_ids: list[int] | None = None
    box: tuple[SkyCoord, SkyCoord] | None = None
    intensity_units: u.Unit = u.Unit("K")
    rerun: bool = False
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
        intensity_units: u.Unit = u.Unit("K"),
        rerun: bool = False,
        log: FilteringBoundLogger | None = None,
    ):
        self.number_to_read = number_to_read
        self.start_time = start_time
        self.end_time = end_time
        self.map_ids = map_ids or []
        self.frequency = frequency
        self.array = array
        self.instrument = instrument
        self.intensity_units = intensity_units
        self.box = box
        self.rerun = rerun
        self.log = log or get_logger()

    def map_list(self):
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

        ## Limit the number of maps to read
        ## but I now want to skip processed ones, so will do that below
        # query = query.limit(self.number_to_read)
        maps = []

        with mapcat_settings.session() as session:
            results = session.execute(query).scalars().all()
            self.log.info("MapCatDatabaseReader.found_maps", number_found=len(results))
            if self.number_to_read is None:
                self.number_to_read = len(results)
            for result in results:
                if not self.rerun and check_if_processed(
                    result.map_id, session=session
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
                        intensity_units=self.intensity_units,
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

        return maps

    def __iter__(self):
        return iter(self.map_list())


def check_if_processed(
    map_id: int, session=None, valid_statuses: list = ["completed", "processing"]
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
        if r.processing_status in valid_statuses:
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
