"""
Read maps from the map tracking database.
"""

from datetime import datetime, timezone

from astropy import units as u

# Libraries are loaded here because of the external database
# connection; this only happens once, and we don't want it to
# affect the import time of this module (or need a database to
# be available just to import sotrplib).
from mapcat.database import DepthOneMapTable, ProcessingStatusTable
from mapcat.helper import settings as mapcat_settings
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
    number_to_read: int = 1
    rerun: bool = False
    log: FilteringBoundLogger

    def __init__(
        self,
        number_to_read: int = 1,
        frequency: str | None = None,
        array: str | None = None,
        instrument: str | None = None,
        rerun: bool = False,
        log: FilteringBoundLogger | None = None,
    ):
        self.number_to_read = number_to_read
        self.frequency = frequency
        self.array = array
        self.instrument = instrument
        self.rerun = rerun
        self.map_ids = []
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
        query = query.limit(self.number_to_read)
        maps = []

        with mapcat_settings.session() as session:
            results = session.execute(query).scalars().all()
            self.log.info("MapCatDatabaseReader.found_maps", number_found=len(results))

            for result in results:
                if not self.rerun and check_if_processed(
                    result.map_id, session=session
                ):
                    self.log.info(
                        "MapCatDatabaseReader.skipping_processed_map",
                        map_id=result.map_id,
                    )
                    continue
                maps.append(
                    IntensityAndInverseVarianceMap(
                        intensity_filename=mapcat_settings.depth_one_parent
                        / result.map_path,
                        inverse_variance_filename=mapcat_settings.depth_one_parent
                        / result.ivar_path,
                        time_filename=mapcat_settings.depth_one_parent
                        / result.time_path,
                        start_time=datetime.fromtimestamp(
                            result.start_time, tz=timezone.utc
                        ),
                        end_time=datetime.fromtimestamp(
                            result.stop_time, tz=timezone.utc
                        ),
                        intensity_units=u.uK,
                        frequency=result.frequency,
                        array=result.tube_slot,
                        instrument=self.instrument,
                        log=self.log,
                    )
                )
                maps[-1]._map_id = result.map_id
                maps[-1]._parent_database = mapcat_settings.database_name
                self.map_ids.append(result.map_id)
                set_processing_start(result.map_id, session=session)

        return maps

    def __iter__(self):
        return iter(self.map_list())


def check_if_processed(map_id: int, session=None) -> bool:
    ## session is mapcat_settings.session() whatever that is
    if session is None:
        session = mapcat_settings.session()
    query = select(ProcessingStatusTable).where(ProcessingStatusTable.map_id == map_id)
    session_results = session.execute(query).one_or_none()
    if session_results is None:
        return False
    for r in session_results:
        if r.processing_status == "completed":
            return True
    return False


def set_processing_start(map_id: int, session=None):
    ## session is mapcat_settings.session() whatever that is
    if session is None:
        session = mapcat_settings.session()
    query = select(ProcessingStatusTable).where(ProcessingStatusTable.map_id == map_id)
    session_results = session.execute(query).one_or_none()
    if session_results is None:
        session_results = [
            ProcessingStatusTable(processing_status_id=map_id, map_id=map_id)
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
    query = select(ProcessingStatusTable).where(
        ProcessingStatusTable.processing_status_id == map_id
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
