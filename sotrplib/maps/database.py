"""
Read maps from the map tracking database.
"""

from datetime import datetime, timezone

from structlog import get_logger
from structlog.types import FilteringBoundLogger

from .core import IntensityAndInverseVarianceMap


class MapCatDatabaseReader:
    """
    Reader for maps from the map tracking database. Note that the
    database connection is configured through the map catalog library
    itself.
    """

    number_to_read: int = 1
    log: FilteringBoundLogger

    def __init__(
        self, number_to_read: int = 1, log: FilteringBoundLogger | None = None
    ):
        self.number_to_read = number_to_read
        self.log = log or get_logger()

    def map_list(self):
        # Libraries are loaded here because of the external database
        # connection; this only happens once, and we don't want it to
        # affect the import time of this module (or need a database to
        # be available just to import sotrplib).
        from mapcat.database import DepthOneMapTable
        from mapcat.helper import settings as mapcat_settings
        from sqlmodel import select

        query = select(DepthOneMapTable).limit(self.number_to_read)
        maps = []

        with mapcat_settings.session() as session:
            results = session.execute(query).scalars().all()

            for result in results:
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
                        frequency="f" + result.frequency,
                        array=result.tube_slot,
                        log=self.log,
                    )
                )

        return maps

    def __iter__(self):
        return iter(self.map_list())
