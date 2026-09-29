"""
Read maps from the map tracking database.
"""

from abc import ABC, abstractmethod
from datetime import timezone
from pathlib import Path
from typing import Literal

import uuid7
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time, TimeDelta

# Libraries are loaded here because of the external database
# connection; this only happens once, and we don't want it to
# affect the import time of this module (or need a database to
# be available just to import sotrplib).
from mapcat.database import (
    DepthOneCoaddTable,
    DepthOneMapTable,
    PointingResidualTable,
    SkyCoverageTable,
    TimeDomainProcessingTable,
)
from mapcat.helper import settings as mapcat_settings
from mapcat.pointing.base import PointingModelStats
from mapcat.pointing.const import ConstantPointingModel
from mapcat.pointing.poly import PolynomialPointingModel
from mapcat.toolkit.update_sky_coverage import dec_to_index, ra_to_index
from sqlalchemy import tuple_
from sqlmodel import select
from structlog import get_logger
from structlog.types import FilteringBoundLogger
from uuid7 import UUID as UUID7

from sotrplib.sources.sources import RegisteredSource

from .core import (
    CoaddRhoAndKappaMap,
    FluxAndSNRMap,
    IntensityAndInverseVarianceMap,
    MapCatMapType,
    RhoAndKappaMap,
)
from .pointing import PointingModel

# How build_query() selects the maps for a time window:
#   "restrictive": all of the map is in the window.
#   "loose":       a part of the map is in the window.
#   "left-bound":  the start_time of the map is in the window.
#   "right-bound": the stop_time of the map is in the window.
# See docs/coadding.md.
TimeBinning = Literal["restrictive", "loose", "left-bound", "right-bound"]


def _apply_time_binning(
    query, table, start_time: Time | None, end_time: Time | None, time_binning
):
    """
    Limit `query` to the rows of `table` in [start_time, end_time), as
    `time_binning` sets. `table` is DepthOneMapTable or DepthOneCoaddTable.
    """
    if time_binning == "restrictive":
        # A map that crosses a window boundary goes into no window.
        if start_time is not None:
            query = query.where(
                table.start_time >= start_time.to_datetime(timezone=timezone.utc)
            )
        if end_time is not None:
            query = query.where(
                table.stop_time < end_time.to_datetime(timezone=timezone.utc)
            )
    elif time_binning == "loose":
        # A map that crosses a window boundary goes into the two windows.
        if start_time is not None:
            query = query.where(
                table.stop_time >= start_time.to_datetime(timezone=timezone.utc)
            )
        if end_time is not None:
            query = query.where(
                table.start_time < end_time.to_datetime(timezone=timezone.utc)
            )
    elif time_binning == "left-bound":
        # Each map goes into one window only, with no gaps.
        if start_time is not None:
            query = query.where(
                table.start_time >= start_time.to_datetime(timezone=timezone.utc)
            )
        if end_time is not None:
            query = query.where(
                table.start_time < end_time.to_datetime(timezone=timezone.utc)
            )
    elif time_binning == "right-bound":
        # Same as "left-bound", but with stop_time.
        if start_time is not None:
            query = query.where(
                table.stop_time >= start_time.to_datetime(timezone=timezone.utc)
            )
        if end_time is not None:
            query = query.where(
                table.stop_time < end_time.to_datetime(timezone=timezone.utc)
            )
    else:
        raise ValueError(
            f"Unknown time_binning {time_binning!r}; expected one of "
            "'restrictive', 'loose', 'left-bound', 'right-bound'."
        )
    return query


class MapCatDatabaseReader(ABC):
    """
    Base reader for maps from the map tracking database. Note that the
    database connection is configured through the map catalog library itself.

    Default is to read all arrays and frequencies for the past 1 day
    by setting start_time to 1 day ago and end_time to now.
    """

    instrument: str | None = None
    frequency: str | None = None
    array: str | None = None
    number_to_read: int | None = 1
    start_time: Time | None = None
    end_time: Time | None = None
    map_ids: list[UUID7] | None = None
    sources: list[RegisteredSource] | None = None
    sky_box: tuple[SkyCoord, SkyCoord] | None = None
    intensity_units: u.Unit = u.Unit("K")
    rerun: bool = False
    track_processing: bool = True
    time_binning: TimeBinning = "loose"
    log: FilteringBoundLogger
    default_map_units: u.Unit
    _valid_unit_equivalent: u.Unit

    def __init__(
        self,
        number_to_read: int | None = None,
        start_time: Time | None = None,
        end_time: Time | None = None,
        map_ids: list[UUID7] | None = None,
        sources: list[RegisteredSource] | None = None,
        frequency: str | None = None,
        array: str | None = None,
        instrument: str | None = None,
        sky_box: tuple[SkyCoord, SkyCoord] | None = None,
        map_units: u.Unit | None = None,
        rerun: bool = False,
        rerun_pointing_model: bool = False,
        track_processing: bool = True,
        stale_processing_time: TimeDelta = TimeDelta(2 * 3600, format="sec"),
        time_binning: TimeBinning = "left-bound",
        log: FilteringBoundLogger | None = None,
    ):
        self.number_to_read = number_to_read
        self.start_time = start_time
        self.end_time = end_time
        self.time_binning = time_binning
        self.map_ids = map_ids or []
        self.sources = sources or []
        self.frequency = frequency
        self.array = array
        self.instrument = instrument
        self.map_units = map_units if map_units is not None else self.default_map_units
        self.sky_box = sky_box
        self.rerun = rerun
        self.rerun_pointing_model = rerun_pointing_model
        # If False, do not read or write the processing status. The reader
        # still skips permafail maps. sotrp-coadd uses False.
        self.track_processing = track_processing
        self._map_list = None
        self.stale_processing_time = stale_processing_time
        self.log = log or get_logger()
        self._validate_units()

    def _validate_units(self):
        if not self.map_units.is_equivalent(self._valid_unit_equivalent):
            raise ValueError(
                f"map_units must be equivalent to {self._valid_unit_equivalent} for "
                f"{type(self).__name__}, got {self.map_units}"
            )

    @abstractmethod
    def _build_map(self, result): ...

    def build_query(self):
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

        query = _apply_time_binning(
            query, DepthOneMapTable, self.start_time, self.end_time, self.time_binning
        )

        if self.map_ids:
            query = query.where(DepthOneMapTable.map_id.in_(self.map_ids))

        if self.sources:
            points = []
            for source in self.sources:
                ra = source.ra.to(u.deg).value
                dec = source.dec.to(u.deg).value

                # These aren't covered since ICRS automatically wraps
                # values back aground to 0-360 for RA and -90 to 90 for Dec.
                if ra < 0 or ra > 360:  # pragma: no cover
                    raise ValueError("RA must be between 0 and 360 degrees")
                if dec < -90 or dec > 90:  # pragma: no cover
                    raise ValueError("Dec must be between -90 and 90 degrees")

                ra_idx = ra_to_index(ra)
                dec_idx = dec_to_index(dec)
                points.append((ra_idx, dec_idx))
            points = set(points)
            query = query.join(DepthOneMapTable.depth_one_sky_coverage).where(
                tuple_(SkyCoverageTable.x, SkyCoverageTable.y).in_(points)
            )
        return query

    def map_list(self):
        if self._map_list is not None:
            return self._map_list

        self.log.info(
            "MapCatDatabaseReader.connecting_to_db",
            db_url=mapcat_settings.database_name,
        )

        query = self.build_query()

        maps = []
        with mapcat_settings.session() as session:
            results = session.execute(query).scalars().all()
            self.log.info("MapCatDatabaseReader.found_maps", number_found=len(results))
            if self.number_to_read is None:
                self.number_to_read = len(results)
            for result in results:
                if check_if_permafailed(result.map_id, session=session):
                    self.log.info(
                        "MapCatDatabaseReader.skipping_permafailed_map",
                        map_id=result.map_id,
                    )
                    continue

                if (
                    self.track_processing
                    and not self.rerun
                    and check_if_processed(
                        result.map_id,
                        session=session,
                        stale_limit=self.stale_processing_time,
                    )
                ):
                    self.log.info(
                        "MapCatDatabaseReader.skipping_processed_map",
                        map_id=result.map_id,
                    )
                    continue

                m = self._build_map(result)
                m.mapcat_id = result.map_id
                m._parent_database = mapcat_settings.database_name
                m.pointing_model = (
                    None
                    if self.rerun_pointing_model
                    else load_pointing_model(m.mapcat_id, session=session)
                )
                maps.append(m)
                self.map_ids.append(m.mapcat_id)
                if self.track_processing:
                    set_processing_start(m.mapcat_id, session=session)
                if len(maps) >= self.number_to_read:
                    break
        self._map_list = maps
        return maps

    def __iter__(self):
        return iter(self.map_list())


class IntensityMapReader(MapCatDatabaseReader):
    """Reader for intensity maps, yielding IntensityAndInverseVarianceMap objects."""

    default_map_units = u.Unit("K")
    _valid_unit_equivalent = u.K

    def _build_map(self, result):
        mean_time_path = result.mean_time_path
        if mean_time_path is None:
            mean_time_path = result.map_path.removesuffix("_map.fits") + "_time.fits"

        return IntensityAndInverseVarianceMap(
            intensity_filename=mapcat_settings.depth_one_parent / result.map_path,
            inverse_variance_filename=mapcat_settings.depth_one_parent
            / result.ivar_path,
            time_filename=mapcat_settings.depth_one_parent / mean_time_path,
            start_time=Time(result.start_time),
            end_time=Time(result.stop_time),
            sky_box=self.sky_box,
            intensity_units=self.map_units,
            frequency=result.frequency,
            array=result.tube_slot,
            instrument=self.instrument,
            log=self.log,
        )


class RhoKappaMapReader(MapCatDatabaseReader):
    """Reader for rho/kappa maps, yielding RhoAndKappaMap objects."""

    default_map_units = u.Unit("Jy")
    _valid_unit_equivalent = u.Jy

    def _build_map(self, result):
        return RhoAndKappaMap(
            rho_filename=mapcat_settings.depth_one_parent / result.rho_path,
            kappa_filename=mapcat_settings.depth_one_parent / result.kappa_path,
            time_filename=mapcat_settings.depth_one_parent / result.mean_time_path,
            start_time=Time(result.start_time),
            end_time=Time(result.stop_time),
            sky_box=self.sky_box,
            flux_units=self.map_units,
            frequency=result.frequency,
            array=result.tube_slot,
            instrument=self.instrument,
            log=self.log,
        )


class FluxMapReader(MapCatDatabaseReader):
    """Reader for flux/SNR maps, yielding FluxAndSNRMap objects."""

    default_map_units = u.Unit("Jy")
    _valid_unit_equivalent = u.Jy

    def _build_map(self, result):
        return FluxAndSNRMap(
            flux_filename=mapcat_settings.depth_one_parent / result.flux_path,
            snr_filename=mapcat_settings.depth_one_parent / result.snr_path,
            time_filename=mapcat_settings.depth_one_parent / result.mean_time_path,
            start_time=Time(result.start_time),
            end_time=Time(result.stop_time),
            sky_box=self.sky_box,
            flux_units=self.map_units,
            frequency=result.frequency,
            array=result.tube_slot,
            instrument=self.instrument,
            log=self.log,
        )


class CoaddRhoKappaMapReader(MapCatDatabaseReader):
    """
    Read registered coadds from the depth_one_coadds table in mapcat. Give a
    CoaddRhoAndKappaMap for each coadd.

    - Paths are relative to MAPCAT_DEPTH_ONE_COADD_PARENT.
    - The filters are frequency, time window, `map_ids` (coadd_ids) and
      `coadd_type`. The `array` and `sources` filters are not available.
    - `array` comes from the tube_slots of the linked depth-1 maps.

    See docs/coadding.md.
    """

    default_map_units = u.Unit("Jy")
    _valid_unit_equivalent = u.Jy

    def __init__(self, *args, coadd_type: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.coadd_type = coadd_type
        if self.array is not None:
            raise ValueError(
                "CoaddRhoKappaMapReader can't filter by array: coadds have no "
                "tube_slot. Filter by frequency, time, coadd_type or map_ids."
            )
        if self.sources:
            raise ValueError(
                "CoaddRhoKappaMapReader can't filter by sources: coadds have no "
                "sky-coverage rows."
            )

    def build_query(self):
        query = select(DepthOneCoaddTable)
        if self.frequency:
            query = query.where(DepthOneCoaddTable.frequency == self.frequency)
        if self.coadd_type:
            query = query.where(DepthOneCoaddTable.coadd_type == self.coadd_type)
        query = _apply_time_binning(
            query,
            DepthOneCoaddTable,
            self.start_time,
            self.end_time,
            self.time_binning,
        )
        if self.map_ids:
            query = query.where(DepthOneCoaddTable.coadd_id.in_(self.map_ids))
        return query.order_by(DepthOneCoaddTable.start_time)

    def _build_map(self, result):
        tube_slots = {m.tube_slot for m in result.maps if m.tube_slot}
        coadd_parent = mapcat_settings.depth_one_coadd_parent
        return CoaddRhoAndKappaMap(
            rho_filename=coadd_parent / result.rho_path,
            kappa_filename=coadd_parent / result.kappa_path,
            time_filename=(
                coadd_parent / result.mean_time_path if result.mean_time_path else None
            ),
            start_time=Time(result.start_time),
            end_time=Time(result.stop_time),
            sky_box=self.sky_box,
            flux_units=self.map_units,
            frequency=result.frequency,
            array="".join(sorted(tube_slots)) or None,
            instrument=self.instrument,
            log=self.log,
        )

    def map_list(self):
        if self._map_list is not None:
            return self._map_list

        self.log.info(
            "CoaddRhoKappaMapReader.connecting_to_db",
            db_url=mapcat_settings.database_name,
        )

        maps = []
        with mapcat_settings.session() as session:
            results = session.execute(self.build_query()).scalars().all()
            self.log.info(
                "CoaddRhoKappaMapReader.found_coadds", number_found=len(results)
            )
            if self.number_to_read is None:
                self.number_to_read = len(results)
            for result in results:
                if check_if_permafailed(
                    result.coadd_id, map_type="coadd", session=session
                ):
                    self.log.info(
                        "CoaddRhoKappaMapReader.skipping_permafailed_coadd",
                        coadd_id=result.coadd_id,
                    )
                    continue

                if (
                    self.track_processing
                    and not self.rerun
                    and check_if_processed(
                        result.coadd_id,
                        map_type="coadd",
                        session=session,
                        stale_limit=self.stale_processing_time,
                    )
                ):
                    self.log.info(
                        "CoaddRhoKappaMapReader.skipping_processed_coadd",
                        coadd_id=result.coadd_id,
                    )
                    continue

                m = self._build_map(result)
                m.mapcat_id = result.coadd_id
                m._parent_database = mapcat_settings.database_name
                m.pointing_model = None
                maps.append(m)
                self.map_ids.append(m.mapcat_id)
                if self.track_processing:
                    set_processing_start(m.mapcat_id, map_type="coadd", session=session)
                if len(maps) >= self.number_to_read:
                    break
        self._map_list = maps
        return maps


def _get_processing_column(map_type: MapCatMapType):
    """Return the TimeDomainProcessingTable column for `map_type`."""
    if map_type == "depth1_map":
        return TimeDomainProcessingTable.map_id
    if map_type == "coadd":
        return TimeDomainProcessingTable.coadd_id
    raise ValueError(
        f"Unknown map_type {map_type!r}; expected 'depth1_map' or 'coadd'."
    )


def _get_processing_row(
    mapcat_id: UUID7,
    *,
    map_type: MapCatMapType = "depth1_map",
    session,
) -> TimeDomainProcessingTable | None:
    column = _get_processing_column(map_type)
    query = select(TimeDomainProcessingTable).where(column == mapcat_id)
    result = session.execute(query).one_or_none()
    if result is None:
        return None
    for r in result:
        return r
    return None


def check_if_permafailed(
    mapcat_id: UUID7,
    *,
    map_type: MapCatMapType = "depth1_map",
    session=None,
) -> bool:
    """
    Return True if the status of the map or coadd is "permafail". The
    pipeline skips these maps, also with rerun=True. Only a person sets this
    status (`mapcatreset --status permafail`).
    """
    if session is None:
        session = mapcat_settings.session()
    row = _get_processing_row(mapcat_id, map_type=map_type, session=session)
    return row is not None and row.processing_status == "permafail"


def check_if_processed(
    mapcat_id: UUID7,
    *,
    map_type: MapCatMapType = "depth1_map",
    session=None,
    completed_status: str = "completed",
    processing_status: str = "processing",
    stale_limit: TimeDelta = TimeDelta(2 * 3600, format="sec"),
) -> bool:
    ## session is mapcat_settings.session() whatever that is
    if session is None:
        session = mapcat_settings.session()
    row = _get_processing_row(mapcat_id, map_type=map_type, session=session)
    if row is None:
        return False
    if row.processing_status == completed_status:
        return True
    # Use astropy Time, because sqlmodel >= 0.0.43 gives UTC-aware datetimes
    # and older versions give naive datetimes. Time() uses UTC for both.
    if row.processing_status == processing_status and (
        (Time.now() - Time(row.processing_start)).to_value("s")
        < stale_limit.to_value("s")
    ):
        return True
    return False


def set_processing_start(
    mapcat_id: UUID7,
    *,
    map_type: MapCatMapType = "depth1_map",
    session=None,
):
    ## session is mapcat_settings.session() whatever that is
    if session is None:
        session = mapcat_settings.session()
    row = _get_processing_row(mapcat_id, map_type=map_type, session=session)
    if row is None:
        row = TimeDomainProcessingTable(
            processing_status_id=uuid7.create(),
            map_id=mapcat_id if map_type == "depth1_map" else None,
            coadd_id=mapcat_id if map_type == "coadd" else None,
        )
    row.processing_start = Time.now().to_datetime(timezone=timezone.utc)
    row.processing_status = "processing"
    session.add(row)
    session.commit()
    return


def load_pointing_model(map_id: UUID7, session=None) -> PointingModel | None:
    """Load a pointing model from the DB for a given map, or None if not found."""
    if session is None:
        session = mapcat_settings.session()
    query = select(PointingResidualTable).where(PointingResidualTable.map_id == map_id)
    result = session.execute(query).one_or_none()

    if result is None:
        return None
    for row in result:
        if row.residual_model.model_type == "constant":
            return ConstantPointingModel(
                ra_offset=row.residual_model.ra_offset,
                dec_offset=row.residual_model.dec_offset,
            )
        elif row.residual_model.model_type == "polynomial":
            return PolynomialPointingModel(
                poly_order=row.residual_model.poly_order,
                ra_model_coefficients=row.residual_model.ra_model_coefficients,
                dec_model_coefficients=row.residual_model.dec_model_coefficients,
            )
    return None


def save_pointing_model(
    map_id: UUID7,
    pointing_model: PointingModel,
    pointing_model_stats: PointingModelStats,
    session=None,
) -> None:
    """
    Serialize a PointingModel subclass to PointingResidualTable.

    Supports ``ConstantPointingModel`` and ``PolynomialPointingModel``.
    Raises ``ValueError`` for unsupported pointing model types.
    """
    if session is None:
        session = mapcat_settings.session()
    if isinstance(pointing_model, ConstantPointingModel):
        model = ConstantPointingModel(
            ra_offset=pointing_model.ra_offset,
            dec_offset=pointing_model.dec_offset,
        )
    elif isinstance(pointing_model, PolynomialPointingModel):
        model = PolynomialPointingModel(
            poly_order=pointing_model.poly_order,
            ra_model_coefficients=pointing_model.ra_model_coefficients,
            dec_model_coefficients=pointing_model.dec_model_coefficients,
        )
    else:
        raise ValueError(
            f"Unsupported pointing model type {type(pointing_model)} for saving to DB."
        )
    query = select(PointingResidualTable).where(PointingResidualTable.map_id == map_id)
    result = session.execute(query).one_or_none()
    if result is None:
        result = [
            PointingResidualTable(
                map_id=map_id,
                residual_model=model,
                residual_stats=pointing_model_stats,
            )
        ]
    for row in result:
        row.residual_model = model
        row.residual_stats = pointing_model_stats
        session.add(row)
        session.commit()


def set_processing_end(
    mapcat_id: UUID7,
    *,
    map_type: MapCatMapType = "depth1_map",
    session=None,
    status: str = "completed",
):
    """
    Set the processing end time and `status` ("completed" or "failed") of a
    map or coadd. Do not use this function to set "permafail".
    """
    ## session is mapcat_settings.session() whatever that is
    if session is None:
        session = mapcat_settings.session()
    row = _get_processing_row(mapcat_id, map_type=map_type, session=session)
    if row is None:
        raise ValueError(
            f"No processing_start status found for {map_type} {mapcat_id} "
            "when trying to set processing_end."
        )
    row.processing_end = Time.now().to_datetime(timezone=timezone.utc)
    row.processing_status = status
    session.add(row)
    session.commit()
    return


def register_coadd(
    coadd,
    map_ids: list[UUID7],
    coadd_name: str,
    coadd_type: str,
    output_paths: dict[str, Path],
    session=None,
) -> UUID7:
    """
    Register a coadd in mapcat, and link it to its depth-1 maps.

    Parameters
    ----------
    coadd : CoaddedRhoKappaMap
        The coadd. It must have frequency, observation_start and
        observation_end.
    map_ids : list[UUID7]
        The map_id of each depth-1 map in the coadd.
    coadd_name : str
        A unique name for the coadd.
    coadd_type : str
        A tag for the type of coadd.
    output_paths : dict[str, Path]
        The FITS paths of the coadd, by field name. "flux" is necessary
        (it is the map_path). "rho", "kappa", "time_first", "time_mean" and
        "time_last" are optional. Paths are stored relative to
        MAPCAT_DEPTH_ONE_COADD_PARENT.

    Returns
    -------
    The coadd_id of the new row.
    """
    if session is None:
        session = mapcat_settings.session()

    coadd_parent = mapcat_settings.depth_one_coadd_parent

    def _rel(key: str) -> str | None:
        path = output_paths.get(key)
        return str(Path(path).relative_to(coadd_parent)) if path else None

    row = DepthOneCoaddTable(
        coadd_name=coadd_name,
        coadd_type=coadd_type,
        map_path=_rel("flux"),
        ivar_path=_rel("kappa"),
        rho_path=_rel("rho"),
        kappa_path=_rel("kappa"),
        start_time_path=_rel("time_first"),
        mean_time_path=_rel("time_mean"),
        end_time_path=_rel("time_last"),
        frequency=coadd.frequency,
        ctime=(
            coadd.observation_start
            + (coadd.observation_end - coadd.observation_start) / 2
        ).to_datetime(timezone=timezone.utc),
        start_time=coadd.observation_start.to_datetime(timezone=timezone.utc),
        stop_time=coadd.observation_end.to_datetime(timezone=timezone.utc),
    )

    if map_ids:
        query = select(DepthOneMapTable).where(DepthOneMapTable.map_id.in_(map_ids))
        row.maps = list(session.execute(query).scalars().all())

    session.add(row)
    session.commit()
    session.refresh(row)
    return row.coadd_id
