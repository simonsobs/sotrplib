"""
Read maps from the map tracking database.
"""

import uuid
from abc import ABC, abstractmethod
from pathlib import Path

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

from sotrplib.sources.sources import RegisteredSource

from .core import FluxAndSNRMap, IntensityAndInverseVarianceMap, RhoAndKappaMap
from .pointing import PointingModel


def _to_uuid(value: str | uuid.UUID | None) -> uuid.UUID | None:
    """
    Coerce a map_id/coadd_id value to a real uuid.UUID before it crosses
    into mapcat's ORM layer. mapcat's map_id/coadd_id columns use
    sa.Uuid() (as_uuid=True), whose bind processor expects an actual
    uuid.UUID instance and errors (AttributeError: 'str' object has no
    attribute 'hex') if handed a plain string. Needed both for query
    filters (`.where(Column == value)`/`.in_(values)`) and for
    constructing rows directly (e.g. TimeDomainProcessingTable(map_id=...))
    -- SQLModel's table=True classes, unlike plain Pydantic models, don't
    validate/coerce field values on construction, so a string assigned to a
    UUID-typed field is stored as-is rather than converted.
    """
    if value is None or isinstance(value, uuid.UUID):
        return value
    return uuid.UUID(str(value))


def _to_uuid_list(values) -> list[uuid.UUID]:
    return [_to_uuid(v) for v in values]


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
    map_ids: list[str] | None = None
    sources: list[RegisteredSource] | None = None
    sky_box: tuple[SkyCoord, SkyCoord] | None = None
    intensity_units: u.Unit = u.Unit("K")
    rerun: bool = False
    log: FilteringBoundLogger
    default_map_units: u.Unit
    _valid_unit_equivalent: u.Unit

    def __init__(
        self,
        number_to_read: int | None = None,
        start_time: Time | None = None,
        end_time: Time | None = None,
        map_ids: list[str] | None = None,
        sources: list[RegisteredSource] | None = None,
        frequency: str | None = None,
        array: str | None = None,
        instrument: str | None = None,
        sky_box: tuple[SkyCoord, SkyCoord] | None = None,
        map_units: u.Unit | None = None,
        rerun: bool = False,
        rerun_pointing_model: bool = False,
        stale_processing_time: TimeDelta = TimeDelta(2 * 3600, format="sec"),
        bucket_by_start_time: bool = False,
        log: FilteringBoundLogger | None = None,
    ):
        self.number_to_read = number_to_read
        self.start_time = start_time
        self.end_time = end_time
        self.bucket_by_start_time = bucket_by_start_time
        self.map_ids = map_ids or []
        self.sources = sources or []
        self.frequency = frequency
        self.array = array
        self.instrument = instrument
        self.map_units = map_units if map_units is not None else self.default_map_units
        self.sky_box = sky_box
        self.rerun = rerun
        self.rerun_pointing_model = rerun_pointing_model
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

        if self.bucket_by_start_time:
            # Assign each map to exactly one caller-defined window based on its
            # own start_time (half-open [start_time, end_time)), rather than
            # any overlap with [start_time, end_time]. The overlap test below
            # double-counts any map whose observation spans a shared boundary
            # between adjacent windows (e.g. weekly coadd windows); bucketing
            # by start_time alone can't, since every map has exactly one.
            if self.start_time is not None:
                query = query.where(DepthOneMapTable.start_time >= self.start_time.unix)
            if self.end_time is not None:
                query = query.where(DepthOneMapTable.start_time < self.end_time.unix)
        else:
            if self.start_time is not None:
                query = query.where(DepthOneMapTable.stop_time >= self.start_time.unix)
            if self.end_time is not None:
                query = query.where(DepthOneMapTable.start_time <= self.end_time.unix)

        if self.map_ids:
            query = query.where(
                DepthOneMapTable.map_id.in_(_to_uuid_list(self.map_ids))
            )

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

                m = self._build_map(result)
                m.map_id = str(result.map_id)
                m._parent_database = mapcat_settings.database_name
                m.pointing_model = (
                    None
                    if self.rerun_pointing_model
                    else load_pointing_model(m.map_id, session=session)
                )
                maps.append(m)
                self.map_ids.append(m.map_id)
                set_processing_start(m.map_id, session=session)
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
        return IntensityAndInverseVarianceMap(
            intensity_filename=mapcat_settings.depth_one_parent / result.map_path,
            inverse_variance_filename=mapcat_settings.depth_one_parent
            / result.ivar_path,
            time_filename=mapcat_settings.depth_one_parent / result.mean_time_path,
            start_time=Time(result.start_time, format="unix"),
            end_time=Time(result.stop_time, format="unix"),
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
            start_time=Time(result.start_time, format="unix"),
            end_time=Time(result.stop_time, format="unix"),
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
            start_time=Time(result.start_time, format="unix"),
            end_time=Time(result.stop_time, format="unix"),
            sky_box=self.sky_box,
            flux_units=self.map_units,
            frequency=result.frequency,
            array=result.tube_slot,
            instrument=self.instrument,
            log=self.log,
        )


class CoaddRhoKappaMapReader:
    """
    Reader for depth-1 coadds (mapcat's depth_one_coadds table), yielding
    RhoAndKappaMap objects. Not a MapCatDatabaseReader subclass -- coadds
    have no tube_slot/array column and a different processing-status target
    (coadd_id, not map_id), so build_query()/map_list() are different enough
    that sharing the base class would fight it more than it would help.

    Coadds are already non-overlapping in time by construction (see
    submit_week_coadds.py's bucket_by_start_time), so selection is either by
    explicit coadd_id, an explicit start/end range (interval overlap, like
    MapCatDatabaseReader's default), or a single instant (target_time) that
    should fall within exactly one coadd's [start_time, stop_time] --
    there's no windowing/double-count concern here the way there is for
    depth-1 maps, so plain interval containment is correct and sufficient.

    default_map_units is mJy, not Jy: coadd rho/kappa FITS files are built
    by coadding matched-filtered depth-1 maps (see
    sotrplib.maps.preprocessor.MatchedFilter), whose output flux_units is
    hardcoded to mJy -- and RhoKappaMapCoadder.coadd_maps() carries that
    label through to the coadd unchanged. FITS files don't persist astropy
    unit metadata, so a reader with the wrong default silently mislabels
    every flux downstream by 1000x (mJy-scale numbers reported as Jy).
    """

    default_map_units = u.Unit("mJy")
    _valid_unit_equivalent = u.Jy

    def __init__(
        self,
        frequency: str | None = None,
        instrument: str | None = None,
        coadd_type: str | None = None,
        coadd_ids: list[str] | None = None,
        target_time: Time | None = None,
        start_time: Time | None = None,
        end_time: Time | None = None,
        number_to_read: int | None = None,
        map_units: u.Unit | None = None,
        rerun: bool = False,
        stale_processing_time: TimeDelta = TimeDelta(2 * 3600, format="sec"),
        log: FilteringBoundLogger | None = None,
    ):
        self.frequency = frequency
        self.instrument = instrument
        self.coadd_type = coadd_type
        self.coadd_ids = coadd_ids or []
        self.target_time = target_time
        self.start_time = start_time
        self.end_time = end_time
        self.number_to_read = number_to_read
        self.map_units = map_units if map_units is not None else self.default_map_units
        self.rerun = rerun
        self.stale_processing_time = stale_processing_time
        self._map_list = None
        self.log = log or get_logger()
        self._validate_units()

    def _validate_units(self):
        if not self.map_units.is_equivalent(self._valid_unit_equivalent):
            raise ValueError(
                f"map_units must be equivalent to {self._valid_unit_equivalent} for "
                f"{type(self).__name__}, got {self.map_units}"
            )

    def build_query(self):
        query = select(DepthOneCoaddTable)

        if self.frequency:
            query = query.where(DepthOneCoaddTable.frequency == self.frequency)
        if self.coadd_type:
            query = query.where(DepthOneCoaddTable.coadd_type == self.coadd_type)
        if self.coadd_ids:
            query = query.where(
                DepthOneCoaddTable.coadd_id.in_(_to_uuid_list(self.coadd_ids))
            )

        if self.target_time is not None:
            query = query.where(
                DepthOneCoaddTable.start_time <= self.target_time.unix,
                DepthOneCoaddTable.stop_time >= self.target_time.unix,
            )
        else:
            if self.start_time is not None:
                query = query.where(
                    DepthOneCoaddTable.stop_time >= self.start_time.unix
                )
            if self.end_time is not None:
                query = query.where(DepthOneCoaddTable.start_time <= self.end_time.unix)
        return query

    def _build_map(self, result) -> RhoAndKappaMap:
        return RhoAndKappaMap(
            rho_filename=mapcat_settings.depth_one_parent / result.rho_path,
            kappa_filename=mapcat_settings.depth_one_parent / result.kappa_path,
            time_filename=mapcat_settings.depth_one_parent / result.mean_time_path,
            start_time=Time(result.start_time, format="unix"),
            end_time=Time(result.stop_time, format="unix"),
            flux_units=self.map_units,
            frequency=result.frequency,
            array=None,
            instrument=self.instrument,
            # Coadd time maps (CoaddedRhoKappaMap.update_times) are written
            # as absolute unix time already, unlike depth-1 time maps
            # (seconds-since-observation-start) -- RhoAndKappaMap.build()
            # must not apply add_time_offset() a second time here.
            time_is_relative=False,
            log=self.log,
        )

    def map_list(self):
        if self._map_list is not None:
            return self._map_list

        self.log.info(
            "CoaddRhoKappaMapReader.connecting_to_db",
            db_url=mapcat_settings.database_name,
        )

        query = self.build_query()

        maps = []
        with mapcat_settings.session() as session:
            results = session.execute(query).scalars().all()
            self.log.info(
                "CoaddRhoKappaMapReader.found_coadds", number_found=len(results)
            )
            if self.number_to_read is None:
                self.number_to_read = len(results)
            for result in results:
                if check_if_permafailed(coadd_id=result.coadd_id, session=session):
                    self.log.info(
                        "CoaddRhoKappaMapReader.skipping_permafailed_coadd",
                        coadd_id=result.coadd_id,
                    )
                    continue

                if not self.rerun and check_if_processed(
                    coadd_id=result.coadd_id,
                    session=session,
                    stale_limit=self.stale_processing_time,
                ):
                    self.log.info(
                        "CoaddRhoKappaMapReader.skipping_processed_coadd",
                        coadd_id=result.coadd_id,
                    )
                    continue

                m = self._build_map(result)
                m.map_id = str(result.coadd_id)
                m.is_coadd = True
                m._parent_database = mapcat_settings.database_name
                maps.append(m)
                self.coadd_ids.append(m.map_id)
                set_processing_start(coadd_id=m.map_id, session=session)
                if len(maps) >= self.number_to_read:
                    break
        self._map_list = maps
        return maps

    def __iter__(self):
        return iter(self.map_list())


def _resolve_processing_target(
    map_id: str | uuid.UUID | None, coadd_id: str | uuid.UUID | None
) -> tuple:
    """Validate exactly one of map_id/coadd_id was given and return the
    (column, value) pair to filter TimeDomainProcessingTable on."""
    if (map_id is None) == (coadd_id is None):
        raise ValueError("Exactly one of map_id or coadd_id must be provided.")
    if map_id is not None:
        return TimeDomainProcessingTable.map_id, _to_uuid(map_id)
    return TimeDomainProcessingTable.coadd_id, _to_uuid(coadd_id)


def _get_processing_row(
    map_id: str | uuid.UUID | None = None,
    *,
    coadd_id: str | uuid.UUID | None = None,
    session,
) -> TimeDomainProcessingTable | None:
    column, value = _resolve_processing_target(map_id, coadd_id)
    query = select(TimeDomainProcessingTable).where(column == value)
    result = session.execute(query).one_or_none()
    if result is None:
        return None
    for r in result:
        return r
    return None


def check_if_permafailed(
    map_id: str | uuid.UUID | None = None,
    *,
    coadd_id: str | uuid.UUID | None = None,
    session=None,
) -> bool:
    """
    True if a map or coadd has been manually marked "permafail" (e.g. via
    mapcat's `mapcatreset --status permafail`) -- a known-pathological
    observation that should never be picked up again by the pipeline, even
    with rerun=True. This is the only status the pipeline itself never
    sets; it's set by a human once something is known to be permanently
    unusable.
    """
    if session is None:
        session = mapcat_settings.session()
    row = _get_processing_row(map_id, coadd_id=coadd_id, session=session)
    return row is not None and row.processing_status == "permafail"


def check_if_processed(
    map_id: str | uuid.UUID | None = None,
    *,
    coadd_id: str | uuid.UUID | None = None,
    session=None,
    completed_status: str = "completed",
    processing_status: str = "processing",
    stale_limit: TimeDelta = TimeDelta(2 * 3600, format="sec"),
) -> bool:
    ## session is mapcat_settings.session() whatever that is
    if session is None:
        session = mapcat_settings.session()
    row = _get_processing_row(map_id, coadd_id=coadd_id, session=session)
    if row is None:
        return False
    if row.processing_status == completed_status:
        return True
    if row.processing_status == processing_status and (
        Time.now().unix - row.processing_start
    ) < stale_limit.to_value("s"):
        return True
    return False


def set_processing_start(
    map_id: str | uuid.UUID | None = None,
    *,
    coadd_id: str | uuid.UUID | None = None,
    session=None,
):
    ## session is mapcat_settings.session() whatever that is
    if session is None:
        session = mapcat_settings.session()
    row = _get_processing_row(map_id, coadd_id=coadd_id, session=session)
    if row is None:
        # SQLModel table=True classes don't validate/coerce field values on
        # construction (unlike plain Pydantic models), so a string map_id
        # would otherwise be stored as-is and break at INSERT time when
        # SQLAlchemy's Uuid bind processor expects a real uuid.UUID.
        row = TimeDomainProcessingTable(
            map_id=_to_uuid(map_id), coadd_id=_to_uuid(coadd_id)
        )
    row.processing_start = Time.now().unix
    row.processing_status = "processing"
    session.add(row)
    session.commit()
    return


def load_pointing_model(map_id: str | uuid.UUID, session=None) -> PointingModel | None:
    """Load a pointing model from the DB for a given map, or None if not found."""
    if session is None:
        session = mapcat_settings.session()
    query = select(PointingResidualTable).where(
        PointingResidualTable.map_id == _to_uuid(map_id)
    )
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
    map_id: str | uuid.UUID,
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
    query = select(PointingResidualTable).where(
        PointingResidualTable.map_id == _to_uuid(map_id)
    )
    result = session.execute(query).one_or_none()
    if result is None:
        result = [
            PointingResidualTable(
                map_id=_to_uuid(map_id),
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
    map_id: str | uuid.UUID | None = None,
    *,
    coadd_id: str | uuid.UUID | None = None,
    session=None,
    status: str = "completed",
):
    """
    Mark a map's or coadd's processing as finished, with the given terminal
    status (default "completed"; pass status="failed" when the caller is
    handling an exception). "permafail" is intentionally not set here --
    it's a manual-only status for known-pathological observations (see
    check_if_permafailed), never set automatically by the pipeline.
    """
    ## session is mapcat_settings.session() whatever that is
    if session is None:
        session = mapcat_settings.session()
    row = _get_processing_row(map_id, coadd_id=coadd_id, session=session)
    if row is None:
        target = f"map_id {map_id}" if map_id is not None else f"coadd_id {coadd_id}"
        raise ValueError(
            f"No processing_start status found for {target} when trying to set processing_end."
        )
    row.processing_end = Time.now().unix
    row.processing_status = status
    session.add(row)
    session.commit()
    return


def register_coadd(
    coadd,
    map_ids: list[str | uuid.UUID],
    coadd_name: str,
    coadd_type: str,
    output_paths: dict[str, Path],
    session=None,
) -> uuid.UUID:
    """
    Register a finished coadd, and link it to the depth-1 maps that went
    into it, in the mapcat database.

    Parameters
    ----------
    coadd : CoaddedRhoKappaMap
        The finished coadd (must have frequency/observation_start/
        observation_end set).
    map_ids : list[str | uuid.UUID]
        map_id of every depth-1 map merged into this coadd (e.g. as
        returned by sotrplib.maps.streaming_coadd.stream_coadd).
    coadd_name : str
        Human-readable, ideally unique name for this coadd.
    coadd_type : str
        Free-form tag describing how this coadd was produced.
    output_paths : dict[str, Path]
        FITS paths already written to disk, keyed by field name: "flux"
        (used as the required map_path, matching DepthOneMapTable's
        "first available of intensity, rho, flux" coverage-map
        convention), and optionally "rho", "kappa", "time_first",
        "time_mean", "time_last". Stored relative to
        MAPCAT_DEPTH_ONE_PARENT, matching how depth-1 maps are recorded.

    Returns
    -------
    coadd_id of the new row.
    """
    if session is None:
        session = mapcat_settings.session()

    depth_one_parent = mapcat_settings.depth_one_parent

    def _rel(key: str) -> str | None:
        path = output_paths.get(key)
        return str(Path(path).relative_to(depth_one_parent)) if path else None

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
        ctime=0.5 * (coadd.observation_start.unix + coadd.observation_end.unix),
        start_time=coadd.observation_start.unix,
        stop_time=coadd.observation_end.unix,
    )

    if map_ids:
        query = select(DepthOneMapTable).where(
            DepthOneMapTable.map_id.in_(_to_uuid_list(map_ids))
        )
        row.maps = list(session.execute(query).scalars().all())

    session.add(row)
    session.commit()
    session.refresh(row)
    return row.coadd_id
