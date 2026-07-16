"""
Tests for CoaddRhoKappaMapReader (reads mapcat's depth_one_coadds table).
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from astropy import units as u
from astropy.time import Time

from sotrplib.maps.core import RhoAndKappaMap
from sotrplib.maps.database import CoaddRhoKappaMapReader


@pytest.fixture
def db_coadd_result(separate_map_set_1):
    """Mock DepthOneCoaddTable row pointing at the tmp FITS files."""
    paths = separate_map_set_1
    r = MagicMock()
    r.coadd_id = "11111111-1111-1111-1111-111111111111"
    r.rho_path = paths["rho"]
    r.kappa_path = paths["kappa"]
    r.mean_time_path = paths["time"]
    r.start_time = Time.now().unix - 3600
    r.stop_time = Time.now().unix
    r.frequency = "f090"
    r.coadd_type = "depth1_streaming_coadd"
    return r


@pytest.fixture
def mock_coadd_session(db_coadd_result):
    session = MagicMock()
    session.execute.return_value.scalars.return_value.all.return_value = [
        db_coadd_result
    ]
    return session


@pytest.fixture
def mock_mapcat_coadd(mock_coadd_session):
    """
    Patches mapcat_settings, check_if_processed, and set_processing_start so
    that map_list() runs without a real database connection.
    """
    with (
        patch("sotrplib.maps.database.mapcat_settings") as settings,
        patch("sotrplib.maps.database.check_if_processed", return_value=False),
        patch("sotrplib.maps.database.check_if_permafailed", return_value=False),
        patch("sotrplib.maps.database.set_processing_start"),
    ):
        settings.database_name = "test_db"
        settings.depth_one_parent = Path("/")
        settings.session.return_value.__enter__.return_value = mock_coadd_session
        yield settings


# ─── unit validation ────────────────────────────────────────────────────────────


def test_coadd_reader_default_units():
    assert CoaddRhoKappaMapReader().map_units.is_equivalent(u.Jy)


def test_coadd_reader_rejects_k():
    with pytest.raises(ValueError, match="CoaddRhoKappaMapReader"):
        CoaddRhoKappaMapReader(map_units=u.K)


# ─── _build_map ─────────────────────────────────────────────────────────────────


def test_coadd_build_map_type(db_coadd_result):
    with patch("sotrplib.maps.database.mapcat_settings") as settings:
        settings.depth_one_parent = Path("/")
        result = CoaddRhoKappaMapReader()._build_map(db_coadd_result)
    assert isinstance(result, RhoAndKappaMap)
    assert result.frequency == "f090"
    assert result.array is None


# ─── build_query ────────────────────────────────────────────────────────────────


def test_coadd_build_query_target_time_is_containment():
    """
    A single instant should be matched via interval containment
    (start_time <= target_time <= stop_time), not overlap -- coadds are
    already non-overlapping in time, so containment is unambiguous.
    """
    reader = CoaddRhoKappaMapReader(target_time=Time(1500, format="unix"))
    query = reader.build_query()
    compiled = str(query.compile(compile_kwargs={"literal_binds": True}))
    where_clause = compiled.split("WHERE", 1)[1]
    assert "depth_one_coadds.start_time <=" in where_clause
    assert "depth_one_coadds.stop_time >=" in where_clause


def test_coadd_build_query_default_is_range_overlap():
    """Without target_time, falls back to an explicit start/end interval
    overlap test, like MapCatDatabaseReader's default."""
    reader = CoaddRhoKappaMapReader(
        start_time=Time(1000, format="unix"), end_time=Time(2000, format="unix")
    )
    query = reader.build_query()
    compiled = str(query.compile(compile_kwargs={"literal_binds": True}))
    where_clause = compiled.split("WHERE", 1)[1]
    assert "depth_one_coadds.stop_time >=" in where_clause
    assert "depth_one_coadds.start_time <=" in where_clause


def test_coadd_build_query_filters_by_frequency_and_coadd_type():
    reader = CoaddRhoKappaMapReader(
        frequency="f090", coadd_type="depth1_streaming_coadd"
    )
    query = reader.build_query()
    compiled = str(query.compile(compile_kwargs={"literal_binds": True}))
    assert "depth_one_coadds.frequency = 'f090'" in compiled
    assert "depth_one_coadds.coadd_type = 'depth1_streaming_coadd'" in compiled


# ─── map_list ───────────────────────────────────────────────────────────────────


def test_coadd_map_list_sets_is_coadd_and_map_id(mock_mapcat_coadd, db_coadd_result):
    maps = CoaddRhoKappaMapReader().map_list()
    assert len(maps) == 1
    assert maps[0].is_coadd is True
    assert maps[0].map_id == str(db_coadd_result.coadd_id)


def test_coadd_map_list_appends_coadd_id(mock_mapcat_coadd, db_coadd_result):
    reader = CoaddRhoKappaMapReader()
    reader.map_list()
    assert str(db_coadd_result.coadd_id) in reader.coadd_ids


def test_coadd_map_list_skips_permafailed(mock_coadd_session):
    with (
        patch("sotrplib.maps.database.mapcat_settings") as settings,
        patch("sotrplib.maps.database.check_if_permafailed", return_value=True),
        patch("sotrplib.maps.database.check_if_processed", return_value=False),
        patch("sotrplib.maps.database.set_processing_start"),
    ):
        settings.database_name = "test_db"
        settings.depth_one_parent = Path("/")
        settings.session.return_value.__enter__.return_value = mock_coadd_session
        maps = CoaddRhoKappaMapReader(rerun=True).map_list()
    assert maps == []


def test_coadd_map_list_skips_processed(mock_coadd_session):
    with (
        patch("sotrplib.maps.database.mapcat_settings") as settings,
        patch("sotrplib.maps.database.check_if_permafailed", return_value=False),
        patch("sotrplib.maps.database.check_if_processed", return_value=True),
        patch("sotrplib.maps.database.set_processing_start"),
    ):
        settings.database_name = "test_db"
        settings.depth_one_parent = Path("/")
        settings.session.return_value.__enter__.return_value = mock_coadd_session
        maps = CoaddRhoKappaMapReader().map_list()
    assert maps == []


def test_coadd_map_list_rerun_ignores_processed(mock_coadd_session):
    with (
        patch("sotrplib.maps.database.mapcat_settings") as settings,
        patch("sotrplib.maps.database.check_if_permafailed", return_value=False),
        patch(
            "sotrplib.maps.database.check_if_processed", return_value=True
        ) as mock_check,
        patch("sotrplib.maps.database.set_processing_start"),
    ):
        settings.database_name = "test_db"
        settings.depth_one_parent = Path("/")
        settings.session.return_value.__enter__.return_value = mock_coadd_session
        maps = CoaddRhoKappaMapReader(rerun=True).map_list()
    mock_check.assert_not_called()
    assert len(maps) == 1


def test_coadd_map_list_calls_status_functions_with_coadd_id_kwarg(
    mock_coadd_session, db_coadd_result
):
    """set_processing_start/check_if_permafailed/check_if_processed must be
    called with coadd_id=, not map_id= -- using the wrong kwarg would create
    a status row that doesn't match the coadd being tracked."""
    with (
        patch("sotrplib.maps.database.mapcat_settings") as settings,
        patch(
            "sotrplib.maps.database.check_if_permafailed", return_value=False
        ) as mock_permafail,
        patch("sotrplib.maps.database.check_if_processed", return_value=False),
        patch("sotrplib.maps.database.set_processing_start") as mock_start,
    ):
        settings.database_name = "test_db"
        settings.depth_one_parent = Path("/")
        settings.session.return_value.__enter__.return_value = mock_coadd_session
        CoaddRhoKappaMapReader().map_list()

    assert mock_permafail.call_args.kwargs["coadd_id"] == db_coadd_result.coadd_id
    assert mock_start.call_args.kwargs["coadd_id"] == str(db_coadd_result.coadd_id)


def test_coadd_map_list_caches(mock_mapcat_coadd):
    reader = CoaddRhoKappaMapReader()
    assert reader.map_list() is reader.map_list()


def test_coadd_iter_delegates_to_map_list(mock_mapcat_coadd):
    reader = CoaddRhoKappaMapReader()
    assert list(reader) == reader.map_list()
