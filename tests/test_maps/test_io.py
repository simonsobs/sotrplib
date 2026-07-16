"""
Tests the map I/O
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from astropy import units as u
from astropy.time import Time

from sotrplib.config.maps import InverseVarianceMapConfig, RhoKappaMapConfig
from sotrplib.handlers.basic import PipelineRunner
from sotrplib.maps.core import (
    FluxAndSNRMap,
    IntensityAndInverseVarianceMap,
    RhoAndKappaMap,
)
from sotrplib.maps.database import (
    FluxMapReader,
    IntensityMapReader,
    MapCatDatabaseReader,
    RhoKappaMapReader,
)
from sotrplib.utils.utils import get_fwhm


def test_basic_pipeline_rhokappa(separate_map_set_1):
    """
    Tests a complete setup of the basic pipeline run with input rho,kappa maps
    """
    paths = separate_map_set_1
    input_map = RhoKappaMapConfig(
        rho_map_path=paths["rho"],
        kappa_map_path=paths["kappa"],
        time_map_path=paths["time"],
        observation_start="2025-01-01T00:00:00Z",
        observation_end="2025-01-01T12:00:00Z",
    ).to_map()
    runner = PipelineRunner(
        map_coadder=None,
        source_catalogs=[],
        source_injector=None,
        preprocessors=None,
        pointing_provider=None,
        pointing_residual_model=None,
        postprocessors=None,
        source_simulators=None,
        forced_photometry=None,
        source_subtractor=None,
        blind_search=None,
        sifter=None,
        source_outputs=None,
        map_outputs=None,
    )

    runner.run([input_map])


def test_basic_pipeline_ivar(separate_map_set_1):
    """
    Tests a complete setup of the basic pipeline run with input intensity, inverse variance maps
    """
    paths = separate_map_set_1
    input_map = InverseVarianceMapConfig(
        intensity_map_path=paths["map"],
        weights_map_path=paths["ivar"],
        time_map_path=paths["time"],
        observation_start="2025-01-01T00:00:00Z",
        observation_end="2025-01-01T12:00:00Z",
    ).to_map()
    runner = PipelineRunner(
        map_coadder=None,
        source_catalogs=[],
        source_injector=None,
        preprocessors=None,
        pointing_provider=None,
        pointing_residual_model=None,
        postprocessors=None,
        source_simulators=None,
        forced_photometry=None,
        source_subtractor=None,
        blind_search=None,
        sifter=None,
        source_outputs=None,
        map_outputs=None,
    )

    runner.run([input_map])


def test_basic_pipeline_instrument(separate_map_set_1):
    """
    Tests a complete setup of the basic pipeline run with input rho,kappa maps
    """
    paths = separate_map_set_1
    input_map = RhoKappaMapConfig(
        rho_map_path=paths["rho"],
        kappa_map_path=paths["kappa"],
        time_map_path=paths["time"],
        observation_start="2025-01-01T00:00:00Z",
        observation_end="2025-01-01T12:00:00Z",
        instrument="SOSAT",
        array=None,
        frequency="f090",
    ).to_map()

    assert get_fwhm(
        arr=input_map.array, freq=input_map.frequency, instrument=input_map.instrument
    ) == get_fwhm(freq="f090", instrument="SOSAT")


# ─── database reader fixtures ─────────────────────────────────────────────────


@pytest.fixture
def db_result(separate_map_set_1):
    """Mock DepthOneMapTable row pointing at the tmp FITS files from conftest."""
    paths = separate_map_set_1
    r = MagicMock()
    r.map_id = "11111111-1111-1111-1111-111111111111"
    r.map_path = paths["map"]
    r.ivar_path = paths["ivar"]
    r.rho_path = paths["rho"]
    r.kappa_path = paths["kappa"]
    # reuse rho/kappa as stand-ins for flux/snr (same shape, valid FITS)
    r.flux_path = paths["rho"]
    r.snr_path = paths["kappa"]
    r.mean_time_path = paths["time"]
    r.start_time = Time.now().unix - 3600
    r.stop_time = Time.now().unix
    r.frequency = "f090"
    r.tube_slot = "pa5"
    return r


@pytest.fixture
def mock_session(db_result):
    """Mock DB session returning one db_result row."""
    session = MagicMock()
    session.execute.return_value.scalars.return_value.all.return_value = [db_result]
    return session


@pytest.fixture
def mock_mapcat(mock_session):
    """
    Patches mapcat_settings, check_if_processed, and set_processing_start so
    that map_list() runs without a real database connection.
    """
    with (
        patch("sotrplib.maps.database.mapcat_settings") as settings,
        patch("sotrplib.maps.database.check_if_processed", return_value=False),
        patch("sotrplib.maps.database.set_processing_start"),
    ):
        settings.database_name = "test_db"
        settings.depth_one_parent = Path("/")
        # session() is used as a context manager; wire __enter__ to our mock
        settings.session.return_value.__enter__.return_value = mock_session
        yield settings


# ─── abstract base ────────────────────────────────────────────────────────────


def test_cannot_instantiate_base():
    with pytest.raises(TypeError):
        MapCatDatabaseReader()


# ─── default units ────────────────────────────────────────────────────────────


def test_intensity_reader_default_units():
    assert IntensityMapReader().map_units.is_equivalent(u.K)


def test_rhokappa_reader_default_units():
    assert RhoKappaMapReader().map_units.is_equivalent(u.Jy)


def test_flux_reader_default_units():
    assert FluxMapReader().map_units.is_equivalent(u.Jy)


# ─── unit validation ──────────────────────────────────────────────────────────


def test_intensity_reader_rejects_jy():
    with pytest.raises(ValueError, match="IntensityMapReader"):
        IntensityMapReader(map_units=u.Jy)


def test_rhokappa_reader_rejects_k():
    with pytest.raises(ValueError, match="RhoKappaMapReader"):
        RhoKappaMapReader(map_units=u.K)


def test_flux_reader_rejects_k():
    with pytest.raises(ValueError, match="FluxMapReader"):
        FluxMapReader(map_units=u.K)


# ─── _build_map ───────────────────────────────────────────────────────────────


def test_intensity_build_map_type(db_result):
    with patch("sotrplib.maps.database.mapcat_settings") as settings:
        settings.depth_one_parent = Path("/")
        result = IntensityMapReader()._build_map(db_result)
    assert isinstance(result, IntensityAndInverseVarianceMap)


def test_rhokappa_build_map_type(db_result):
    with patch("sotrplib.maps.database.mapcat_settings") as settings:
        settings.depth_one_parent = Path("/")
        result = RhoKappaMapReader()._build_map(db_result)
    assert isinstance(result, RhoAndKappaMap)


def test_flux_build_map_type(db_result):
    with patch("sotrplib.maps.database.mapcat_settings") as settings:
        settings.depth_one_parent = Path("/")
        result = FluxMapReader()._build_map(db_result)
    assert isinstance(result, FluxAndSNRMap)


def test_build_map_passes_metadata(db_result):
    with patch("sotrplib.maps.database.mapcat_settings") as settings:
        settings.depth_one_parent = Path("/")
        result = IntensityMapReader(map_units=u.mK)._build_map(db_result)
    assert result.frequency == db_result.frequency
    assert result.array == db_result.tube_slot


# ─── build_query time windowing ────────────────────────────────────────────────


def test_build_query_default_is_interval_overlap():
    """
    Default (bucket_by_start_time=False): a map is included if its
    [start_time, stop_time] interval overlaps [start_time, end_time] at all,
    inclusive on both ends.
    """
    start = Time(1000, format="unix")
    end = Time(2000, format="unix")
    query = IntensityMapReader(start_time=start, end_time=end).build_query()
    compiled = str(query.compile(compile_kwargs={"literal_binds": True}))
    assert "depth_one_maps.stop_time >=" in compiled
    assert "depth_one_maps.start_time <=" in compiled
    assert "depth_one_maps.start_time >=" not in compiled


def test_build_query_bucket_by_start_time_is_half_open():
    """
    bucket_by_start_time=True: a map is included solely based on whether its
    own start_time falls in [start_time, end_time) -- so a map whose
    observation spans a boundary between two adjacent windows lands in
    exactly one of them, never both. This is what submit_week_coadds.py
    relies on to avoid double-coadding maps that straddle a week boundary.
    """
    start = Time(1000, format="unix")
    end = Time(2000, format="unix")
    query = IntensityMapReader(
        start_time=start, end_time=end, bucket_by_start_time=True
    ).build_query()
    compiled = str(query.compile(compile_kwargs={"literal_binds": True}))
    where_clause = compiled.split("WHERE", 1)[1]
    assert "depth_one_maps.start_time >=" in where_clause
    assert "depth_one_maps.start_time <" in where_clause
    assert "depth_one_maps.start_time <=" not in where_clause
    assert "stop_time" not in where_clause


def test_build_query_bucket_by_start_time_no_gap_or_overlap_at_shared_boundary():
    """
    Two adjacent windows sharing an exact boundary value must partition maps
    with no gap and no overlap, even for a map whose observation interval
    spans the boundary (the case that caused real double-coadding).
    """
    boundary = 1500.0
    straddling_map = SimpleNamespace(start_time=1400.0, stop_time=1600.0)

    window_a = IntensityMapReader(
        start_time=Time(1000, format="unix"),
        end_time=Time(boundary, format="unix"),
        bucket_by_start_time=True,
    )
    window_b = IntensityMapReader(
        start_time=Time(boundary, format="unix"),
        end_time=Time(2000, format="unix"),
        bucket_by_start_time=True,
    )

    def _matches(reader, m):
        return reader.start_time.unix <= m.start_time < reader.end_time.unix

    assert _matches(window_a, straddling_map) != _matches(window_b, straddling_map)


# ─── map_list ─────────────────────────────────────────────────────────────────


def test_map_list_intensity_type(mock_mapcat):
    maps = IntensityMapReader().map_list()
    assert len(maps) == 1
    assert isinstance(maps[0], IntensityAndInverseVarianceMap)


def test_map_list_rhokappa_type(mock_mapcat):
    maps = RhoKappaMapReader().map_list()
    assert len(maps) == 1
    assert isinstance(maps[0], RhoAndKappaMap)


def test_map_list_flux_type(mock_mapcat):
    maps = FluxMapReader().map_list()
    assert len(maps) == 1
    assert isinstance(maps[0], FluxAndSNRMap)


def test_map_list_sets_map_id(mock_mapcat, db_result):
    maps = IntensityMapReader().map_list()
    assert maps[0].map_id == str(db_result.map_id)


def test_map_list_appends_map_id(mock_mapcat, db_result):
    reader = IntensityMapReader()
    reader.map_list()
    assert str(db_result.map_id) in reader.map_ids


def test_map_list_skips_processed(mock_session):
    with (
        patch("sotrplib.maps.database.mapcat_settings") as settings,
        patch("sotrplib.maps.database.check_if_processed", return_value=True),
        patch("sotrplib.maps.database.set_processing_start"),
    ):
        settings.database_name = "test_db"
        settings.depth_one_parent = Path("/")
        settings.session.return_value.__enter__.return_value = mock_session
        maps = IntensityMapReader().map_list()
    assert maps == []


def test_map_list_rerun_ignores_processed(mock_session):
    with (
        patch("sotrplib.maps.database.mapcat_settings") as settings,
        patch(
            "sotrplib.maps.database.check_if_processed", return_value=True
        ) as mock_check,
        patch("sotrplib.maps.database.set_processing_start"),
    ):
        settings.database_name = "test_db"
        settings.depth_one_parent = Path("/")
        settings.session.return_value.__enter__.return_value = mock_session
        maps = IntensityMapReader(rerun=True).map_list()
    mock_check.assert_not_called()
    assert len(maps) == 1


def test_map_list_skips_permafailed(mock_session):
    """
    permafail is set manually (e.g. mapcatreset --status permafail) for a
    known-pathological observation, and must never be picked up again --
    not even with rerun=True.
    """
    with (
        patch("sotrplib.maps.database.mapcat_settings") as settings,
        patch("sotrplib.maps.database.check_if_permafailed", return_value=True),
        patch("sotrplib.maps.database.check_if_processed", return_value=False),
        patch("sotrplib.maps.database.set_processing_start"),
    ):
        settings.database_name = "test_db"
        settings.depth_one_parent = Path("/")
        settings.session.return_value.__enter__.return_value = mock_session
        maps = IntensityMapReader(rerun=True).map_list()
    assert maps == []


def test_map_list_caches(mock_mapcat):
    reader = IntensityMapReader()
    assert reader.map_list() is reader.map_list()


def test_map_list_number_to_read_limits(mock_session, db_result):
    """number_to_read=1 stops after the first map even when the DB has more rows."""
    db_result2 = MagicMock()
    db_result2.map_id = "22222222-2222-2222-2222-222222222222"
    db_result2.map_path = db_result.map_path
    db_result2.ivar_path = db_result.ivar_path
    db_result2.mean_time_path = db_result.mean_time_path
    db_result2.start_time = db_result.start_time
    db_result2.stop_time = db_result.stop_time
    db_result2.frequency = db_result.frequency
    db_result2.tube_slot = db_result.tube_slot
    mock_session.execute.return_value.scalars.return_value.all.return_value = [
        db_result,
        db_result2,
    ]
    with (
        patch("sotrplib.maps.database.mapcat_settings") as settings,
        patch("sotrplib.maps.database.check_if_processed", return_value=False),
        patch("sotrplib.maps.database.set_processing_start"),
    ):
        settings.database_name = "test_db"
        settings.depth_one_parent = Path("/")
        settings.session.return_value.__enter__.return_value = mock_session
        maps = IntensityMapReader(number_to_read=1).map_list()
    assert len(maps) == 1


def test_iter_delegates_to_map_list(mock_mapcat):
    reader = IntensityMapReader()
    assert list(reader) == reader.map_list()


# ─── processing status lifecycle ───────────────────────────────────────────────


def _session_with_status(status: str | None, processing_start: float = 0.0):
    """A mock session whose TimeDomainProcessingTable row has the given status."""
    session = MagicMock()
    if status is None:
        session.execute.return_value.one_or_none.return_value = []
    else:
        row = MagicMock()
        row.processing_status = status
        row.processing_start = processing_start
        session.execute.return_value.one_or_none.return_value = [row]
    return session


def test_check_if_permafailed_true():
    from sotrplib.maps.database import check_if_permafailed

    assert (
        check_if_permafailed(
            "11111111-1111-1111-1111-111111111111",
            session=_session_with_status("permafail"),
        )
        is True
    )


def test_check_if_permafailed_false_for_other_statuses():
    from sotrplib.maps.database import check_if_permafailed

    assert (
        check_if_permafailed(
            "11111111-1111-1111-1111-111111111111",
            session=_session_with_status("completed"),
        )
        is False
    )
    assert (
        check_if_permafailed(
            "11111111-1111-1111-1111-111111111111",
            session=_session_with_status("processing"),
        )
        is False
    )
    assert (
        check_if_permafailed(
            "11111111-1111-1111-1111-111111111111", session=_session_with_status(None)
        )
        is False
    )


def test_check_if_processed_true_for_completed():
    from sotrplib.maps.database import check_if_processed

    assert (
        check_if_processed(
            "11111111-1111-1111-1111-111111111111",
            session=_session_with_status("completed"),
        )
        is True
    )


def test_check_if_processed_false_for_permafail():
    """
    permafail is deliberately not treated as "processed" by check_if_processed
    -- it's handled by the separate, unconditional check_if_permafailed check
    in map_list() instead, so it isn't bypassable by rerun=True.
    """
    from sotrplib.maps.database import check_if_processed

    assert (
        check_if_processed(
            "11111111-1111-1111-1111-111111111111",
            session=_session_with_status("permafail"),
        )
        is False
    )


def test_check_if_processed_true_for_recent_processing():
    from sotrplib.maps.database import check_if_processed

    session = _session_with_status("processing", processing_start=Time.now().unix)
    assert (
        check_if_processed("11111111-1111-1111-1111-111111111111", session=session)
        is True
    )


def test_check_if_processed_false_for_stale_processing():
    from astropy.time import TimeDelta

    from sotrplib.maps.database import check_if_processed

    stale_start = Time.now().unix - 3 * 3600
    session = _session_with_status("processing", processing_start=stale_start)
    assert (
        check_if_processed(
            "11111111-1111-1111-1111-111111111111",
            session=session,
            stale_limit=TimeDelta(2 * 3600, format="sec"),
        )
        is False
    )


def test_set_processing_end_default_status_is_completed():
    from sotrplib.maps.database import set_processing_end

    row = MagicMock()
    session = MagicMock()
    session.execute.return_value.one_or_none.return_value = [row]

    set_processing_end("11111111-1111-1111-1111-111111111111", session=session)
    assert row.processing_status == "completed"
    assert session.commit.called


def test_set_processing_end_accepts_failed_status():
    from sotrplib.maps.database import set_processing_end

    row = MagicMock()
    session = MagicMock()
    session.execute.return_value.one_or_none.return_value = [row]

    set_processing_end(
        "11111111-1111-1111-1111-111111111111", session=session, status="failed"
    )
    assert row.processing_status == "failed"


# ─── map_id/coadd_id dual-target generalization ────────────────────────────────


@pytest.mark.parametrize(
    "func_name",
    [
        "check_if_permafailed",
        "check_if_processed",
        "set_processing_start",
        "set_processing_end",
    ],
)
def test_status_functions_reject_neither_target(func_name):
    import sotrplib.maps.database as db_module

    func = getattr(db_module, func_name)
    with pytest.raises(ValueError, match="Exactly one of map_id or coadd_id"):
        func(session=MagicMock())


@pytest.mark.parametrize(
    "func_name",
    [
        "check_if_permafailed",
        "check_if_processed",
        "set_processing_start",
        "set_processing_end",
    ],
)
def test_status_functions_reject_both_targets(func_name):
    import sotrplib.maps.database as db_module

    func = getattr(db_module, func_name)
    with pytest.raises(ValueError, match="Exactly one of map_id or coadd_id"):
        func(1, coadd_id=2, session=MagicMock())


def test_check_if_permafailed_works_with_coadd_id():
    from sotrplib.maps.database import check_if_permafailed

    session = _session_with_status("permafail")
    assert (
        check_if_permafailed(
            coadd_id="33333333-3333-3333-3333-333333333333", session=session
        )
        is True
    )


def test_set_processing_start_creates_row_for_coadd_id():
    """A coadd-linked TimeDomainProcessingTable row should be created with
    coadd_id set and map_id left None, not the other way around."""
    from sotrplib.maps.database import set_processing_start

    session = _session_with_status(None)
    set_processing_start(
        coadd_id="44444444-4444-4444-4444-444444444444", session=session
    )
    created_row = session.add.call_args.args[0]
    assert str(created_row.coadd_id) == "44444444-4444-4444-4444-444444444444"
    assert created_row.map_id is None


def test_set_processing_start_does_not_couple_processing_status_id_to_map_id():
    """
    Regression test: set_processing_start() used to do
    TimeDomainProcessingTable(processing_status_id=map_id, map_id=map_id),
    directly reusing map_id's value as a *different* autoincrement PK
    column. That's a hard type error once map_id is a UUID (can't assign a
    UUID string into an int PK) -- the fix lets processing_status_id
    autoincrement instead of being set explicitly.
    """
    from sotrplib.maps.database import set_processing_start

    session = _session_with_status(None)
    set_processing_start("019f6c0c-7a24-7ad7-85a3-68acbc551549", session=session)
    created_row = session.add.call_args.args[0]
    assert str(created_row.map_id) == "019f6c0c-7a24-7ad7-85a3-68acbc551549"
    # processing_status_id must not have been forced to equal map_id
    assert getattr(created_row, "processing_status_id", None) != (
        "019f6c0c-7a24-7ad7-85a3-68acbc551549"
    )


def test_set_processing_end_works_with_coadd_id():
    from sotrplib.maps.database import set_processing_end

    row = MagicMock()
    session = MagicMock()
    session.execute.return_value.one_or_none.return_value = [row]

    set_processing_end(
        coadd_id="44444444-4444-4444-4444-444444444444",
        session=session,
        status="completed",
    )
    assert row.processing_status == "completed"
    assert session.commit.called


def test_set_processing_end_raises_with_target_specific_message_for_coadd():
    from sotrplib.maps.database import set_processing_end

    session = _session_with_status(None)
    with pytest.raises(ValueError, match="coadd_id 44444444"):
        set_processing_end(
            coadd_id="44444444-4444-4444-4444-444444444444", session=session
        )
