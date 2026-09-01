"""
Tests the map I/O
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from astropy import units as u

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
    assert maps[0].map_id == db_result.map_id


def test_map_list_appends_map_id(mock_mapcat, db_result):
    reader = IntensityMapReader()
    reader.map_list()
    assert db_result.map_id in reader.map_ids


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


def test_map_list_caches(mock_mapcat):
    reader = IntensityMapReader()
    assert reader.map_list() is reader.map_list()


def test_map_list_number_to_read_limits(mock_session, db_result):
    """number_to_read=1 stops after the first map even when the DB has more rows."""
    db_result2 = MagicMock()
    db_result2.map_id = 99
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
