"""
Tests for the pointing model that the mapcat reader loads for a map: the
matched filter keeps it, the runner uses it, and the runner does not replace
a model in mapcat with an unmeasured zero offset.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from astropy import units as u
from mapcat.pointing.const import ConstantPointingModel

from sotrplib.config.maps import InverseVarianceMapConfig
from sotrplib.config.preprocessors import MatchedFilterConfig
from sotrplib.handlers.basic import PipelineRunner
from sotrplib.maps.pointing import ConstantPointingOffset, EmptyPointingOffset


def _ivar_map(paths):
    input_map = InverseVarianceMapConfig(
        intensity_map_path=paths["map"],
        weights_map_path=paths["ivar"],
        time_map_path=paths["time"],
        observation_start="2025-01-01T00:00:00Z",
        observation_end="2025-01-01T12:00:00Z",
    ).to_map()
    # As a map from the mapcat reader, so that the runner saves its model.
    input_map._parent_database = Path("mapcat.sqlite")
    return input_map


def _patch_mapcat():
    """Patch the mapcat writes of the runner. Returns the save_pointing_model mock."""
    patch("sotrplib.handlers.base.set_processing_end").start()
    return patch("sotrplib.handlers.base.save_pointing_model").start()


def _runner(pointing_residual_model, preprocessors=None):
    return PipelineRunner(
        map_coadder=None,
        source_catalogs=[],
        source_injector=None,
        preprocessors=preprocessors,
        pointing_provider=None,
        pointing_residual_model=pointing_residual_model,
        postprocessors=None,
        source_simulators=None,
        forced_photometry=None,
        source_subtractor=None,
        blind_search=None,
        sifter=None,
        source_outputs=None,
        map_outputs=None,
    )


@pytest.fixture(autouse=True)
def _stop_patches():
    yield
    patch.stopall()


def test_matched_filter_keeps_pointing_model(separate_map_set_1):
    input_map = _ivar_map(separate_map_set_1)
    input_map.build()
    model = ConstantPointingModel(ra_offset=10 * u.arcsec, dec_offset=-5 * u.arcsec)
    input_map.pointing_model = model

    filtered = MatchedFilterConfig().to_preprocessor().preprocess(input_map=input_map)

    assert filtered is not input_map
    assert filtered.pointing_model is model


def test_runner_does_not_save_unmeasured_pointing_model(separate_map_set_1):
    save = _patch_mapcat()
    _runner(EmptyPointingOffset()).run([_ivar_map(separate_map_set_1)])
    save.assert_not_called()


def test_runner_saves_measured_pointing_model(separate_map_set_1):
    save = _patch_mapcat()
    _runner(ConstantPointingOffset()).run([_ivar_map(separate_map_set_1)])
    save.assert_called_once()


def test_runner_uses_loaded_pointing_model(separate_map_set_1):
    """
    A model that the reader loaded goes through the matched filter. The
    runner uses it, and does not make or save a new model.
    """
    input_map = _ivar_map(separate_map_set_1)
    input_map.pointing_model = ConstantPointingModel(
        ra_offset=10 * u.arcsec, dec_offset=-5 * u.arcsec
    )
    residual_model = ConstantPointingOffset()
    residual_model.build_model = MagicMock(wraps=residual_model.build_model)

    save = _patch_mapcat()
    _runner(
        residual_model, preprocessors=[MatchedFilterConfig().to_preprocessor()]
    ).run([input_map])

    residual_model.build_model.assert_not_called()
    save.assert_not_called()
