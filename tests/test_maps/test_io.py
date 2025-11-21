"""
Tests the map I/O
"""

from datetime import datetime, timedelta, timezone

from sotrplib.config.maps import InverseVarianceMapConfig, RhoKappaMapConfig
from sotrplib.handlers.basic import PipelineRunner


def test_basic_pipeline_rhokappa(tmp_path, ones_map_set):
    """
    Tests a complete setup of the basic pipeline run with input rho,kappa maps
    """
    paths = ones_map_set
    input_map = RhoKappaMapConfig(
        rho_map_path=paths["rho"],
        kappa_map_path=paths["kappa"],
        time_map_path=paths["time"],
        observation_start=datetime.now(tz=timezone.utc),
        observation_end=datetime.now(tz=timezone.utc) + timedelta(hours=1),
    ).to_map()
    runner = PipelineRunner(
        maps=[input_map],
        source_catalogs=[],
        sso_catalogs=[],
        source_injector=None,
        preprocessors=None,
        pointing_provider=None,
        pointing_residual=None,
        postprocessors=None,
        source_simulators=None,
        forced_photometry=None,
        source_subtractor=None,
        blind_search=None,
        sifter=None,
        outputs=[],
    )

    runner.run()


def test_basic_pipeline_ivar(tmp_path, ones_map_set):
    """
    Tests a complete setup of the basic pipeline run with input intensity, inverse variance maps
    """
    paths = ones_map_set
    input_map = InverseVarianceMapConfig(
        intensity_map_path=paths["map"],
        weights_map_path=paths["ivar"],
        time_map_path=paths["time"],
        observation_start=datetime.now(tz=timezone.utc),
        observation_end=datetime.now(tz=timezone.utc) + timedelta(hours=1),
    ).to_map()
    runner = PipelineRunner(
        maps=[input_map],
        source_catalogs=[],
        sso_catalogs=[],
        source_injector=None,
        preprocessors=None,
        pointing_provider=None,
        pointing_residual=None,
        postprocessors=None,
        source_simulators=None,
        forced_photometry=None,
        source_subtractor=None,
        blind_search=None,
        sifter=None,
        outputs=[],
    )

    runner.run()
