"""
Tests the map I/O
"""

from sotrplib.config.maps import RhoKappaMapConfig
from sotrplib.handlers.basic import PipelineRunner
from sotrplib.map_io import load_maps


def test_get_depth1_mapset():
    map_list = load_maps.get_depth1_mapset(map_path="example_map/map.fits")

    assert len(map_list) == 5


def test_load_full_depth1_mapset(ones_map_set):
    paths = ones_map_set

    ret = load_maps.load_depth1_mapset(
        map_path=paths["map"],
        ivar_path=paths["ivar"],
        rho_path=paths["rho"],
        kappa_path=paths["kappa"],
        time_path=paths["time"],
    )

    # all maps are present
    assert ret is not None
    assert ret.intensity_map is not None
    assert ret.inverse_variance_map is not None
    assert ret.rho_map is not None
    assert ret.kappa_map is not None
    assert ret.time_map is not None


def test_load_unfiltered_depth1_mapset(ones_unfiltered_partial_map_set):
    paths = ones_unfiltered_partial_map_set

    ret = load_maps.load_depth1_mapset(
        map_path=paths["map"], ivar_path=paths["ivar"], time_path=paths["time"]
    )

    # all maps are present
    assert ret is not None
    assert ret.intensity_map is not None
    assert ret.inverse_variance_map is not None
    assert ret.rho_map is None
    assert ret.kappa_map is None
    assert ret.time_map is not None


def test_load_filtered_depth1_mapset(ones_filtered_partial_map_set):
    paths = ones_filtered_partial_map_set

    ret = load_maps.load_depth1_mapset(
        map_path=paths["rho"],
        rho_path=paths["rho"],
        kappa_path=paths["kappa"],
        time_path=paths["time"],
    )

    # all maps are present
    assert ret is not None
    assert ret.intensity_map is None
    assert ret.inverse_variance_map is None
    assert ret.rho_map is not None
    assert ret.kappa_map is not None
    assert ret.time_map is not None


def test_basic_pipeline_rhokappa(tmp_path, ones_map_set):
    """
    Tests a complete setup of the basic pipeline run with input rho,kappa maps
    """
    paths = ones_map_set
    input_map = RhoKappaMapConfig(
        rho_map_path=paths["rho"],
        kappa_map_path=paths["kappa"],
        time_map_path=paths["time"],
    ).to_map()
    runner = PipelineRunner(
        maps=[input_map],
        preprocessors=None,
        postprocessors=None,
        source_simulators=None,
        forced_photometry=None,
        source_subtractor=None,
        blind_search=None,
        sifter=None,
        outputs=[],
    )

    runner.run()
