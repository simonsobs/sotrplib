"""
Tests running the pipeline under prefect with simulated data.
"""

import json

from prefect.testing.utilities import prefect_test_harness

from sotrplib.handlers.prefect import PrefectRunner, analyze_from_configuration
from sotrplib.outputs.core import PickleSerializer
from sotrplib.sifter.core import DefaultSifter, SifterResult
from sotrplib.sims.maps import SimulatedMap
from sotrplib.sims.sources.core import ProcessableMapWithSimulatedSources
from sotrplib.sources.blind import SigmaClipBlindSearch
from sotrplib.sources.force import Scipy2DGaussianFitter
from sotrplib.sources.sources import MeasuredSource, RegisteredSource

sample_setup = {
    "maps": [
        {
            "map_type": "simulated",
            "observation_start": "2025-01-01",
            "observation_end": "2025-01-02",
        },
        {
            "map_type": "simulated",
            "observation_start": "2025-01-02",
            "observation_end": "2025-01-03",
        },
    ],
    "source_simulators": [
        {
            "simulation_type": "random",
            "parameters": {
                "n_sources": 32,
                "min_flux": "3.0 Jy",
                "max_flux": "10.0 Jy",
                "fwhm_uncertainty_frac": 0.0,
                "fraction_return": 0.5,
            },
        }
    ],
    "preprocessors": [{"preprocessor_type": "kappa_rho"}],
    "source_subtractor": {"subtractor_type": "photutils"},
    "blind_search": {"search_type": "photutils"},
    "forced_photometry": {"photometry_type": "scipy", "reproject_thumbnails": "True"},
    "sifter": {"sifter_type": "default"},
    "outputs": [
        {
            "output_type": "pickle",
        }
    ],
}


def test_basic_pipeline_scipy(
    tmp_path, map_with_sources: tuple[SimulatedMap, list[RegisteredSource]]
):
    """
    Tests a complete setup of the basic pipeline run.
    """

    new_map, sources = map_with_sources
    maps = [new_map, new_map]

    runner = PrefectRunner(
        maps=maps,
        forced_photometry_catalog=None,
        source_catalogs=[],
        preprocessors=None,
        postprocessors=None,
        source_simulators=None,
        forced_photometry=Scipy2DGaussianFitter(sources=sources),
        source_subtractor=None,
        blind_search=SigmaClipBlindSearch(),
        sifter=DefaultSifter(catalog_sources=sources),
        outputs=[PickleSerializer(directory=tmp_path)],
    )

    with prefect_test_harness():
        result = runner.run()
        _validate_pipeline_result(result, len(maps))


def test_basic_pipeline_from_file(
    tmp_path, map_with_sources: tuple[SimulatedMap, list[RegisteredSource]]
):
    """
    Tests a complete setup of the basic pipeline run.
    """

    config = sample_setup.copy()
    config["outputs"][0]["directory"] = str(tmp_path)
    config_file = tmp_path / "config.json"
    with open(config_file, "w") as ff:
        json.dump(config, ff)

    with prefect_test_harness():
        result = analyze_from_configuration(config_file)
        _validate_pipeline_result(result, len(config["maps"]))


def _validate_pipeline_result(result, nmaps):
    assert len(result) == nmaps
    candidates, sifter_result, output_map = result[0]
    assert all([isinstance(candidate, MeasuredSource) for candidate in candidates])
    assert isinstance(sifter_result, SifterResult)
    assert isinstance(output_map, ProcessableMapWithSimulatedSources)
