"""
Tests running the pipeline under prefect with simulated data.
"""

from prefect.testing.utilities import prefect_test_harness

from sotrplib.handlers.prefect import PrefectRunner
from sotrplib.outputs.core import PickleSerializer
from sotrplib.sifter.core import DefaultSifter, SifterResult
from sotrplib.sims.maps import SimulatedMap
from sotrplib.sims.sources.core import ProcessableMapWithSimulatedSources
from sotrplib.source_catalog.core import RegisteredSourceCatalog
from sotrplib.sources.blind import SigmaClipBlindSearch
from sotrplib.sources.force import Scipy2DGaussianFitter
from sotrplib.sources.sources import MeasuredSource, RegisteredSource


def test_basic_pipeline_scipy(
    tmp_path, map_with_sources: tuple[SimulatedMap, list[RegisteredSource]]
):
    """
    Tests a complete setup of the basic pipeline run.
    """

    new_map, sources = map_with_sources
    maps = [new_map, new_map]
    source_cat = RegisteredSourceCatalog(sources=sources)
    runner = PrefectRunner(
        maps=maps,
        map_coadder=None,
        source_catalogs=[],
        source_injector=None,
        preprocessors=None,
        pointing_provider=None,
        pointing_residual=None,
        postprocessors=None,
        source_simulators=None,
        forced_photometry=Scipy2DGaussianFitter(),
        source_subtractor=None,
        blind_search=SigmaClipBlindSearch(),
        sifter=DefaultSifter(),
        outputs=[PickleSerializer(directory=tmp_path)],
    )

    with prefect_test_harness():
        result = runner.run()
        _validate_pipeline_result(result, len(maps))


def _validate_pipeline_result(result, nmaps):
    assert len(result) == nmaps
    print(result[0])
    candidates, sifter_result, output_map = result[0]
    assert all([isinstance(candidate, MeasuredSource) for candidate in candidates])
    assert isinstance(sifter_result, SifterResult)
    assert isinstance(output_map, ProcessableMapWithSimulatedSources)
