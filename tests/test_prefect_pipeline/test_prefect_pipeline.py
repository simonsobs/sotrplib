"""
Tests running the pipeline under prefect with simulated data.
"""

from astropy import units as u
from astropy.time import Time, TimeDelta
from prefect.testing.utilities import prefect_test_harness

from sotrplib.handlers.prefect import PrefectRunner
from sotrplib.outputs.core import PickleSerializer
from sotrplib.sifter.core import DefaultSifter, SifterResult
from sotrplib.sifter.map_matching import EmptyMapMatcher
from sotrplib.sims.maps import SimulatedMap
from sotrplib.sims.sim_source_generators import FixedSourceGenerator
from sotrplib.sims.source_injector import PhotutilsSourceInjector
from sotrplib.sources.blind import SigmaClipBlindSearch
from sotrplib.sources.force import TwoDGaussianFitter
from sotrplib.sources.sources import MeasuredSource


def _make_runner_and_maps(tmp_path, fit_mode: str, map_matcher: EmptyMapMatcher):
    empty_map = SimulatedMap(
        observation_start=Time.now() - TimeDelta(1, format="jd"),
        observation_end=Time.now(),
    )

    maps = [empty_map, empty_map]
    source_simulator = FixedSourceGenerator(
        number=200,
        min_flux=1 * u.Jy,
        max_flux=10 * u.Jy,
        catalog_fraction=1.0,
    )
    runner = PrefectRunner(
        map_coadder=None,
        source_catalogs=[],
        source_injector=PhotutilsSourceInjector(),
        preprocessors=None,
        pointing_provider=None,
        pointing_residual_model=None,
        postprocessors=None,
        source_simulators=[source_simulator],
        forced_photometry=TwoDGaussianFitter(mode=fit_mode),
        source_subtractor=None,
        blind_search=SigmaClipBlindSearch(),
        sifter=DefaultSifter(),
        source_outputs=[PickleSerializer(directory=tmp_path)],
        map_outputs=None,
        map_matcher=map_matcher,
    )
    return runner, maps


def test_basic_pipeline(tmp_path, fit_mode: str):
    """
    Tests a complete setup of the basic pipeline run.
    """
    runner, maps = _make_runner_and_maps(
        tmp_path, fit_mode, EmptyMapMatcher(hold_all_maps=True)
    )

    with prefect_test_harness():
        result = runner.run(maps)
        _validate_pipeline_result(result, len(maps))


def test_pipeline_without_holding(tmp_path, fit_mode: str):
    """
    The default map matcher does not hold the results: the workers write the
    source outputs and the runner returns an empty list.
    """
    runner, maps = _make_runner_and_maps(tmp_path, fit_mode, EmptyMapMatcher())

    with prefect_test_harness():
        result = runner.run(maps)

    assert result == []
    assert list(tmp_path.glob("*.pickle"))


def _validate_pipeline_result(result, nmaps):
    assert len(result) == nmaps
    candidates, sifter_result = result[0]
    assert all([isinstance(candidate, MeasuredSource) for candidate in candidates])
    assert isinstance(sifter_result, SifterResult)
