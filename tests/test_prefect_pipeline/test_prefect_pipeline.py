"""
Tests running the pipeline under prefect with simulated data.
"""
import datetime

from astropy import units as u
from prefect.testing.utilities import prefect_test_harness

from sotrplib.handlers.prefect import PrefectRunner
from sotrplib.outputs.core import PickleSerializer
from sotrplib.sifter.core import DefaultSifter, SifterResult
from sotrplib.sims.maps import SimulatedMap
from sotrplib.sims.sim_source_generators import FixedSourceGenerator
from sotrplib.sims.source_injector import PhotutilsSourceInjector
from sotrplib.sources.blind import SigmaClipBlindSearch
from sotrplib.sources.force import TwoDGaussianFitter
from sotrplib.sources.sources import MeasuredSource


def test_basic_pipeline(tmp_path, fit_mode: str):
    """
    Tests a complete setup of the basic pipeline run.
    """
    empty_map = SimulatedMap(
        observation_start=datetime.datetime.now(tz=datetime.timezone.utc)
        - datetime.timedelta(days=1),
        observation_end=datetime.datetime.now(tz=datetime.timezone.utc),
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
        sso_catalogs=[],
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
    )

    with prefect_test_harness():
        result = runner.run(maps)
        _validate_pipeline_result(result, len(maps))


def _validate_pipeline_result(result, nmaps):
    assert len(result) == nmaps
    candidates, sifter_result = result[0]
    assert all([isinstance(candidate, MeasuredSource) for candidate in candidates])
    assert isinstance(sifter_result, SifterResult)
