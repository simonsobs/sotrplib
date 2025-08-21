"""
Tests for the fully simulated pipeline setup.
"""

import structlog
from astropy import units as u

from sotrplib.handlers.basic import PipelineRunner
from sotrplib.maps.core import SimulatedMap
from sotrplib.outputs.core import PickleSerializer
from sotrplib.sims.sources.core import (
    RandomSourceSimulation,
    RandomSourceSimulationParameters,
)
from sotrplib.sources.core import PhotutilsGaussianFitter
from sotrplib.sources.forced_photometry import photutils_2D_gauss_fit


def test_created_map(empty_map: SimulatedMap):
    assert isinstance(empty_map, SimulatedMap)


def test_injected_sources(empty_map: SimulatedMap):
    parameters = RandomSourceSimulationParameters(
        n_sources=64,
        # Use bright sources so we can guarantee recovery
        min_flux=u.Quantity(1.0, "Jy"),
        max_flux=u.Quantity(10.0, "Jy"),
        fwhm_uncertainty_frac=0.5,
    )

    simulator = RandomSourceSimulation(parameters=parameters)

    new_map, sources = simulator.simulate(input_map=empty_map)

    assert new_map.finalized
    assert len(sources) > 0

    assert (new_map.flux != empty_map.flux).any()
    assert (new_map.snr != empty_map.snr).any()

    # See if we can recover them
    fits, thumbnails = photutils_2D_gauss_fit(
        flux_map=new_map.flux,
        snr_map=new_map.snr,
        source_catalog=sources,
        log=structlog.get_logger(),
        return_thumbnails=True,
    )

    # Can't look at length of fits because the fits
    # are skipped for items near the edge.
    assert len(thumbnails) == len(sources)


def test_basic_pipeline(tmp_path, empty_map: SimulatedMap):
    """
    Tests a complete setup of the basic pipeline run.
    """

    parameters = RandomSourceSimulationParameters(
        n_sources=64,
        # Use bright sources so we can guarantee recovery
        min_flux=u.Quantity(1.0, "Jy"),
        max_flux=u.Quantity(10.0, "Jy"),
        fwhm_uncertainty_frac=0.5,
    )

    simulator = RandomSourceSimulation(parameters=parameters)

    new_map, sources = simulator.simulate(input_map=empty_map)

    runner = PipelineRunner(
        maps=[new_map, new_map],
        preprocessors=None,
        postprocessors=None,
        forced_photometry=PhotutilsGaussianFitter(sources=sources),
        source_subtractor=None,
        blind_search=None,
        sifter=None,
        outputs=[PickleSerializer(directory=tmp_path)],
    )

    runner.run()
