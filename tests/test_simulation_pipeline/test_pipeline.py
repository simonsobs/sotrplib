"""
Tests for the fully simulated pipeline setup.
"""

from astropy import units as u

from sotrplib.maps.core import SimulatedMap
from sotrplib.sims.sources.core import (
    RandomSourceSimulation,
    RandomSourceSimulationParameters,
)


def test_created_map(empty_map: SimulatedMap):
    assert isinstance(empty_map, SimulatedMap)


def test_injected_sources(empty_map: SimulatedMap):
    parameters = RandomSourceSimulationParameters(
        n_sources=64,
        min_flux=u.Quantity(0.05, "Jy"),
        max_flux=u.Quantity(10.0, "Jy"),
        fwhm_uncertainty_frac=0.5,
    )

    simulator = RandomSourceSimulation(parameters=parameters)

    new_map, sources = simulator.simulate(input_map=empty_map)

    assert len(sources) > 0

    assert (new_map.flux != empty_map.flux).any()
    assert (new_map.snr != empty_map.snr).any()
