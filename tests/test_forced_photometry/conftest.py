"""
Fixtures for the dependency-injected completely simulated pipeline.
"""

import datetime

import pytest
from astropy import units as u

from sotrplib.sims.maps import SimulatedMap
from sotrplib.sims.sources.core import (
    RandomSourceSimulation,
    RandomSourceSimulationParameters,
)


@pytest.fixture
def empty_map():
    map = SimulatedMap(
        observation_start=datetime.datetime.now(tz=datetime.UTC)
        - datetime.timedelta(days=1),
        observation_end=datetime.datetime.now(tz=datetime.UTC),
    )

    map.build()
    map.finalize()

    assert map.finalized

    yield map


@pytest.fixture
def map_with_single_source(empty_map):
    parameters = RandomSourceSimulationParameters(
        n_sources=1,
        # Use bright sources so we can guarantee recovery
        min_flux=u.Quantity(9.9999, "Jy"),
        max_flux=u.Quantity(10.0001, "Jy"),
        fwhm_uncertainty_frac=0.01,
        fraction_return=1.0,
    )

    simulator = RandomSourceSimulation(parameters=parameters)

    new_map, sources = simulator.simulate(input_map=empty_map)

    yield new_map, sources


@pytest.fixture
def map_with_sources(empty_map):
    parameters = RandomSourceSimulationParameters(
        n_sources=10,
        # Use bright sources so we can guarantee recovery
        min_flux=u.Quantity(9.999, "Jy"),
        max_flux=u.Quantity(10.001, "Jy"),
        fwhm_uncertainty_frac=0.01,
        fraction_return=1.0,
    )

    simulator = RandomSourceSimulation(parameters=parameters)

    new_map, sources = simulator.simulate(input_map=empty_map)

    yield new_map, sources
