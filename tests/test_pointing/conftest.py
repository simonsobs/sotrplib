"""
Fixtures for the dependency-injected completely simulated pipeline.
"""

import datetime

import pytest
from astropy import units as u

from sotrplib.sims.maps import SimulatedMap, SimulationParameters
from sotrplib.sims.sources.core import (
    RandomSourceSimulation,
    RandomSourceSimulationParameters,
)

map_sim_params = SimulationParameters(
    center_ra=u.Quantity(20.0, "deg"),
    center_dec=u.Quantity(-1.0, "deg"),
    width_ra=u.Quantity(4.0, "deg"),
    width_dec=u.Quantity(4.0, "deg"),
    resolution=u.Quantity(0.5, "arcmin"),
    map_noise=u.Quantity(0.01, "Jy"),
)


@pytest.fixture
def empty_map():
    map = SimulatedMap(
        simulation_parameters=map_sim_params,
        observation_start=datetime.datetime.now(tz=datetime.UTC),
        observation_end=datetime.datetime.now(tz=datetime.UTC)
        + datetime.timedelta(hours=8),
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
        min_flux=u.Quantity(0.9999, "Jy"),
        max_flux=u.Quantity(1.0001, "Jy"),
        fwhm_uncertainty_frac=0.01,
        fraction_return=1.0,
    )

    simulator = RandomSourceSimulation(parameters=parameters)

    new_map, sources = simulator.simulate(input_map=empty_map)

    yield new_map, sources


@pytest.fixture
def map_with_sources(empty_map):
    parameters = RandomSourceSimulationParameters(
        n_sources=30,
        # Use bright sources so we can guarantee recovery
        min_flux=u.Quantity(0.999, "Jy"),
        max_flux=u.Quantity(1.001, "Jy"),
        fwhm_uncertainty_frac=0.01,
        fraction_return=1.0,
    )

    simulator = RandomSourceSimulation(parameters=parameters)

    new_map, sources = simulator.simulate(input_map=empty_map)

    yield new_map, sources
