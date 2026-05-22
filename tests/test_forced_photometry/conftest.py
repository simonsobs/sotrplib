"""
Fixtures for the dependency-injected completely simulated pipeline.
"""

import datetime

import pytest
from astropy import units as u

from sotrplib.sims.maps import SimulatedMap


@pytest.fixture
def empty_map():
    sim_map = SimulatedMap(
        observation_start=datetime.datetime.now(tz=datetime.UTC)
        - datetime.timedelta(days=1),
        observation_end=datetime.datetime.now(tz=datetime.UTC),
    )

    sim_map.build()
    sim_map.finalize()

    assert sim_map.finalized

    yield sim_map


@pytest.fixture
def map_with_single_source(empty_map, fixed_source_map_builder):
    yield fixed_source_map_builder(
        empty_map,
        n_sources=1,
        min_flux=u.Quantity(9.9999, "Jy"),
        max_flux=u.Quantity(10.0001, "Jy"),
    )


@pytest.fixture
def map_with_sources(empty_map, fixed_source_map_builder):
    yield fixed_source_map_builder(
        empty_map,
        n_sources=10,
        min_flux=u.Quantity(9.999, "Jy"),
        max_flux=u.Quantity(10.001, "Jy"),
    )


@pytest.fixture
def map_with_single_asymmetric_source(empty_map, fixed_source_map_builder):
    yield fixed_source_map_builder(
        empty_map,
        n_sources=1,
        min_flux=u.Quantity(0.9999, "Jy"),
        max_flux=u.Quantity(1.0001, "Jy"),
        fwhm_uncertainty_fraction=0.5,
    )
