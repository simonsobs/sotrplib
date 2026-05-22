"""
Fixtures for the dependency-injected completely simulated pipeline.
"""

import datetime

import pytest
from astropy import units as u

from sotrplib.sims.maps import SimulatedMap


@pytest.fixture
def empty_map():
    map = SimulatedMap(
        observation_start=datetime.datetime.now(tz=datetime.timezone.utc)
        - datetime.timedelta(days=1),
        observation_end=datetime.datetime.now(tz=datetime.timezone.utc),
    )

    map.build()
    map.finalize()

    assert map.finalized

    yield map


@pytest.fixture
def map_with_single_source(empty_map, fixed_source_map_builder):
    yield fixed_source_map_builder(
        empty_map,
        n_sources=1,
        min_flux=u.Quantity(1.0, "Jy"),
        max_flux=u.Quantity(10.0, "Jy"),
        fwhm_uncertainty_fraction=0.5,
    )


@pytest.fixture
def map_with_sources(empty_map, fixed_source_map_builder):
    yield fixed_source_map_builder(
        empty_map,
        n_sources=4,
        min_flux=u.Quantity(1.0, "Jy"),
        max_flux=u.Quantity(10.0, "Jy"),
    )
