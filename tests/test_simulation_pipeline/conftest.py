"""
Fixtures for the dependency-injected completely simulated pipeline.
"""

import datetime

import pytest

from sotrplib.maps.core import SimulatedMap


@pytest.fixture
def empty_map():
    map = SimulatedMap(
        observation_start=datetime.datetime.now() - datetime.timedelta(days=1),
        observation_end=datetime.datetime.now(),
    )

    map.build()

    yield map
