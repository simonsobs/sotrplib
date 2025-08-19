"""
Fixtures for the dependency-injected completely simulated pipeline.
"""

import datetime

import pytest

from sotrplib.maps.core import SimulatedMap


@pytest.fixture
def empty_map():
    map = SimulatedMap(
        start_time=datetime.datetime.now() - datetime.timedelta(days=1),
        end_time=datetime.datetime.now(),
    )

    map.build()

    yield map
