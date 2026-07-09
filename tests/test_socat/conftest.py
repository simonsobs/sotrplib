"""
Fixtures for things that rely on socat.
"""

import os
import pickle
import random

import pytest
from astropy import units as u
from astropy.coordinates import ICRS


@pytest.fixture
def socat_pickle(tmp_path_factory):
    """
    Path to a temporary pickle file for socat tests.
    """
    location = tmp_path_factory.mktemp("socat") / "socat.pkl"

    os.environ["socat_client_client_type"] = "pickle"
    os.environ["socat_client_pickle_path"] = str(location)

    from socat.client.mock import Client

    client = Client()

    # Fill with some sample sources
    for i in range(128):
        client.create_source(
            position=ICRS(
                ra=random.uniform(0, 360) * u.degree,
                dec=random.uniform(-90, 90) * u.degree,
            ),
            flux=random.uniform(0.01, 1.0) * u.Jy,
            name=f"Source {i + 1}",
        )

    with open(location, "wb") as f:
        pickle.dump(client, f)

    yield location
