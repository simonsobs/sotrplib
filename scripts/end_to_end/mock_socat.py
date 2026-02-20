"""
Mock SOcat script for end-to-end testing
"""

import os
import pickle
import random

from astropy import units as u
from astropy.coordinates import ICRS
from socat.client.mock import Client

location = os.getenv("socat_client_pickle_path")

client = Client()

# Fill with some sample sources
for i in range(128):
    client.create_source(
        position=ICRS(
            ra=random.uniform(90 - 30, 90 + 30) * u.degree,
            dec=random.uniform(0 - 20, 0 + 20) * u.degree,
        ),
        flux=random.uniform(0.0, 1.0) * u.Jy,
        name=f"Source {i + 1}",
    )

with open(location, "wb") as f:
    pickle.dump(client, f)
