"""
Map testing
"""

import pytest
from pixell import enmap
import os

def create_map(full_path, shape=(64, 128), on: bool = False):
    if not on:
        enmap.zeros(shape).write(full_path, fmt="fits")
    else:
        enmap.ones(shape).write(full_path, fmt="fits")


@pytest.fixture
def ones_map_set(tmp_path):
    """
    Creates a set of five maps: map, ivar, rho, kappa and time.
    For now ivar is zero, and time is also a map of just zeros.
    """
    map_paths = {x: tmp_path / f"{x}.fits" for x in ["map", "ivar", "rho", "kappa", "time"]}

    create_map(map_paths["map"], on=True)
    create_map(map_paths["ivar"], on=False)
    create_map(map_paths["rho"], on=True)
    create_map(map_paths["kappa"], on=True)
    create_map(map_paths["time"], on=False)

    yield map_paths

    for x in map_paths.values():
        os.unlink(x)
