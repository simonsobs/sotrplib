"""
Map testing
"""

import pytest
from pixell import enmap
import os


def create_map(tmp_path, maptype="map", shape=(64, 128), value=0):
    full_path = str(tmp_path / f"{maptype}.fits")
    if value == 0:
        enmap.zeros(shape).write(full_path, fmt="fits")
    else:
        enmap.ones(shape).write(full_path, fmt="fits")

    yield full_path
    os.unlink(full_path)

@pytest.fixture
def empty_map(tmp_path,maptype="map"):
    full_path = create_map(tmp_path, maptype=maptype, value=0)
    return full_path
   
@pytest.fixture
def ones_map(tmp_path,maptype="map"):
    full_path = create_map(tmp_path, maptype=maptype, value=1)
    return full_path

@pytest.fixture
def ones_map_set(tmp_path):
    map_paths = {}
    for maptype in ["map", "ivar", "rho", "kappa", "time"]:
        full_path = create_map(tmp_path, maptype=maptype, value=1)
        map_paths[maptype] = next(full_path,None)

    return map_paths
