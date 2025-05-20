"""
Map testing
"""

import pytest
from pixell import enmap
import os

@pytest.fixture
def empty_map(tmp_path):
    full_path = str(tmp_path / "map.fits")
    enmap.zeros((64, 128)).write(full_path, fmt="fits")

    yield full_path
    os.unlink(full_path)