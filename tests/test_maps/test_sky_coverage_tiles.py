"""
The mapcat sky-coverage tiles of a source, in the two RA conventions of
the tiles.
"""

import pytest

from sotrplib.maps.database import sky_coverage_tiles


@pytest.mark.parametrize(
    "ra, dec, expected",
    [
        (15.0, 0.5, [(1, 9)]),
        (179.9, -5.0, [(17, 8)]),
        (350.75, -0.9, [(35, 8)]),
        (180.0, 10.0, [(18, 10)]),
    ],
)
def test_sky_coverage_tiles(ra, dec, expected):
    assert sky_coverage_tiles(ra, dec) == expected
