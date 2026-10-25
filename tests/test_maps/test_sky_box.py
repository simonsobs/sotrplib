"""
Tests for the conversion of a config sky box to a pixell box.
"""

import numpy as np
from astropy import units as u
from astropy.coordinates import ICRS, SkyCoord

from sotrplib.config.maps import MapCatDatabaseConfig
from sotrplib.maps.utils import skycoord_box_to_enmap_box


def test_frame_and_skycoord_corners_give_the_same_box():
    frames = (
        ICRS(ra=347.8 * u.deg, dec=2.2 * u.deg),
        ICRS(ra=348.8 * u.deg, dec=3.2 * u.deg),
    )
    skycoords = tuple(SkyCoord(f) for f in frames)

    expected = np.deg2rad([[2.2, 348.8], [3.2, 347.8]])
    np.testing.assert_allclose(skycoord_box_to_enmap_box(frames), expected)
    np.testing.assert_allclose(skycoord_box_to_enmap_box(skycoords), expected)


def test_config_sky_box_converts():
    """The sky_box of a mapcat_database config (frame objects) converts."""
    config = MapCatDatabaseConfig.model_validate(
        {
            "sky_box": [
                {
                    "ra": {"value": 359.5, "unit": "deg"},
                    "dec": {"value": -1, "unit": "deg"},
                },
                {
                    "ra": {"value": 0.5, "unit": "deg"},
                    "dec": {"value": 1, "unit": "deg"},
                },
            ]
        }
    )
    box = skycoord_box_to_enmap_box(config.sky_box)
    # The box wraps through RA 0: ra_min is moved below zero.
    np.testing.assert_allclose(box, np.deg2rad([[-1, 0.5], [1, 359.5 - 360]]))
