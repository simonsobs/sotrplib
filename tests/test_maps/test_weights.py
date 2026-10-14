import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time, TimeDelta
from pixell import enmap

from sotrplib.maps.core import RhoAndKappaMap
from sotrplib.maps.weights import (
    POOR_WEIGHTS_CENTER,
    POOR_WEIGHTS_EXTENDED,
    POOR_WEIGHTS_LOCAL,
    PoorWeightsCriteria,
    get_weight_thresholds,
    poor_weights_flag,
)

## 0.5 arcmin pixels: default local box is 11x11 pix, extended box is 41x41 pix.
RES = 0.5 * u.arcmin
CRITERIA = PoorWeightsCriteria()


@pytest.fixture
def weights():
    """
    Weights map with a smooth ramp of values from 1 to 2 so percentiles are
    well-defined and none of it is 'poor' relative to a patch set near zero.
    """
    box = np.array([[-2.0, 2.0], [2.0, -2.0]]) * u.deg.to(u.rad)
    shape, wcs = enmap.geometry(pos=box, res=RES.to_value(u.rad), proj="car")
    rng = np.random.default_rng(0)
    return enmap.ndmap(1.0 + rng.random(shape), wcs)


def pix_to_sky(weights, y, x):
    dec, ra = weights.pix2sky([y, x])
    return SkyCoord(ra=ra * u.rad, dec=dec * u.rad)


def center(weights):
    ny, nx = weights.shape
    return ny // 2, nx // 2


def flag_at(weights, y, x):
    thresholds = get_weight_thresholds(weights, CRITERIA)
    return poor_weights_flag(weights, pix_to_sky(weights, y, x), CRITERIA, thresholds)


def test_clean_source_not_flagged(weights):
    y, x = center(weights)
    assert flag_at(weights, y, x) is None


def test_center_pixel_flagged(weights):
    y, x = center(weights)
    weights[y, x] = 0.0
    assert flag_at(weights, y, x) == POOR_WEIGHTS_CENTER


def test_local_box_flagged(weights):
    y, x = center(weights)
    ## low-weight strip covering ~5/11 of the local box, but not the center pixel
    weights[y - 5 : y + 6, x + 1 : x + 6] = 1e-3
    assert flag_at(weights, y, x) == POOR_WEIGHTS_LOCAL


def test_extended_box_flagged(weights):
    y, x = center(weights)
    ## low-weight region filling more than half of the extended box,
    ## but outside the local box
    weights[y - 20 : y + 21, x + 6 : x + 21] = 1e-3
    weights[y - 20 : y + 21, x - 20 : x - 10] = 1e-3
    assert flag_at(weights, y, x) == POOR_WEIGHTS_EXTENDED


def test_nan_weights_count_as_low(weights):
    y, x = center(weights)
    weights[y - 5 : y + 6, x + 1 : x + 6] = np.nan
    assert flag_at(weights, y, x) == POOR_WEIGHTS_LOCAL


def test_source_near_edge_uses_clipped_box(weights):
    ## box is clipped at the edge instead of wrapping or coming back empty
    assert flag_at(weights, 2, 2) is None


def test_source_off_map_flagged(weights):
    thresholds = get_weight_thresholds(weights, CRITERIA)
    off_map = SkyCoord(ra=30 * u.deg, dec=30 * u.deg)
    assert (
        poor_weights_flag(weights, off_map, CRITERIA, thresholds) == POOR_WEIGHTS_CENTER
    )


def test_thresholds_ignore_invalid_pixels(weights):
    weights[:10] = 0.0
    weights[-10:] = np.nan
    thresholds = get_weight_thresholds(weights, CRITERIA)
    assert thresholds.center >= 1.0
    assert thresholds.center <= thresholds.local <= thresholds.extended


def test_thresholds_none_without_valid_pixels(weights):
    weights[:] = 0.0
    assert get_weight_thresholds(weights, CRITERIA) is None


def test_finalize_keeps_kappa_as_weights(separate_map_set_1):
    start_time = Time("2025-10-10", format="iso")
    input_map = RhoAndKappaMap(
        rho_filename=separate_map_set_1["rho"],
        kappa_filename=separate_map_set_1["kappa"],
        time_filename=separate_map_set_1["time"],
        frequency="f090",
        start_time=start_time,
        end_time=start_time + TimeDelta(3600, format="sec"),
    )
    input_map.build()
    kappa = input_map.kappa
    input_map.finalize()

    assert input_map.weights is kappa
    with pytest.raises(AttributeError):
        _ = input_map.kappa
