import numpy as np
import pytest

from sotrplib.maps.maps import clean_map


def _maps():
    # Footprint covers 40% of the map; the rest has zero weight.
    ivar = np.zeros((10, 10))
    ivar[:4] = 1.0
    ivar[3, :5] = 0.1  # low-weight strip inside the footprint
    return np.ones_like(ivar), ivar


@pytest.mark.parametrize("cut_on", ["median", "max"])
def test_clean_map_ignores_zero_weight_pixels(cut_on):
    imap, ivar = _maps()
    out = clean_map(imap, ivar, fraction=0.2, cut_on=cut_on)
    assert np.all(out[3, :5] == 0)
    assert np.all(out[:3] == 1)
    assert np.all(out[3, 5:] == 1)


def test_clean_map_percentile_uses_hit_pixels():
    imap, ivar = _maps()
    out = clean_map(imap, ivar, fraction=20, cut_on="percentile")
    assert np.all(out[3, :5] == 0)
    assert np.all(out[:3] == 1)


def test_clean_map_no_hits():
    imap = np.ones((4, 4))
    out = clean_map(imap, np.zeros((4, 4)), fraction=0.2, cut_on="median")
    assert np.all(out == 1)
