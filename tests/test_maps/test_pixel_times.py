"""
Gap fill of pixel times in ProcessableMap.get_pixel_times.
"""

import numpy as np
import pytest
from astropy.time import Time
from pixell import enmap

from sotrplib.maps.core import IntensityAndInverseVarianceMap, RhoAndKappaMap

N = 20
T0 = 1.7e9
DT = 10.0
START = Time(T0, format="unix")
END = Time(T0 + 600, format="unix")


def _blank():
    shape, wcs = enmap.geometry(
        pos=np.deg2rad([[-1, -1], [1, 1]]), shape=(N, N), proj="car"
    )
    return enmap.zeros(shape, wcs)


def _time_map():
    """Time increases by DT per column."""
    tmap = _blank()
    tmap[:] = T0 + DT * np.arange(N)[None, :]
    return tmap


def _make_map(kind, weight, tmap):
    """
    kind="ivar": hits = ivar > 0 (zero in time gaps).
    kind="kappa": hits = kappa > 0 (kappa stays > 0 in time gaps).
    """
    if kind == "ivar":
        m = IntensityAndInverseVarianceMap(
            intensity_filename=None,
            inverse_variance_filename=None,
            start_time=START,
            end_time=END,
        )
        m.inverse_variance = weight
    else:
        m = RhoAndKappaMap(
            rho_filename=None, kappa_filename=None, start_time=START, end_time=END
        )
        m.kappa = weight
    m.time_first = tmap.copy()
    m.time_mean = tmap.copy()
    m.time_last = tmap.copy()
    return m


def _build(kind, gap):
    """
    Map with time map set to 0 in `gap`. For ivar maps the weight is also 0 in
    `gap`; for kappa maps the weight stays 1, as for a smoothed kappa.
    """
    tmap = _time_map()
    tmap[gap] = 0
    weight = _blank() + 1
    if kind == "ivar":
        weight[gap] = 0
    return _make_map(kind, weight, tmap)


def _offset(m, pix):
    times = m.get_pixel_times(pix)
    return [None if t is None else t.unix - T0 for t in times]


KINDS = ["ivar", "kappa"]


@pytest.mark.parametrize("kind", KINDS)
def test_valid_pixel_unchanged(kind):
    m = _build(kind, (slice(0, 0), slice(0, 0)))
    assert _offset(m, (5, 7)) == pytest.approx([70.0] * 3)


@pytest.mark.parametrize("kind", KINDS)
def test_single_missing_pixel(kind):
    m = _build(kind, (10, 10))
    assert _offset(m, (10, 10)) == pytest.approx([100.0] * 3)


@pytest.mark.parametrize("kind", KINDS)
def test_streak_one_pixel_wide(kind):
    m = _build(kind, (slice(None), 10))
    for row in (0, 5, N - 1):
        assert _offset(m, (row, 10)) == pytest.approx([100.0] * 3)


@pytest.mark.parametrize("kind", KINDS)
def test_streak_three_pixels_wide_uses_5x5(kind):
    m = _build(kind, (slice(None), slice(9, 12)))
    assert _offset(m, (5, 10)) == pytest.approx([100.0] * 3)
    assert _offset(m, (5, 9)) == pytest.approx([80.0] * 3)


@pytest.mark.parametrize("kind", KINDS)
def test_streak_too_wide_gives_none(kind):
    m = _build(kind, (slice(None), slice(8, 13)))
    assert _offset(m, (5, 10)) == [None, None, None]


@pytest.mark.parametrize("kind", KINDS)
def test_edge_without_times(kind):
    """Last 3 columns have no times; fill reaches 2 pixels from valid data."""
    m = _build(kind, (slice(None), slice(N - 3, None)))
    assert _offset(m, (5, N - 3)) == pytest.approx([DT * (N - 4)] * 3)
    assert _offset(m, (5, N - 2)) == pytest.approx([DT * (N - 4)] * 3)
    assert _offset(m, (5, N - 1)) == [None, None, None]


@pytest.mark.parametrize("kind", KINDS)
def test_corner_pixel(kind):
    m = _build(kind, (0, 0))
    # neighbours (0,1)=10, (1,0)=0, (1,1)=10
    assert _offset(m, (0, 0)) == pytest.approx([20.0 / 3] * 3)


def test_fill_is_hits_weighted():
    """Coadd maps carry hit counts; the fill weights by them."""
    tmap = _time_map()
    tmap[:, 10] = 0
    m = _make_map("kappa", _blank() + 1, tmap)
    hits = _blank() + 1
    hits[:, 11:] = 3
    hits[:, 10] = 0
    m._hits = hits
    # (90 * 3 + 110 * 9) / 12
    assert _offset(m, (5, 10)) == pytest.approx([105.0] * 3)


def test_zero_time_neighbour_not_used():
    """A neighbour with hits but time 0 must not pull the mean to 1970."""
    m = _build("kappa", (5, slice(10, 12)))
    # (5,10) neighbours exclude (5,11); mean of 90,100,110 rows 4 and 6 and (5,9)
    assert _offset(m, (5, 10)) == pytest.approx([(90 * 3 + 100 * 2 + 110 * 2) / 7] * 3)


def test_nan_time_is_missing():
    tmap = _time_map()
    tmap[10, 10] = np.nan
    m = _make_map("kappa", _blank() + 1, tmap)
    assert _offset(m, (10, 10)) == pytest.approx([100.0] * 3)


def test_no_time_map_uses_observation_times():
    m = _build("kappa", (slice(0, 0), slice(0, 0)))
    m.time_first = m.time_mean = m.time_last = None
    assert _offset(m, (5, 5)) == pytest.approx([0.0, 300.0, 600.0])
