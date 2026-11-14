"""Tests for the RA-wrap-aware map union used by the coadders."""

import numpy as np
import pytest
from pixell import enmap

from sotrplib.maps.utils import pixell_map_union, ra_wrap_pixels, union_geometry

RES = np.deg2rad(1.0)
NWRAP = 360


def _strip(ra_from: float, ra_to: float, value: float = 1.0, shift: int = 0):
    """A 1 deg CAR map from RA ra_from down to ra_to (deg), Dec -5..5.
    shift moves crpix by whole wraps (the same sky)."""
    shape, wcs = enmap.geometry(
        pos=np.deg2rad([[-5, ra_from], [5, ra_to]]), res=RES, proj="car"
    )
    wcs.wcs.crpix[0] += shift * NWRAP
    return enmap.full(shape, wcs, value)


def _value_at(m, ra: float) -> float:
    return float(m.at(np.deg2rad([0.0, ra]), order=0))


def test_ra_wrap_pixels():
    assert ra_wrap_pixels(_strip(10, 0).wcs) == NWRAP


def test_far_from_seam_matches_pixell():
    a, b = _strip(40, 20, 1.0), _strip(30, 0, 2.0)
    ours = pixell_map_union(a, b)
    ref = enmap.map_union(a, b)
    assert ours.shape == ref.shape
    assert np.array_equal(np.asarray(ours), np.asarray(ref))


@pytest.mark.parametrize("shift", [0, -1, 2])
def test_seam_union_stays_narrow(shift):
    # Two patches either side of RA 180, on different RA branches.
    a = _strip(180, 170, 1.0)
    b = _strip(-170, -180, 2.0, shift=shift)
    assert enmap.union_geometry([a.geometry, b.geometry])[0][-1] > NWRAP / 2
    u = pixell_map_union(a, b)
    assert u.shape[-1] == 20
    assert _value_at(u, 175.0) == 1.0
    assert _value_at(u, -175.0) == 2.0
    assert np.isclose(u.sum(), a.sum() + b.sum())


def test_full_ra_union_is_capped_at_one_wrap():
    # Strips that cover all RA, with an overlap, one on another branch.
    strips = [
        _strip(180, 60, 1.0),
        _strip(70, -60, 1.0, shift=-1),
        _strip(-50, -180, 1.0, shift=1),
    ]
    u = strips[0]
    for s in strips[1:]:
        u = pixell_map_union(u, s)
    assert u.shape[-1] == NWRAP
    assert np.isclose(u.sum(), sum(s.sum() for s in strips))
    ra = np.rad2deg(u.posmap()[1][0])
    assert np.isclose(ra[0], 179.5) or np.isclose(ra[0], 180.0)
    assert _value_at(u, 65.0) == 2.0  # overlap of strips 0 and 1
    assert _value_at(u, 0.0) == 1.0


def test_union_geometry_order_does_not_matter_for_width():
    a = _strip(180, 170, 1.0)
    b = _strip(-170, -180, 2.0, shift=-1)
    g1 = union_geometry([a.geometry, b.geometry])
    g2 = union_geometry([b.geometry, a.geometry])
    assert g1[0] == g2[0]
