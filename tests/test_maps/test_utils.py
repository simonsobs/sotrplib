import numpy as np
from pixell import enmap
from pixell import utils as pixell_utils

from sotrplib.maps.utils import enmap_footprint_box


def _dense_pix2sky_dec_extrema(shape, wcs, n=80):
    ny, nx = shape[-2:]
    ys = np.linspace(-0.5, ny - 0.5, n)
    xs = np.linspace(-0.5, nx - 0.5, n)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    dec, _ = enmap.pix2sky(shape, wcs, np.array([yy.ravel(), xx.ravel()]))
    return dec.min(), dec.max()


def test_enmap_footprint_box_matches_enmap_box_for_car():
    """
    For a rectilinear (CAR) projection, dec and ra vary independently along
    each pixel axis, so the corner-to-corner diagonal enmap.box() uses
    already bounds the map, and the edge-sampled box should agree with it.
    """
    shape, wcs = enmap.geometry(
        pos=[0, -40 * pixell_utils.degree],
        shape=(200, 400),
        res=1 * pixell_utils.arcmin,
        proj="car",
    )
    old_box = np.asarray(enmap.box(shape, wcs))
    new_box = np.asarray(enmap_footprint_box(shape, wcs, nsamp=50))

    np.testing.assert_allclose(sorted(new_box[:, 0]), sorted(old_box[:, 0]), atol=1e-6)
    np.testing.assert_allclose(sorted(new_box[:, 1]), sorted(old_box[:, 1]), atol=1e-6)


def test_enmap_footprint_box_captures_true_zea_footprint():
    """
    On a curved (ZEA) projection, like the SPT3G depth-1 maps this fixes,
    enmap.box()'s corner-to-corner diagonal can undercount the map's true
    footprint by many degrees: the dec/ra extremum is reached partway along
    an edge, not at a corner. The edge-sampled box must not undercount it.

    This is the bug that let known SPT3G point sources sitting outside the
    (wrong) box slip past the socat crossmatch in the sifter and get
    misclassified as new blind-search transient candidates.
    """
    shape, wcs = enmap.geometry(
        pos=[0, -57.5 * pixell_utils.degree],
        shape=(2000, 6000),
        res=0.5 * pixell_utils.arcmin,
        proj="zea",
    )

    true_dec_min, true_dec_max = _dense_pix2sky_dec_extrema(shape, wcs)

    old_box = np.asarray(enmap.box(shape, wcs))
    new_box = np.asarray(enmap_footprint_box(shape, wcs, nsamp=200))

    old_dec_min, old_dec_max = sorted(old_box[:, 0])
    new_dec_min, new_dec_max = sorted(new_box[:, 0])

    # The edge-sampled box matches the densely-sampled true footprint...
    np.testing.assert_allclose(new_dec_min, true_dec_min, atol=1e-4)
    np.testing.assert_allclose(new_dec_max, true_dec_max, atol=1e-4)

    # ...while the corner-only box measurably undercounts it.
    assert old_dec_min > true_dec_min + 1e-3
    assert old_dec_max < true_dec_max - 1e-3
