"""
Tests for thumbnails near NaN (masked) map pixels.
"""

from types import SimpleNamespace

import numpy as np
from astropy import units as u
from pixell import enmap, reproject
from pixell.utils import arcmin, degree

from sotrplib.maps.maps import get_thumbnail, nan_safe_thumbnail
from sotrplib.sources.sources import MeasuredSource

RA, DEC = 30.0, 0.0
FWHM = 2.2 * arcmin
AMPLITUDE = 100.0
R = 11 * arcmin  # 5 * FWHM, as for an f090 blind-search thumbnail


def _map_with_hole(hole_offset_arcmin: float = 8.0) -> enmap.ndmap:
    box = np.array([[DEC - 1, RA - 1], [DEC + 1, RA + 1]]) * degree
    shape, wcs = enmap.geometry(pos=box, res=0.5 * arcmin, proj="car")
    imap = enmap.zeros(shape, wcs)
    dist = imap.modrmap(ref=[DEC * degree, RA * degree])
    imap += AMPLITUDE * np.exp(-0.5 * (dist / (FWHM / np.sqrt(8 * np.log(2)))) ** 2)
    if hole_offset_arcmin is not None:
        # A 3x3 arcmin masked region to the north of the source.
        ny, nx = imap.sky2pix([(DEC + hole_offset_arcmin / 60) * degree, RA * degree])
        imap[int(ny) - 3 : int(ny) + 3, int(nx) - 3 : int(nx) + 3] = np.nan
    return imap


def _centre(thumb):
    ny, nx = thumb.shape
    return thumb[ny // 2, nx // 2]


def test_pixell_thumbnail_is_all_nan_near_a_hole():
    # The behaviour that nan_safe_thumbnail works around.
    thumb = reproject.thumbnails(_map_with_hole(), [DEC * degree, RA * degree], r=R)
    assert np.all(np.isnan(thumb))


def test_thumbnail_near_a_hole_keeps_the_source():
    thumb = nan_safe_thumbnail(_map_with_hole(), DEC * degree, RA * degree, r=R)
    assert np.isfinite(_centre(thumb))
    assert abs(np.nanmax(thumb) - AMPLITUDE) < 0.02 * AMPLITUDE
    n_nan = np.sum(np.isnan(thumb))
    assert 0 < n_nan < 0.1 * thumb.size


def test_thumbnail_nan_pixels_are_at_the_hole():
    thumb = nan_safe_thumbnail(_map_with_hole(), DEC * degree, RA * degree, r=R)
    dec_off = thumb.posmap()[0] / arcmin  # dec offset from the centre, arcmin
    assert np.all(dec_off[np.isnan(thumb)] > 4.0)


def test_thumbnail_without_nan_is_unchanged():
    imap = _map_with_hole(hole_offset_arcmin=None)
    pos = [DEC * degree, RA * degree]
    np.testing.assert_array_equal(
        nan_safe_thumbnail(imap, *pos, r=R), reproject.thumbnails(imap, pos, r=R)
    )


def test_thumbnail_outside_the_map_stays_nan():
    imap = enmap.full(
        *enmap.geometry(pos=np.array([[-1, 29], [1, 31]]) * degree, res=0.5 * arcmin),
        np.nan,
    )
    thumb = nan_safe_thumbnail(imap, DEC * degree, RA * degree, r=R)
    assert np.all(np.isnan(thumb))


def test_get_thumbnail_near_a_hole():
    thumb = get_thumbnail(_map_with_hole(), RA, DEC, size_deg=R / degree)
    assert np.isfinite(_centre(thumb))
    assert np.any(np.isnan(thumb))


def test_extract_thumbnail_near_a_hole():
    source = MeasuredSource(
        ra=RA * u.deg, dec=DEC * u.deg, flux=AMPLITUDE * u.mJy, snr=10.0
    )
    input_map = SimpleNamespace(
        flux=_map_with_hole(), map_resolution=0.5 * u.arcmin, flux_units=u.mJy
    )
    source.extract_thumbnail(input_map, thumb_width=R * u.rad, reproject_thumb=True)
    assert np.isfinite(_centre(source.thumbnail))
    assert abs(np.nanmax(source.thumbnail) - AMPLITUDE) < 0.02 * AMPLITUDE


def test_large_masked_region_is_nan_and_only_there():
    imap = _map_with_hole(hole_offset_arcmin=None)
    # Mask everything 3 arcmin or more to the south, as at the edge of a map.
    imap[imap.posmap()[0] < (DEC - 3 / 60) * degree] = np.nan
    for proj in [None, "tan"]:
        thumb = nan_safe_thumbnail(imap, DEC * degree, RA * degree, r=R, proj=proj)
        dec_off = thumb.posmap()[0] / arcmin
        assert np.all(np.isnan(thumb[dec_off < -3.5]))
        assert np.all(np.isfinite(thumb[dec_off > -2.5]))
