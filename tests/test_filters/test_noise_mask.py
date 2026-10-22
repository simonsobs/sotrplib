"""
Tests for the noise model mask of the matched filter and the snr rms mask.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from astropy import units as u
from pixell import enmap
from pixell import utils as pixell_utils

from sotrplib.filters.filters import matched_filter_depth1_map
from sotrplib.maps.maps import (
    grow_mask,
    snr_rms_tile_mask,
    snr_tile_rms,
    tiles_to_pixels,
)
from sotrplib.maps.postprocessor import SnrRmsMask


def test_snr_tile_rms_gaussian_noise():
    rng = np.random.default_rng(2)
    snr = rng.normal(0, 1, (100, 150))
    snr[:50, :50] *= 2
    rms = snr_tile_rms(snr, np.ones(snr.shape, bool), tile_pix=50)

    assert rms.shape == (2, 3)
    assert rms[0, 0] == pytest.approx(2, rel=0.1)
    assert np.delete(rms.ravel(), 0) == pytest.approx(1, rel=0.1)


def test_snr_tile_rms_few_valid_pixels_is_nan():
    snr = np.ones((20, 20))
    valid = np.zeros(snr.shape, bool)
    valid[:3, :3] = True

    assert np.isnan(snr_tile_rms(snr, valid, tile_pix=10)).all()


def test_tiles_to_pixels_crops_edge_tiles():
    pixels = tiles_to_pixels(np.array([[1, 2], [3, 4]]), (15, 12), tile_pix=10)

    assert pixels.shape == (15, 12)
    assert pixels[0, 0] == 1 and pixels[0, 11] == 2 and pixels[14, 0] == 3


def test_snr_rms_tile_mask_limits_and_grow():
    rng = np.random.default_rng(3)
    snr = rng.normal(0, 1, (100, 100))
    snr[40:60, 40:60] *= 3
    snr[:20, 80:] *= 0.3
    valid = np.ones(snr.shape, bool)

    bad, _ = snr_rms_tile_mask(snr, valid, tile_pix=20, max_rms=1.2)
    assert bad[50, 50] and not bad[50, 30] and not bad[10, 90]

    bad, _ = snr_rms_tile_mask(snr, valid, tile_pix=20, max_rms=1.2, min_rms=0.75)
    assert bad[50, 50] and bad[10, 90]

    bad, _ = snr_rms_tile_mask(snr, valid, tile_pix=20, max_rms=1.2, grow_tiles=1)
    assert bad[30, 30] and not bad[10, 10]


def test_snr_rms_tile_mask_bright_source_does_not_make_tile_bad():
    # Noise rms 1.1 (below the limit) and a 100 sigma source with wide wings.
    rng = np.random.default_rng(4)
    snr = rng.normal(0, 1.1, (100, 100))
    yy, xx = np.mgrid[:100, :100]
    snr += 100 * np.exp(-((yy - 25) ** 2 + (xx - 25) ** 2) / (2 * 4.0**2))
    valid = np.ones(snr.shape, bool)

    bad, rms = snr_rms_tile_mask(
        snr, valid, tile_pix=50, max_rms=1.2, source_nsigma=None
    )
    assert bad[25, 25] and rms[0, 0] > 1.2

    bad, rms = snr_rms_tile_mask(
        snr, valid, tile_pix=50, max_rms=1.2, source_nsigma=5.0, source_radius_pix=6
    )
    assert not bad.any() and rms[0, 0] < 1.2


def test_grow_mask():
    mask = np.zeros((20, 30), bool)
    mask[10, 15] = True
    mask[0, 0] = True
    grown = grow_mask(mask, 2)
    assert grown[8:13, 13:18].all() and grown[:3, :3].all()
    assert grown.sum() == 25 + 9


def _stripe_map():
    """
    White noise map with a stripe of 6 times more noise. The ivar map does not
    show the stripe, as for the end-of-obs noise of the depth-1 maps. Also
    returns the map without the stripe.
    """
    shape, wcs = enmap.geometry(
        pos=np.array([[-2, 2], [2, -2]]) * pixell_utils.degree,
        res=0.5 * pixell_utils.arcmin,
    )
    rng = np.random.default_rng(1)
    sigma_noise = 1e-5  # K per pixel
    clean = enmap.enmap(rng.normal(0, sigma_noise, shape), wcs)
    stripe = np.zeros(shape, bool)
    stripe[:, 360:420] = True
    imap = clean.copy()
    imap[stripe] *= 6
    ivarmap = enmap.ones(shape, wcs) / sigma_noise**2
    return imap, clean, ivarmap, stripe


def test_noise_model_mask_removes_stripe_from_noise_model():
    imap, clean, ivarmap, stripe = _stripe_map()
    quiet = np.zeros(stripe.shape, bool)
    quiet[40:-40, 40:300] = True

    def snr(imap, noise_model_mask=None):
        rho, kappa = matched_filter_depth1_map(
            imap.copy(),
            ivarmap.copy(),
            band_center=90 * u.GHz,
            beam_fwhm=2.2 * u.arcmin,
            noise_model_mask=noise_model_mask,
        )
        return np.asarray(rho / np.sqrt(kappa))

    keep = enmap.ndmap((~stripe).astype(float), imap.wcs)
    mask = np.asarray(enmap.apod_mask(keep > 0, (10 * u.arcmin).to_value(u.rad)))
    rms_clean = np.std(snr(clean)[quiet])
    rms_full = np.std(snr(imap)[quiet])
    rms_masked = np.std(snr(imap, mask)[quiet])

    # The stripe makes the noise model too large, thus the snr rms of the quiet
    # region is too small. Without the stripe in the noise model, the snr rms
    # is the same as for the map without the stripe.
    assert rms_full < 0.75 * rms_clean
    assert rms_masked == pytest.approx(rms_clean, rel=0.06)
    # The mask does not change the result of a map without the stripe.
    assert np.std(snr(clean, mask)[quiet]) == pytest.approx(rms_clean, rel=0.03)


def test_snr_rms_mask_postprocessor():
    rng = np.random.default_rng(4)
    shape, wcs = enmap.geometry(
        pos=np.array([[-1, 1], [1, -1]]) * pixell_utils.degree,
        res=0.5 * pixell_utils.arcmin,
    )
    snr = enmap.enmap(rng.normal(0, 1, shape), wcs)
    snr[:, :60] *= 2
    input_map = SimpleNamespace(
        snr=snr,
        flux=snr * 10,
        mask=None,
        res=0.5 * u.arcmin,
        map_name="test",
    )

    out = SnrRmsMask(max_rms=1.2, tile_size=0.5 * u.deg).postprocess(input_map)

    assert (out.snr[:, :60] == 0).all() and (out.flux[:, :60] == 0).all()
    assert (out.mask[:, :60] == 0).all() and (out.mask[:, 60:] == 1).all()
    assert np.std(out.snr[:, 60:]) == pytest.approx(1, rel=0.1)
