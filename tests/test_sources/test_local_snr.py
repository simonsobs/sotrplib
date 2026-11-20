"""
Tests for the local SNR that the sifter calculates for transient candidates.
"""

from types import SimpleNamespace

import numpy as np
from astropy import units as u
from pixell import enmap
from pixell.utils import arcmin, degree

from sotrplib.sifter.crossmatch import recalculate_local_snr
from sotrplib.sources.sources import MeasuredSource

RA, DEC = 30.0, 0.0
FWHM = 2.2 * u.arcmin


def _map(seed: int = 1) -> enmap.ndmap:
    box = np.array([[DEC - 1, RA - 1], [DEC + 1, RA + 1]]) * degree
    shape, wcs = enmap.geometry(pos=box, res=0.5 * arcmin, proj="car")
    rng = np.random.default_rng(seed)
    return enmap.ndmap(rng.normal(0.0, 1.0, shape), wcs)


def _add_beam(imap: enmap.ndmap, amplitude: float) -> enmap.ndmap:
    dist = imap.modrmap(ref=[DEC * degree, RA * degree])
    sigma = FWHM.to_value(u.rad) / np.sqrt(8 * np.log(2))
    return imap + amplitude * np.exp(-0.5 * (dist / sigma) ** 2)


def _snr(flux_map: enmap.ndmap, flux: float, snr: float) -> tuple[float, bool]:
    candidate = MeasuredSource(
        ra=RA * u.deg, dec=DEC * u.deg, flux=flux * u.mJy, snr=snr
    )
    transients, _ = recalculate_local_snr(
        [candidate],
        SimpleNamespace(flux=flux_map, flux_units=u.mJy),
        thumb_size=0.25 * u.deg,
        fwhm=FWHM,
    )
    return candidate.snr, bool(transients)


def test_noise_only_snr_unchanged():
    snr, kept = _snr(_map(), flux=10.0, snr=10.0)
    assert abs(snr - 10.0) < 0.5
    assert kept


def test_masked_pixels_do_not_inflate_snr():
    imap = _map()
    # Mask half the thumbnail, as at the edge of the map.
    dec_pix, _ = imap.sky2pix([DEC * degree, RA * degree])
    imap[int(dec_pix) + 3 :, :] = 0.0
    snr, _ = _snr(imap, flux=10.0, snr=10.0)
    assert abs(snr - 10.0) < 0.5


def test_bright_source_wings_do_not_lower_snr():
    snr, kept = _snr(_add_beam(_map(), 100.0), flux=100.0, snr=100.0)
    assert abs(snr - 100.0) < 5.0
    assert kept


def test_too_few_pixels_keeps_snr():
    imap = _map() * 0.0
    snr, kept = _snr(imap, flux=10.0, snr=10.0)
    assert snr == 10.0
    assert kept
