import numpy as np
import pytest
from astropy import units as u
from pixell import enmap
from pixell import utils as pixell_utils

from sotrplib.filters.filters import matched_filter_depth1_map


def test_matched_filter_requires_beam():
    """
    Without beam1d or beam_fwhm, the matched filter raises a ValueError.
    """
    shape, wcs = enmap.geometry(
        pos=np.array([[-1, 1], [1, -1]]) * pixell_utils.degree,
        res=0.5 * pixell_utils.arcmin,
    )
    imap = enmap.zeros(shape, wcs)
    ivarmap = enmap.ones(shape, wcs)

    with pytest.raises(ValueError, match="beam1d or beam_fwhm"):
        matched_filter_depth1_map(imap, ivarmap, band_center=90 * u.GHz)


def test_matched_filter_recovers_point_sources():
    """
    The matched filter of a map with Gaussian point sources in white noise
    gives flux = rho / kappa near the injected flux, at the source pixel.
    The injected flux is peak * fconv * 2 pi sigma**2 (mJy).
    """
    fwhm = 2.2 * u.arcmin
    shape, wcs = enmap.geometry(
        pos=np.array([[-2, 2], [2, -2]]) * pixell_utils.degree,
        res=0.5 * pixell_utils.arcmin,
    )
    rng = np.random.default_rng(1)
    sigma_noise = 1e-5  # K per pixel
    ivarmap = enmap.ones(shape, wcs) / sigma_noise**2
    imap = enmap.enmap(rng.normal(0, sigma_noise, shape), wcs)
    sources = [(0.0, 0.0, 1e-3), (0.8, -0.7, 5e-4), (-0.9, 0.6, 2e-4)]  # dec, ra, K
    pos = imap.posmap()
    bsigma = fwhm.to_value(u.rad) * pixell_utils.fwhm
    for dec, ra, amp in sources:
        dec, ra = dec * pixell_utils.degree, ra * pixell_utils.degree
        r2 = (pos[0] - dec) ** 2 + ((pos[1] - ra) * np.cos(dec)) ** 2
        imap += amp * np.exp(-0.5 * r2 / bsigma**2)

    rho, kappa = matched_filter_depth1_map(
        imap, ivarmap, band_center=90 * u.GHz, beam_fwhm=fwhm
    )
    flux = rho / kappa
    snr = rho / np.sqrt(kappa)

    fconv = pixell_utils.dplanck(90e9) * 1e3  # mJy/sr per K
    omega = 2 * np.pi * bsigma**2
    for dec, ra, amp in sources:
        y, x = (
            int(round(v))
            for v in enmap.sky2pix(
                shape, wcs, [dec * pixell_utils.degree, ra * pixell_utils.degree]
            )
        )
        cut = snr[y - 3 : y + 4, x - 3 : x + 4]
        assert np.unravel_index(np.argmax(cut), cut.shape) == (3, 3)
        assert flux[y, x] == pytest.approx(amp * fconv * omega, rel=0.15)

    # Away from the sources the S/N is noise of approximately unit width.
    assert 0.3 < np.std(snr[20:120, 20:120]) < 1.5


@pytest.mark.parametrize("noise_only", [False, True])
def test_noisemask_noise_only_keeps_bright_source(noise_only):
    """
    A source above noisemask_lim is zeroed by the default noisemask.
    With noisemask_noise_only, it leaves only the noise model and is recovered.
    """
    fwhm = 2.2 * u.arcmin
    shape, wcs = enmap.geometry(
        pos=np.array([[-2, 2], [2, -2]]) * pixell_utils.degree,
        res=0.5 * pixell_utils.arcmin,
    )
    rng = np.random.default_rng(2)
    sigma_noise = 1e-5  # K per pixel
    ivarmap = enmap.ones(shape, wcs) / sigma_noise**2
    imap = enmap.enmap(rng.normal(0, sigma_noise, shape), wcs)
    amp = 0.5  # K, above the 0.1 K limit
    bsigma = fwhm.to_value(u.rad) * pixell_utils.fwhm
    pos = imap.posmap()
    imap += amp * np.exp(-0.5 * (pos[0] ** 2 + pos[1] ** 2) / bsigma**2)

    rho, kappa = matched_filter_depth1_map(
        imap,
        ivarmap,
        band_center=90 * u.GHz,
        beam_fwhm=fwhm,
        noisemask_lim=0.1,
        noisemask_noise_only=noise_only,
    )
    y, x = np.round(enmap.sky2pix(shape, wcs, [0.0, 0.0])).astype(int)
    fconv = pixell_utils.dplanck(90e9) * 1e3
    flux = rho[y, x] / kappa[y, x]
    if noise_only:
        assert flux == pytest.approx(amp * fconv * 2 * np.pi * bsigma**2, rel=0.15)
    else:
        assert abs(rho[y, x]) < 1e-3 * abs(amp * fconv * kappa[y, x])
