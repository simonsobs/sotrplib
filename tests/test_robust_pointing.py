import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from mapcat.pointing.const import ConstantPointingModel

from sotrplib.config.pointing_residual import RobustResidualConfig
from sotrplib.maps.robust_pointing import (
    GaussianProcessPointingModel,
    RobustPointingOffset,
)


def _field(n=25, seed=0, ra0=0.0, true=(6.0, -4.0), noise_snr=(20, 60)):
    """Simulated sources on a field that crosses RA = 0."""
    rng = np.random.default_rng(seed)
    ra = (ra0 + rng.uniform(-30, 30, n)) % 360.0
    dec = rng.uniform(-8, 8, n)
    snr = rng.uniform(*noise_snr, n)
    sig = 0.6 * 120.0 / snr
    dra = true[0] + rng.normal(0, sig)
    ddec = true[1] + rng.normal(0, sig)
    return ra, dec, dra, ddec, snr, np.full(n, 120.0)


@pytest.mark.parametrize("method", ["constant", "gp"])
def test_recovers_constant_offset_with_outliers(method):
    ra, dec, dra, ddec, snr, fw = _field()
    # Noise fits with SNR just above the threshold, as seen in deep56.
    dra[:4] = [280.0, -240.0, 150.0, 30.0]
    ddec[:4] = [10.0, 245.0, -300.0, 200.0]
    snr[:4] = 6.0
    est = RobustPointingOffset(method=method)
    model, stats, used = est.fit_arrays(ra, dec, dra, ddec, snr, fw)
    assert not used[:4].any()
    assert model.ra_offset.to_value(u.arcsec) == pytest.approx(6.0, abs=1.5)
    assert model.dec_offset.to_value(u.arcsec) == pytest.approx(-4.0, abs=1.5)
    assert stats.n_sources == len(ra) - 4


def test_gp_reverts_to_constant_far_from_sources():
    ra, dec, dra, ddec, snr, fw = _field()
    est = RobustPointingOffset(method="gp")
    model, _, _ = est.fit_arrays(ra, dec, dra, ddec, snr, fw)
    assert isinstance(model, GaussianProcessPointingModel)
    far = SkyCoord(ra=180.0 * u.deg, dec=40.0 * u.deg)
    off_ra, off_dec = model.offsets_at(far)
    assert off_ra == model.ra_offset
    assert off_dec == model.dec_offset


def test_gp_correction_is_bounded():
    ra, dec, dra, ddec, snr, fw = _field()
    est = RobustPointingOffset(method="gp", max_correction=3 * u.arcsec)
    model, _, _ = est.fit_arrays(ra, dec, dra, ddec, snr, fw)
    grid = SkyCoord(
        ra=np.linspace(-40, 40, 200) % 360 * u.deg, dec=np.zeros(200) * u.deg
    )
    off_ra, _ = model.offsets_at(grid)
    corr = (off_ra - model.ra_offset).to_value(u.arcsec)
    assert np.all(np.abs(corr) <= 3.0 + 1e-9)


def test_too_few_sources_gives_zero():
    ra, dec, dra, ddec, snr, fw = _field(n=2)
    model, stats, _ = RobustPointingOffset().fit_arrays(ra, dec, dra, ddec, snr, fw)
    assert isinstance(model, ConstantPointingModel)
    assert model.ra_offset == 0 * u.deg
    assert stats.n_sources == 2


def test_offset_gate_and_model_limit():
    ra, dec, dra, ddec, snr, fw = _field(true=(400.0, 0.0))
    # All offsets are above the 5 arcmin gate: no sources remain.
    model, _, used = RobustPointingOffset().fit_arrays(ra, dec, dra, ddec, snr, fw)
    assert not used.any()
    assert model.ra_offset == 0 * u.deg
    # With a wide gate, the model limit rejects the large offset.
    est = RobustPointingOffset(max_offset=10 * u.arcmin, max_model_offset=5 * u.arcmin)
    model, _, _ = est.fit_arrays(ra, dec, dra, ddec, snr, fw)
    assert model.ra_offset == 0 * u.deg


def test_config_and_serialisation():
    est = RobustResidualConfig(pointing_residual_type="robust_gp").to_residuals()
    ra, dec, dra, ddec, snr, fw = _field()
    model, _, _ = est.fit_arrays(ra, dec, dra, ddec, snr, fw)
    again = GaussianProcessPointingModel.model_validate_json(model.model_dump_json())
    pos = SkyCoord(ra=[359.0, 1.0] * u.deg, dec=[0.0, 2.0] * u.deg)
    a = model.predict(pos)
    b = again.predict(pos)
    assert np.allclose(a.ra.deg, b.ra.deg) and np.allclose(a.dec.deg, b.dec.deg)
