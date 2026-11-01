import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord

from sotrplib.maps.parametric_pointing import (
    ParametricPointingModel,
    ParametricPointingOffset,
)


def _field(n=60, seed=0, grad=(8.0, -4.0), drift=0.0):
    """Sources on a 60 deg field across RA = 0. Offsets have a constant part,
    an RA gradient (arcsec per 30 deg), and a drift in time (arcsec/hour)."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-30, 30, n)
    ra = x % 360.0
    dec = rng.uniform(-8, 8, n)
    t = (x + 30) / 60 * 3 * 3600.0  # 3 h scan, RA ~ time
    snr = rng.uniform(15, 60, n)
    sig = 0.6 * 120.0 / snr
    dra = 5.0 + grad[0] * x / 30 + drift * t / 3600 + rng.normal(0, sig)
    ddec = -3.0 + grad[1] * x / 30 + rng.normal(0, sig)
    return ra, dec, dra, ddec, snr, np.full(n, 120.0), t, x


def test_tangent_model_recovers_gradient_and_rejects_outliers():
    ra, dec, dra, ddec, snr, fw, t, x = _field()
    dra[:3] = [250.0, -200.0, 120.0]
    snr[:3] = 6.0
    model, stats, used = ParametricPointingOffset("tangent").fit_arrays(
        ra, dec, dra, ddec, snr, fw
    )
    assert not used[:3].any()
    assert model.order >= 1
    pos = SkyCoord(ra=[350.0, 10.0] * u.deg, dec=[0.0, 0.0] * u.deg)
    off_ra, off_dec = model.offsets_at(pos)
    assert off_ra == pytest.approx([5.0 - 8.0 / 3, 5.0 + 8.0 / 3], abs=1.5)
    assert off_dec == pytest.approx([-3.0 + 4.0 / 3, -3.0 - 4.0 / 3], abs=1.5)


def test_tangent_model_does_not_extrapolate():
    ra, dec, dra, ddec, snr, fw, t, x = _field()
    model, _, _ = ParametricPointingOffset("tangent").fit_arrays(
        ra, dec, dra, ddec, snr, fw
    )
    edge = model.offsets_at(SkyCoord(ra=30.0 * u.deg, dec=0.0 * u.deg))[0]
    far = model.offsets_at(SkyCoord(ra=80.0 * u.deg, dec=0.0 * u.deg))[0]
    assert abs(far[0] - edge[0]) < 3.0


def test_sparse_map_falls_back_to_constant():
    ra, dec, dra, ddec, snr, fw, t, x = _field(n=5)
    model, _, _ = ParametricPointingOffset("tangent").fit_arrays(
        ra, dec, dra, ddec, snr, fw
    )
    assert model.order == 0
    assert len(model.coeffs_ra) == 1


def test_time_model_recovers_drift():
    ra, dec, dra, ddec, snr, fw, t, x = _field(grad=(0.0, 0.0), drift=6.0)
    model, _, _ = ParametricPointingOffset("time").fit_arrays(
        ra, dec, dra, ddec, snr, fw, times=t
    )
    assert len(model.nodes) >= 2
    off_ra, _ = model.offsets_at(times=np.array([0.0, 3 * 3600.0]))
    assert off_ra == pytest.approx([5.0, 5.0 + 18.0], abs=2.0)


def test_tangent_model_stays_bounded_away_from_a_line_of_sources():
    # Sources on a narrow diagonal strip (as on the Galactic plane), plus a
    # few far away. The model must not swing far outside the measured range
    # in the empty corners of the box.
    rng = np.random.default_rng(3)
    s = rng.uniform(-30, 30, 80)
    ra = (s + rng.normal(0, 0.5, 80)) % 360
    dec = 0.6 * s + rng.normal(0, 0.5, 80)
    ra = np.r_[ra, [25.0, 335.0]]
    dec = np.r_[dec, [-20.0, 20.0]]
    snr = rng.uniform(15, 60, 82)
    dra = 5.0 + 6.0 * np.r_[s, 0, 0] / 30 + rng.normal(0, 0.6 * 120 / snr)
    ddec = -2.0 + rng.normal(0, 0.6 * 120 / snr)
    fw = np.full(82, 120.0)
    model, _, _ = ParametricPointingOffset("tangent").fit_arrays(
        ra, dec, dra, ddec, snr, fw
    )
    corners = SkyCoord(
        ra=[25.0, 335.0, 25.0, 335.0] * u.deg, dec=[18.0, -18.0, -18.0, 18.0] * u.deg
    )
    off_ra, off_dec = model.offsets_at(corners)
    assert np.all((off_ra > -10) & (off_ra < 20))
    assert np.all(np.abs(off_dec + 2.0) < 10)


def test_time_model_with_too_few_sources_gives_zero():
    ra, dec, dra, ddec, snr, fw, t, x = _field(n=2)
    model, _, _ = ParametricPointingOffset("time").fit_arrays(
        ra, dec, dra, ddec, snr, fw, times=t
    )
    off_ra, off_dec = model.offsets_at(times=np.array([100.0]))
    assert off_ra[0] == 0.0 and off_dec[0] == 0.0


def test_directions_and_cos_dec():
    model = ParametricPointingModel(
        coordinate="tangent", coeffs_ra=[36.0], coeffs_dec=[18.0]
    )
    pos = SkyCoord(ra=10.0 * u.deg, dec=-60.0 * u.deg)
    in_map = model.sky_to_map(pos)
    # 36 arcsec on the sky at Dec -60 is 72 arcsec of RA.
    assert (pos.ra - in_map.ra).to_value(u.arcsec) == pytest.approx(72.0, rel=1e-6)
    assert (pos.dec - in_map.dec).to_value(u.arcsec) == pytest.approx(18.0, rel=1e-6)
    back = model.map_to_sky(in_map)
    assert back.separation(pos).to_value(u.arcsec) < 0.1


def test_round_trip_json_is_small():
    ra, dec, dra, ddec, snr, fw, t, x = _field()
    model, _, _ = ParametricPointingOffset("tangent").fit_arrays(
        ra, dec, dra, ddec, snr, fw
    )
    text = model.model_dump_json()
    again = ParametricPointingModel.model_validate_json(text)
    pos = SkyCoord(ra=[355.0, 5.0] * u.deg, dec=[1.0, -1.0] * u.deg)
    assert np.allclose(model.offsets_at(pos), again.offsets_at(pos))
    assert len(text) < 1000
