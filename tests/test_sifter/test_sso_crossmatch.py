"""
Tests for solar-system-object trajectory crossmatching
(sotrplib.sifter.sso_crossmatch).
"""

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import ICRS
from astropy.time import Time

from sotrplib.sifter.sso_crossmatch import (
    crossmatch_sso_trajectories,
    sample_trajectory,
)
from sotrplib.sources.sources import MeasuredSource


class FakeSourceGenerator:
    """Minimal stand-in for socat.core.SourceGenerator: linear motion in
    RA at fixed Dec, valid over [t_min, t_max]."""

    def __init__(self, name, t_min, t_max, ra0, dec0, ra_rate_per_hour):
        self.source = type("FakeSource", (), {"name": name})()
        self.t_min = t_min
        self.t_max = t_max
        self._ra0 = ra0
        self._dec0 = dec0
        self._ra_rate = ra_rate_per_hour

    def at_time(self, t: Time):
        dt_hours = (t - self.t_min).sec / 3600.0
        ra = self._ra0 + self._ra_rate * dt_hours
        return ICRS(ra=ra * u.deg, dec=self._dec0 * u.deg), None


def make_candidate(ra_deg, dec_deg):
    return MeasuredSource(ra=ra_deg * u.deg, dec=dec_deg * u.deg, flux=10 * u.mJy)


def test_sample_trajectory_endpoints_and_values():
    t_min = Time("2025-01-01T00:00:00")
    t_max = t_min + 10 * u.hour
    sg = FakeSourceGenerator(
        "Test", t_min, t_max, ra0=10.0, dec0=5.0, ra_rate_per_hour=0.1
    )

    times, ra, dec = sample_trajectory(sg, t_min, t_max, cadence=2 * u.hour)

    assert times[0] == t_min
    assert abs((times[-1] - t_max).sec) < 1.0
    assert ra[0] == pytest.approx(np.radians(10.0))
    assert ra[-1] == pytest.approx(np.radians(11.0), abs=1e-6)
    assert np.all(dec == pytest.approx(np.radians(5.0)))


def test_crossmatch_sso_trajectories_matches_midpath_candidate():
    """A candidate sitting on the object's mid-window position is only
    findable by sampling the trajectory, not by checking t_min/t_max
    alone -- this is the whole point of the trajectory approach."""
    t_min = Time("2025-01-01T00:00:00")
    t_max = t_min + 10 * u.hour
    sg = FakeSourceGenerator(
        "Ceres", t_min, t_max, ra0=10.0, dec0=5.0, ra_rate_per_hour=0.1
    )
    candidate = make_candidate(ra_deg=10.5, dec_deg=5.0)

    result = crossmatch_sso_trajectories(
        candidates=[candidate],
        sso_generators={"Ceres": sg},
        t_min=t_min,
        t_max=t_max,
        radius=1.0 * u.arcmin,
        cadence=1 * u.hour,
    )

    assert result is not None
    assert candidate.source_type == "sso"
    assert candidate.crossmatches is not None
    assert len(candidate.crossmatches) == 1
    match = candidate.crossmatches[0]
    assert match.source_id == "Ceres"
    assert match.source_type == "sso"
    assert match.catalog_name == "socat"
    assert match.angular_separation < 1.0 * u.arcmin


def test_crossmatch_sso_trajectories_no_match_outside_radius():
    t_min = Time("2025-01-01T00:00:00")
    t_max = t_min + 10 * u.hour
    sg = FakeSourceGenerator(
        "Ceres", t_min, t_max, ra0=10.0, dec0=5.0, ra_rate_per_hour=0.1
    )
    # 30 arcmin from the object's entire path (dec=5.0 throughout).
    candidate = make_candidate(ra_deg=10.5, dec_deg=5.5)

    crossmatch_sso_trajectories(
        candidates=[candidate],
        sso_generators={"Ceres": sg},
        t_min=t_min,
        t_max=t_max,
        radius=1.0 * u.arcmin,
        cadence=1 * u.hour,
    )

    assert candidate.source_type is None
    assert candidate.crossmatches is None


def test_crossmatch_sso_trajectories_empty_inputs():
    t_min = Time("2025-01-01T00:00:00")
    t_max = t_min + 1 * u.hour

    assert crossmatch_sso_trajectories([], {}, t_min, t_max) == []

    candidate = make_candidate(10.0, 5.0)
    result = crossmatch_sso_trajectories([candidate], {}, t_min, t_max)
    assert result == [candidate]
    assert candidate.source_type is None
