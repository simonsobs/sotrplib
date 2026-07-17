"""
Tests for DefaultSifter's SSO-trajectory-crossmatch wiring: aggregating
sso generators across catalogs, reclassifying matched candidates out of
noise/transient into source_candidates, and the no-observation-time guard.
"""

import astropy.units as u
import pytest
from astropy.coordinates import ICRS
from astropy.time import Time

from sotrplib.sifter.core import DefaultSifter
from sotrplib.sources.sources import MeasuredSource


class FakeMap:
    def __init__(self, observation_start, observation_end):
        self.observation_start = observation_start
        self.observation_end = observation_end
        self.map_id = "fake-map"


class FakeSourceGenerator:
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


class FakeSSOCatalog:
    """Only implements get_sso_generators_in_map, matching how SOCat is
    duck-typed in DefaultSifter._crossmatch_sso_trajectories."""

    def __init__(self, generators: dict):
        self._generators = generators

    def get_sso_generators_in_map(self, input_map):
        return self._generators


class CatalogWithoutSSO:
    """Represents a catalog that doesn't implement SSO trajectories at
    all (e.g. a plain fixed-source catalog) -- must be skipped, not
    crash on a missing attribute."""

    pass


def make_candidate(ra_deg, dec_deg):
    return MeasuredSource(ra=ra_deg * u.deg, dec=dec_deg * u.deg, flux=10 * u.mJy)


@pytest.fixture
def week_window():
    t_min = Time("2025-01-01T00:00:00")
    t_max = t_min + 7 * u.day
    return t_min, t_max


def test_matched_candidate_moves_to_source_candidates(week_window):
    t_min, t_max = week_window
    sg = FakeSourceGenerator(
        "Ceres", t_min, t_max, ra0=10.0, dec0=5.0, ra_rate_per_hour=0.01
    )
    on_path_candidate = make_candidate(ra_deg=10.5, dec_deg=5.0)  # ~mid-path
    off_path_candidate = make_candidate(ra_deg=50.0, dec_deg=-20.0)

    sifter = DefaultSifter(crossmatch_sso_trajectories=True)
    remaining_noise, remaining_transient, sso_matched = (
        sifter._crossmatch_sso_trajectories(
            noise_candidates=[on_path_candidate],
            transient_candidates=[off_path_candidate],
            catalogs=[FakeSSOCatalog({"Ceres": sg}), CatalogWithoutSSO()],
            input_map=FakeMap(t_min, t_max),
        )
    )

    assert remaining_noise == []
    assert remaining_transient == [off_path_candidate]
    assert sso_matched == [on_path_candidate]
    assert on_path_candidate.source_type == "sso"
    assert off_path_candidate.source_type is None


def test_no_generators_leaves_buckets_unchanged(week_window):
    t_min, t_max = week_window
    noise = [make_candidate(10.0, 5.0)]
    transient = [make_candidate(20.0, -5.0)]

    sifter = DefaultSifter(crossmatch_sso_trajectories=True)
    remaining_noise, remaining_transient, sso_matched = (
        sifter._crossmatch_sso_trajectories(
            noise_candidates=noise,
            transient_candidates=transient,
            catalogs=[FakeSSOCatalog({}), CatalogWithoutSSO()],
            input_map=FakeMap(t_min, t_max),
        )
    )

    assert remaining_noise == noise
    assert remaining_transient == transient
    assert sso_matched == []


def test_no_observation_time_is_a_noop():
    sifter = DefaultSifter(crossmatch_sso_trajectories=True)
    noise = [make_candidate(10.0, 5.0)]
    transient = [make_candidate(20.0, -5.0)]

    remaining_noise, remaining_transient, sso_matched = (
        sifter._crossmatch_sso_trajectories(
            noise_candidates=noise,
            transient_candidates=transient,
            catalogs=[FakeSSOCatalog({"Ceres": object()})],
            input_map=FakeMap(None, None),
        )
    )

    assert remaining_noise == noise
    assert remaining_transient == transient
    assert sso_matched == []


def test_aggregates_generators_across_multiple_catalogs(week_window):
    t_min, t_max = week_window
    ceres = FakeSourceGenerator(
        "Ceres", t_min, t_max, ra0=10.0, dec0=5.0, ra_rate_per_hour=0.01
    )
    pallas = FakeSourceGenerator(
        "Pallas", t_min, t_max, ra0=100.0, dec0=-30.0, ra_rate_per_hour=0.01
    )
    near_ceres = make_candidate(ra_deg=10.5, dec_deg=5.0)
    near_pallas = make_candidate(ra_deg=100.5, dec_deg=-30.0)

    sifter = DefaultSifter(crossmatch_sso_trajectories=True)
    remaining_noise, remaining_transient, sso_matched = (
        sifter._crossmatch_sso_trajectories(
            noise_candidates=[near_ceres, near_pallas],
            transient_candidates=[],
            catalogs=[
                FakeSSOCatalog({"Ceres": ceres}),
                FakeSSOCatalog({"Pallas": pallas}),
            ],
            input_map=FakeMap(t_min, t_max),
        )
    )

    assert remaining_noise == []
    assert remaining_transient == []
    assert {c.crossmatches[0].source_id for c in sso_matched} == {"Ceres", "Pallas"}
