"""
Tests for SOCat.update: add the confirmed map-match groups of a run to socat
as monitored sources.
"""

import json
import os
import pickle

import pytest
from astropy import units as u
from astropy.time import Time

from sotrplib.config.source_catalog import SOCatConfig
from sotrplib.handlers.basic import PipelineRunner
from sotrplib.sifter.core import SifterResult
from sotrplib.sifter.map_matching import MapResult, MultiArrayMapMatcher
from sotrplib.source_catalog.core import RegisteredSourceCatalog
from sotrplib.source_catalog.socat import SOCat, transient_name
from sotrplib.sources.sources import MeasuredSource

T0 = Time("2025-09-10T05:24:35")


@pytest.fixture
def empty_socat(tmp_path, monkeypatch):
    """An empty socat pickle catalog, used through the socat env settings."""
    location = tmp_path / "socat.pkl"
    monkeypatch.setenv("socat_client_client_type", "pickle")
    monkeypatch.setenv("socat_client_pickle_path", str(location))

    from socat.client.mock import Client

    with open(location, "wb") as f:
        pickle.dump(Client(), f)
    yield location
    os.environ.pop("socat_client_pickle_path", None)


def _candidate(ra: float, dec: float, snr: float, flux_mjy: float) -> MeasuredSource:
    return MeasuredSource(
        ra=ra * u.deg,
        dec=dec * u.deg,
        flux=flux_mjy * u.mJy,
        snr=snr,
        observation_mean_time=T0,
    )


def _result(name: str, array: str, candidates: list[MeasuredSource]) -> MapResult:
    return MapResult(
        map_name=name,
        mapcat_id=None,
        array=array,
        frequency="f090",
        observation_start=T0,
        forced_photometry_candidates=[],
        sifter_result=SifterResult(
            source_candidates=[], transient_candidates=candidates, noise_candidates=[]
        ),
    )


def _summary():
    """One confirmed group (i1 + i3) and one unconfirmed single (i5)."""
    a = _candidate(348.3478, 2.6759, snr=12.0, flux_mjy=800.0)
    b = _candidate(348.3476, 2.6760, snr=6.0, flux_mjy=350.0)
    lone = _candidate(7.7443, -9.5003, snr=9.0, flux_mjy=100.0)
    results = [
        _result("i1", "i1", [a]),
        _result("i3", "i3", [b]),
        _result("i5", "i5", [lone]),
    ]
    _, summary = MultiArrayMapMatcher().match(results)
    return summary, (a, b), lone


def test_transient_name():
    assert transient_name("SOT", 348.3478 * u.deg, 2.6759 * u.deg) == (
        "SOT J231323.5+024033"
    )
    assert transient_name("SOT", 7.7443 * u.deg, -9.5003 * u.deg).startswith(
        "SOT J003058.6-0930"
    )


def test_update_adds_only_confirmed_groups(empty_socat, tmp_path):
    catalog = SOCat(add_confirmed_transients=True, record_directory=tmp_path / "rec")
    summary, members, lone = _summary()

    added = catalog.update(summary)

    assert len(added) == 1
    new = added[0]
    assert new.source_id == "SOT J231323.5+024033"
    assert new.flux.to_value(u.mJy) == pytest.approx(800.0)

    fixed = catalog.catalog.get_box_fixed(
        lower_left=_icrs(0, -89.9), upper_right=_icrs(359.9, 89.9)
    )
    assert len(fixed) == 1
    assert fixed[0].monitored
    assert fixed[0].name == new.source_id

    socat_id = new.crossmatches[0].catalog_idx
    assert socat_id == fixed[0].source_id
    for candidate in members:
        assert candidate.crossmatches[-1].catalog_idx == socat_id
        assert candidate.crossmatches[-1].catalog_name == "socat"
    assert not lone.crossmatches

    records = list((tmp_path / "rec").glob("socat_update_*.json"))
    assert len(records) == 1
    record = json.loads(records[0].read_text())
    assert [g["action"] for g in record["groups"]] == ["added"]
    assert record["groups"][0]["confirmed"] is True
    assert record["groups"][0]["socat_id"] == str(socat_id)


def test_update_monitors_a_known_unmonitored_source(empty_socat, tmp_path):
    """A confirmed group on a catalog source that is not monitored (e.g. a
    flaring star) sets the source to monitored and reports it."""
    catalog = SOCat(add_confirmed_transients=True, record_directory=tmp_path)
    known = catalog.catalog.create_source(
        position=_icrs(348.3477, 2.6760),
        flux=50 * u.mJy,
        name="ACT-S J2313.3+0240",
        flags={"monitored": False},
    )
    summary, _, _ = _summary()

    reported = catalog.update(summary)

    assert [s.source_id for s in reported] == ["ACT-S J2313.3+0240"]
    assert reported[0].crossmatches[0].catalog_idx == known.source_id
    assert catalog.catalog.get_source(source_id=known.source_id).monitored
    fixed = catalog.catalog.get_box_fixed(
        lower_left=_icrs(0, -89.9), upper_right=_icrs(359.9, 89.9)
    )
    assert len(fixed) == 1  # no new source

    # A rerun reports the source as already in the catalog, in a new file.
    summary, _, _ = _summary()
    assert catalog.update(summary) == []
    records = sorted(tmp_path.glob("socat_update_*.json"))
    assert len(records) == 2
    actions = [json.loads(r.read_text())["groups"][0]["action"] for r in records]
    assert sorted(actions) == ["already_in_catalog", "monitored"]
    assert all(
        json.loads(r.read_text())["groups"][0]["socat_id"] == str(known.source_id)
        for r in records
    )


def test_update_does_not_add_the_same_event_again(empty_socat):
    catalog = SOCat(add_confirmed_transients=True)
    summary, _, _ = _summary()
    assert len(catalog.update(summary)) == 1

    summary, _, _ = _summary()
    assert catalog.update(summary) == []

    fixed = catalog.catalog.get_box_fixed(
        lower_left=_icrs(0, -89.9), upper_right=_icrs(359.9, 89.9)
    )
    assert len(fixed) == 1


def test_update_is_off_by_default(empty_socat):
    catalog = SOCat()
    summary, _, _ = _summary()
    assert catalog.update(summary) == []


def test_base_catalog_update_does_nothing():
    summary, _, _ = _summary()
    assert RegisteredSourceCatalog(sources=[]).update(summary) == []


def test_socat_config_update_block(empty_socat, tmp_path):
    config = SOCatConfig.model_validate(
        {
            "catalog_type": "socat",
            "update": {
                "duplicate_radius": "30 arcsec",
                "name_prefix": "MOCK",
                "record_directory": str(tmp_path),
            },
        }
    )
    catalog = config.to_source_catalog()
    assert catalog.add_confirmed_transients
    assert catalog.duplicate_radius == 30 * u.arcsec
    assert catalog.name_prefix == "MOCK"
    assert catalog.record_directory == tmp_path

    assert not SOCatConfig().to_source_catalog().add_confirmed_transients


def test_runner_calls_update_after_map_matching(empty_socat):
    catalog = SOCat(add_confirmed_transients=True)
    runner = PipelineRunner(
        map_coadder=None,
        source_simulators=None,
        source_injector=None,
        source_catalogs=[catalog],
        preprocessors=None,
        pointing_provider=None,
        pointing_residual_model=None,
        postprocessors=None,
        forced_photometry=None,
        source_subtractor=None,
        blind_search=None,
        sifter=None,
        source_outputs=None,
        map_outputs=None,
    )
    summary, _, _ = _summary()
    runner.update_source_catalogs([], summary)

    fixed = catalog.catalog.get_box_fixed(
        lower_left=_icrs(0, -89.9), upper_right=_icrs(359.9, 89.9)
    )
    assert len(fixed) == 1


def _icrs(ra: float, dec: float):
    from astropy.coordinates import ICRS

    return ICRS(ra=ra * u.deg, dec=dec * u.deg)
