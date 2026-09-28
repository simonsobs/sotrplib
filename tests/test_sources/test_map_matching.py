"""
Tests for grouping transient candidates across maps (map matching).
"""

import json

import pytest
from astropy import units as u
from astropy.time import Time, TimeDelta

from sotrplib.config.map_matching import MultiArrayMapMatcherConfig
from sotrplib.sifter.core import SifterResult
from sotrplib.sifter.map_matching import (
    EmptyMapMatcher,
    MapResult,
    MultiArrayMapMatcher,
    map_match_significance,
)
from sotrplib.sources.sources import MeasuredSource

T0 = Time("2025-09-10T05:24:35")
HOUR = TimeDelta(3600, format="sec")


def _candidate(ra: float, dec: float, snr: float = 10.0) -> MeasuredSource:
    return MeasuredSource(ra=ra * u.deg, dec=dec * u.deg, flux=500 * u.mJy, snr=snr)


def _result(
    name: str,
    array: str,
    candidates: list[MeasuredSource],
    start: Time = T0,
    frequency: str = "f090",
) -> MapResult:
    return MapResult(
        map_name=name,
        mapcat_id=None,
        array=array,
        frequency=frequency,
        observation_start=start,
        observation_end=start + 3 * HOUR,
        forced_photometry_candidates=[],
        sifter_result=SifterResult(
            source_candidates=[],
            transient_candidates=candidates,
            noise_candidates=[],
        ),
    )


def test_detection_in_two_arrays_is_confirmed():
    a, b = _candidate(348.3478, 2.6759), _candidate(348.3476, 2.6760)
    results, groups = MultiArrayMapMatcher().match(
        [_result("i1", "i1", [a]), _result("i3", "i3", [b])]
    )

    assert len(groups) == 1
    match = groups[0].match
    assert match.confirmed
    assert (match.n_maps, match.n_arrays, match.n_bands) == (2, 2, 1)
    assert a.map_match is match and b.map_match is match
    for r in results:
        assert len(r.sifter_result.transient_candidates) == 1
        assert r.sifter_result.unconfirmed_transient_candidates == []


def test_single_array_detection_is_unconfirmed():
    lone = _candidate(7.7443, -9.5003)
    far = _candidate(348.3478, 2.6759)
    results, groups = MultiArrayMapMatcher().match(
        [_result("i3", "i3", [lone]), _result("i1", "i1", [far])]
    )

    assert len(groups) == 2
    assert not any(g.match.confirmed for g in groups)
    assert lone.map_match is not None
    assert lone.map_match.match_id != far.map_match.match_id
    for r in results:
        assert r.sifter_result.transient_candidates == []
        assert len(r.sifter_result.unconfirmed_transient_candidates) == 1


def test_same_array_in_two_bands_counts_once():
    _, groups = MultiArrayMapMatcher().match(
        [
            _result("c1_f220", "c1", [_candidate(10.0, -5.0)], frequency="f220"),
            _result("c1_f280", "c1", [_candidate(10.0, -5.0)], frequency="f280"),
        ]
    )

    assert len(groups) == 1
    assert (groups[0].match.n_arrays, groups[0].match.n_bands) == (1, 2)
    assert not groups[0].match.confirmed


def test_three_arrays_three_bands_form_one_group():
    maps = [
        _result("i1_f090", "i1", [_candidate(348.3478, 2.6759)]),
        _result("i3_f150", "i3", [_candidate(348.3473, 2.6766)], frequency="f150"),
        _result("c1_f220", "c1", [_candidate(348.3469, 2.6754)], frequency="f220"),
    ]
    _, groups = MultiArrayMapMatcher().match(maps)

    assert len(groups) == 1
    assert groups[0].to_dict()["arrays"] == ["c1", "i1", "i3"]
    assert groups[0].match.n_bands == 3
    assert len(groups[0].members) == 3


def test_different_observations_are_not_matched():
    _, groups = MultiArrayMapMatcher().match(
        [
            _result("i1", "i1", [_candidate(10.0, -5.0)], start=T0),
            _result("i3", "i3", [_candidate(10.0, -5.0)], start=T0 + 5 * HOUR),
        ]
    )

    assert len(groups) == 2
    assert not any(g.match.confirmed for g in groups)


def test_match_across_ra_wrap():
    _, groups = MultiArrayMapMatcher().match(
        [
            _result("i1", "i1", [_candidate(359.9999, 0.0)]),
            _result("i3", "i3", [_candidate(0.0001, 0.0)]),
        ]
    )

    assert len(groups) == 1
    assert groups[0].match.confirmed
    # the mean position must not average RA 359.9999 and 0.0001 to 180
    assert groups[0].dec.to_value(u.deg) == pytest.approx(0.0, abs=1e-6)
    assert (
        min(
            groups[0].ra.to_value(u.deg) % 360, 360 - groups[0].ra.to_value(u.deg) % 360
        )
        < 1e-3
    )


def test_outside_radius_is_not_matched():
    _, groups = MultiArrayMapMatcher(radius=1.0 * u.arcmin).match(
        [
            _result("i1", "i1", [_candidate(10.0, 0.0)]),
            _result("i3", "i3", [_candidate(10.0, 2.0 / 60)]),
        ]
    )

    assert len(groups) == 2


def test_min_arrays_one_keeps_single_detections():
    results, groups = MultiArrayMapMatcher(min_arrays=1).match(
        [_result("i1", "i1", [_candidate(10.0, -5.0)])]
    )

    assert groups[0].match.confirmed
    assert len(results[0].sifter_result.transient_candidates) == 1


def test_empty_matcher_is_passthrough():
    candidate = _candidate(10.0, -5.0)
    results, groups = EmptyMapMatcher().match([_result("i1", "i1", [candidate])])

    assert groups == []
    assert results[0].sifter_result.transient_candidates == [candidate]
    assert candidate.map_match is None


@pytest.mark.parametrize(
    "snrs, combined, significance",
    [
        ([11.0], 11.0, 10.9),  # one map: ~ the two-sided-equivalent SNR
        ([5.0, 5.0], 7.07, 6.7),  # two modest detections beat either alone
        ([6.0, 6.0], 8.49, 8.1),
    ],
)
def test_significance_values(snrs, combined, significance):
    c, s = map_match_significance(snrs)
    assert c == pytest.approx(combined, abs=0.01)
    assert s == pytest.approx(significance, abs=0.1)


def test_significance_grows_with_maps_and_stays_finite():
    one = map_match_significance([8.0])[1]
    two = map_match_significance([8.0, 8.0])[1]
    six = map_match_significance([64.8, 56.7, 52.8, 51.5, 25.7, 9.7])[1]

    assert one < two < six
    assert six < float("inf")
    assert map_match_significance([]) == (0.0, 0.0)
    assert map_match_significance([None, 5.0]) == map_match_significance([5.0])


def test_confirmed_groups_rank_above_brighter_unconfirmed():
    bright_single = _candidate(50.0, 0.0, snr=40.0)
    pair_a, pair_b = _candidate(10.0, 0.0, snr=6.0), _candidate(10.0, 0.0, snr=6.0)
    strong_a, strong_b = (
        _candidate(20.0, 0.0, snr=12.0),
        _candidate(20.0, 0.0, snr=12.0),
    )
    faint_single = _candidate(30.0, 0.0, snr=5.0)
    _, groups = MultiArrayMapMatcher().match(
        [
            _result("i1", "i1", [bright_single, pair_a, strong_a]),
            _result("i3", "i3", [pair_b, strong_b, faint_single]),
        ]
    )

    ranked = sorted(groups, key=lambda g: g.match.rank)
    assert [g.match.rank for g in ranked] == [1, 2, 3, 4]
    assert [g.match.confirmed for g in ranked] == [True, True, False, False]
    # within each tier, by significance
    assert strong_a.map_match.rank == 1
    assert pair_a.map_match.rank == 2
    assert bright_single.map_match.rank == 3
    assert faint_single.map_match.rank == 4


def test_summary_written(tmp_path):
    a, b = _candidate(348.3478, 2.6759, snr=64.8), _candidate(348.3476, 2.676, snr=56.7)
    lone = _candidate(7.7443, -9.5003, snr=11.0)
    MultiArrayMapMatcher(summary_directory=tmp_path).match(
        [_result("f090_i1", "i1", [a, lone]), _result("f090_i3", "i3", [b])]
    )

    (path,) = tmp_path.glob("map_match_summary_*.json")
    summary = json.loads(path.read_text())
    assert summary["min_arrays"] == 2
    assert [g["rank"] for g in summary["groups"]] == [1, 2]
    top = summary["groups"][0]
    assert top["confirmed"] and top["arrays"] == ["i1", "i3"]
    assert {m["map_name"] for m in top["members"]} == {"f090_i1", "f090_i3"}
    assert top["ra_deg"] == pytest.approx(348.3477, abs=1e-3)
    assert not summary["groups"][1]["confirmed"]


def test_multi_array_config(tmp_path):
    matcher = MultiArrayMapMatcherConfig.model_validate(
        {
            "matcher_type": "multi_array",
            "radius": "1 arcmin",
            "min_arrays": 3,
            "summary_directory": str(tmp_path),
        }
    ).to_matcher()

    assert isinstance(matcher, MultiArrayMapMatcher)
    assert matcher.radius == 1.0 * u.arcmin
    assert matcher.min_arrays == 3
    assert matcher.summary_directory == tmp_path
