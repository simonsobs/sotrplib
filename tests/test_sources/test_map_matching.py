"""
Tests for grouping transient candidates across maps (map matching).
"""

from astropy import units as u
from astropy.time import Time, TimeDelta

from sotrplib.config.map_matching import MultiArrayMapMatcherConfig
from sotrplib.sifter.core import SifterResult
from sotrplib.sifter.map_matching import (
    EmptyMapMatcher,
    MapResult,
    MultiArrayMapMatcher,
)
from sotrplib.sources.sources import MeasuredSource

T0 = Time("2025-09-10T05:24:35")
HOUR = TimeDelta(3600, format="sec")


def _candidate(ra: float, dec: float) -> MeasuredSource:
    return MeasuredSource(ra=ra * u.deg, dec=dec * u.deg, flux=500 * u.mJy)


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
    assert groups[0].confirmed
    assert groups[0].arrays == {"i1", "i3"}
    assert a.map_match_id == b.map_match_id == groups[0].map_match_id
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
    assert not any(g.confirmed for g in groups)
    assert lone.map_match_id is not None
    assert lone.map_match_id != far.map_match_id
    for r in results:
        assert r.sifter_result.transient_candidates == []
        assert len(r.sifter_result.unconfirmed_transient_candidates) == 1


def test_same_array_in_two_bands_counts_once():
    results, groups = MultiArrayMapMatcher().match(
        [
            _result("c1_f220", "c1", [_candidate(10.0, -5.0)], frequency="f220"),
            _result("c1_f280", "c1", [_candidate(10.0, -5.0)], frequency="f280"),
        ]
    )

    assert len(groups) == 1
    assert groups[0].arrays == {"c1"}
    assert not groups[0].confirmed


def test_three_arrays_two_bands_form_one_group():
    maps = [
        _result("i1_f090", "i1", [_candidate(348.3478, 2.6759)]),
        _result("i3_f150", "i3", [_candidate(348.3473, 2.6766)], frequency="f150"),
        _result("c1_f220", "c1", [_candidate(348.3469, 2.6754)], frequency="f220"),
    ]
    _, groups = MultiArrayMapMatcher().match(maps)

    assert len(groups) == 1
    assert groups[0].arrays == {"i1", "i3", "c1"}
    assert len(groups[0].members) == 3


def test_different_observations_are_not_matched():
    _, groups = MultiArrayMapMatcher().match(
        [
            _result("i1", "i1", [_candidate(10.0, -5.0)], start=T0),
            _result("i3", "i3", [_candidate(10.0, -5.0)], start=T0 + 5 * HOUR),
        ]
    )

    assert len(groups) == 2
    assert not any(g.confirmed for g in groups)


def test_match_across_ra_wrap():
    _, groups = MultiArrayMapMatcher().match(
        [
            _result("i1", "i1", [_candidate(359.9999, 0.0)]),
            _result("i3", "i3", [_candidate(0.0001, 0.0)]),
        ]
    )

    assert len(groups) == 1
    assert groups[0].confirmed


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

    assert groups[0].confirmed
    assert len(results[0].sifter_result.transient_candidates) == 1


def test_empty_matcher_is_passthrough():
    candidate = _candidate(10.0, -5.0)
    results, groups = EmptyMapMatcher().match([_result("i1", "i1", [candidate])])

    assert groups == []
    assert results[0].sifter_result.transient_candidates == [candidate]
    assert candidate.map_match_id is None


def test_multi_array_config():
    matcher = MultiArrayMapMatcherConfig.model_validate(
        {"matcher_type": "multi_array", "radius": "1 arcmin", "min_arrays": 3}
    ).to_matcher()

    assert isinstance(matcher, MultiArrayMapMatcher)
    assert matcher.radius == 1.0 * u.arcmin
    assert matcher.min_arrays == 3
