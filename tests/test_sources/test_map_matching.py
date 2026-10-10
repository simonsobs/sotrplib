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
    group = groups[0]
    assert group.confirmed
    assert (group.n_maps, group.n_arrays, group.n_bands) == (2, 2, 1)
    assert a.group_id == b.group_id == group.group_id
    assert a.group_rank == b.group_rank == 1
    for r in results:
        assert len(r.sifter_result.transient_candidates) == 1
        assert r.sifter_result.noise_candidates == []


def test_single_array_detection_is_unconfirmed():
    lone = _candidate(7.7443, -9.5003)
    far = _candidate(348.3478, 2.6759)
    results, groups = MultiArrayMapMatcher().match(
        [_result("i3", "i3", [lone]), _result("i1", "i1", [far])]
    )

    assert len(groups) == 2
    assert not any(g.confirmed for g in groups)
    # groups that are not confirmed keep their group_id in the noise
    assert lone.group_id is not None
    assert lone.group_id != far.group_id
    for r in results:
        assert r.sifter_result.transient_candidates == []
        assert len(r.sifter_result.noise_candidates) == 1


def test_same_array_in_two_bands_counts_once():
    _, groups = MultiArrayMapMatcher().match(
        [
            _result("c1_f220", "c1", [_candidate(10.0, -5.0)], frequency="f220"),
            _result("c1_f280", "c1", [_candidate(10.0, -5.0)], frequency="f280"),
        ]
    )

    assert len(groups) == 1
    assert (groups[0].n_arrays, groups[0].n_bands) == (1, 2)
    assert not groups[0].confirmed


def test_three_arrays_three_bands_form_one_group():
    maps = [
        _result("i1_f090", "i1", [_candidate(348.3478, 2.6759)]),
        _result("i3_f150", "i3", [_candidate(348.3473, 2.6766)], frequency="f150"),
        _result("c1_f220", "c1", [_candidate(348.3469, 2.6754)], frequency="f220"),
    ]
    _, groups = MultiArrayMapMatcher().match(maps)

    assert len(groups) == 1
    assert groups[0].to_dict()["arrays"] == ["c1", "i1", "i3"]
    assert groups[0].n_bands == 3
    assert len(groups[0].members) == 3


def test_maps_at_different_times_are_matched():
    # No time cut: the maps of the run set the time range.
    _, groups = MultiArrayMapMatcher().match(
        [
            _result("i1", "i1", [_candidate(10.0, -5.0)], start=T0),
            _result("i3", "i3", [_candidate(10.0, -5.0)], start=T0 + 30 * 24 * HOUR),
        ]
    )

    assert len(groups) == 1
    assert groups[0].confirmed


def test_candidates_of_one_map_are_not_matched():
    a, b = _candidate(10.0, -5.0), _candidate(10.0, -5.0 + 0.5 / 60)
    _, groups = MultiArrayMapMatcher(min_arrays=1).match([_result("i1", "i1", [a, b])])

    assert len(groups) == 2
    assert a.group_id != b.group_id


def test_match_across_ra_wrap():
    _, groups = MultiArrayMapMatcher().match(
        [
            _result("i1", "i1", [_candidate(359.9999, 0.0)]),
            _result("i3", "i3", [_candidate(0.0001, 0.0)]),
        ]
    )

    assert len(groups) == 1
    assert groups[0].confirmed
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

    assert groups[0].confirmed
    assert len(results[0].sifter_result.transient_candidates) == 1


def test_empty_matcher_is_passthrough():
    candidate = _candidate(10.0, -5.0)
    results, groups = EmptyMapMatcher().match([_result("i1", "i1", [candidate])])

    assert groups == []
    assert results[0].sifter_result.transient_candidates == [candidate]
    assert candidate.group_id is None


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

    ranked = sorted(groups, key=lambda g: g.rank)
    assert [g.rank for g in ranked] == [1, 2, 3, 4]
    assert [g.confirmed for g in ranked] == [True, True, False, False]
    # within each tier, by significance
    assert strong_a.group_rank == 1
    assert pair_a.group_rank == 2
    assert bright_single.group_rank == 3
    assert faint_single.group_rank == 4


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
    assert top["group_id"] == str(a.group_id)
    assert {m["measurement_id"] for m in top["members"]} == {
        str(a.measurement_id),
        str(b.measurement_id),
    }
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


def test_low_sig_detection_confirms_high_sig_seed():
    seed, support = _candidate(10.0, 0.0, snr=6.0), _candidate(10.0, 0.0, snr=3.5)
    results, groups = MultiArrayMapMatcher(high_sig=5.0, low_sig=3.0).match(
        [_result("i1", "i1", [seed]), _result("i3", "i3", [support])]
    )

    (group,) = groups
    assert group.confirmed
    assert group.max_snr == 6.0
    assert results[1].sifter_result.transient_candidates == [support]


def test_low_sig_only_group_is_not_confirmed():
    a, b = _candidate(10.0, 0.0, snr=4.5), _candidate(10.0, 0.0, snr=4.5)
    results, groups = MultiArrayMapMatcher(high_sig=5.0, low_sig=3.0).match(
        [_result("i1", "i1", [a]), _result("i3", "i3", [b])]
    )

    (group,) = groups
    assert group.n_arrays == 2
    assert not group.confirmed
    for r in results:
        assert r.sifter_result.transient_candidates == []
        assert len(r.sifter_result.noise_candidates) == 1
    assert a.group_id == b.group_id == group.group_id


def test_candidates_below_low_sig_are_not_matched():
    seed, faint = _candidate(10.0, 0.0, snr=6.0), _candidate(10.0, 0.0, snr=2.5)
    results, groups = MultiArrayMapMatcher(high_sig=5.0, low_sig=3.0).match(
        [_result("i1", "i1", [seed]), _result("i3", "i3", [faint])]
    )

    (group,) = groups
    assert not group.confirmed
    assert faint.group_id is None
    assert results[1].sifter_result.noise_candidates == [faint]


def test_notable_groups_rank_above_low_sig_singles():
    confirmed = [_candidate(10.0, 0.0, snr=4.0), _candidate(10.0, 0.0, snr=5.5)]
    seed_single = _candidate(20.0, 0.0, snr=5.2)
    low_pair = [_candidate(30.0, 0.0, snr=4.0), _candidate(30.0, 0.0, snr=4.0)]
    low_single = _candidate(40.0, 0.0, snr=4.9)
    _, groups = MultiArrayMapMatcher(high_sig=5.0, low_sig=3.0).match(
        [
            _result("i1", "i1", [confirmed[0], seed_single, low_pair[0], low_single]),
            _result("i3", "i3", [confirmed[1], low_pair[1]]),
        ]
    )

    assert confirmed[0].group_rank == 1
    # seed_single (5.2) and low_pair (2 arrays) are notable, so they rank
    # above low_single (4.9, one array). low_pair (5.18 sigma) is more
    # significant than seed_single (5.07 sigma).
    assert low_pair[0].group_rank == 2
    assert seed_single.group_rank == 3
    assert low_single.group_rank == 4


def test_summary_lists_only_notable_groups(tmp_path):
    seed, support = _candidate(10.0, 0.0, snr=6.0), _candidate(10.0, 0.0, snr=3.5)
    noise = [_candidate(20.0 + i, 0.0, snr=3.2) for i in range(5)]
    MultiArrayMapMatcher(summary_directory=tmp_path).match(
        [_result("i1", "i1", [seed, *noise]), _result("i3", "i3", [support])]
    )

    (path,) = tmp_path.glob("map_match_summary_*.json")
    summary = json.loads(path.read_text())
    assert (summary["high_sig"], summary["low_sig"]) == (5.0, 3.0)
    assert summary["n_groups"] == 6
    assert summary["n_groups_not_listed"] == 5
    assert [g["rank"] for g in summary["groups"]] == [1]
    assert summary["groups"][0]["max_snr"] == 6.0


def test_multi_array_config_thresholds():
    matcher = MultiArrayMapMatcherConfig(high_sig=6.0, low_sig=3.5).to_matcher()
    assert (matcher.high_sig, matcher.low_sig) == (6.0, 3.5)

    with pytest.raises(ValueError, match="low_sig"):
        MultiArrayMapMatcherConfig(high_sig=3.0, low_sig=4.0)
