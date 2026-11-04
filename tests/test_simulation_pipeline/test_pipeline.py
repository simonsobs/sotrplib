"""
Tests for the fully simulated pipeline setup.
"""

import copy
import json
import pickle
from itertools import combinations

import numpy as np
import pytest
from astropy import units as u

from sotrplib.handlers.basic import PipelineRunner
from sotrplib.maps.preprocessor import MapPreprocessor
from sotrplib.outputs.core import (
    JSONMapMatchOutput,
    PickleSerializer,
    SourceOutput,
)
from sotrplib.sifter.core import DefaultSifter, SimpleCatalogSifter
from sotrplib.sifter.crossmatch import crossmatch_mask, n_wise_crossmatch
from sotrplib.sifter.map_matching import EmptyMapMatcher, MultiArrayMapMatcher
from sotrplib.sims.maps import SimulatedMap
from sotrplib.source_catalog.core import RegisteredSourceCatalog
from sotrplib.sources.blind import SigmaClipBlindSearch
from sotrplib.sources.force import TwoDGaussianFitter
from sotrplib.sources.forced_photometry import gaussian_fit
from sotrplib.sources.sources import RegisteredSource


def test_created_map(empty_map: SimulatedMap):
    assert isinstance(empty_map, SimulatedMap)


def test_injected_sources_lmfit(
    map_with_sources: tuple[SimulatedMap, list[RegisteredSource]],
):
    new_map, sources = map_with_sources

    assert new_map.finalized
    assert len(sources) > 0

    # See if we can recover them
    forced_sources = gaussian_fit(new_map, source_list=sources, fit_method="lmfit")

    assert len(forced_sources) == len(sources)


def test_basic_pipeline_lmfit(
    tmp_path, map_with_sources: tuple[SimulatedMap, list[RegisteredSource]]
):
    """
    Tests a complete setup of the basic pipeline run.
    """

    new_map, sources = map_with_sources
    source_cat = RegisteredSourceCatalog(sources=sources)
    # The runner releases each map after its analysis, and this map has its data
    # only in memory.
    maps = [copy.deepcopy(new_map), copy.deepcopy(new_map)]

    runner = PipelineRunner(
        map_coadder=None,
        source_catalogs=[source_cat],
        source_injector=None,
        preprocessors=None,
        pointing_provider=None,
        pointing_residual_model=None,
        postprocessors=None,
        source_simulators=None,
        forced_photometry=TwoDGaussianFitter(mode="lmfit"),
        source_subtractor=None,
        blind_search=SigmaClipBlindSearch(),
        sifter=DefaultSifter(),
        source_outputs=[PickleSerializer(directory=tmp_path)],
        map_outputs=None,
    )

    runner.run(maps)


def test_pipeline_map_matching_before_outputs(
    tmp_path, map_with_sources: tuple[SimulatedMap, list[RegisteredSource]]
):
    """
    With no catalog every injected source is a transient candidate; the two
    copies of the map are two maps of the run, so the map matcher groups each
    source's two detections, and the source outputs (written only after
    matching) carry the shared, ranked group_id.
    """
    new_map, sources = map_with_sources
    # The runner releases each map after its analysis, and this map has its data
    # only in memory.
    maps = [copy.deepcopy(new_map), copy.deepcopy(new_map)]

    runner = PipelineRunner(
        map_coadder=None,
        source_catalogs=[],
        source_injector=None,
        preprocessors=None,
        pointing_provider=None,
        pointing_residual_model=None,
        postprocessors=None,
        source_simulators=None,
        forced_photometry=None,
        source_subtractor=None,
        blind_search=SigmaClipBlindSearch(),
        sifter=SimpleCatalogSifter(radius=u.Quantity(30.0, "arcsec")),
        source_outputs=[PickleSerializer(directory=tmp_path)],
        map_outputs=None,
        # both copies share one array, so only min_arrays=1 confirms them
        map_matcher=MultiArrayMapMatcher(min_arrays=1),
        map_match_outputs=[JSONMapMatchOutput(directory=tmp_path / "match")],
    )

    results = runner.run(maps)

    transients = [res[1].transient_candidates for res in results]
    assert [len(t) for t in transients] == [len(sources), len(sources)]
    assert all(c.group_id is not None for t in transients for c in t)
    assert {c.group_id for c in transients[0]} == {c.group_id for c in transients[1]}
    assert sorted(c.group_rank for c in transients[0]) == list(
        range(1, len(sources) + 1)
    )
    assert all(c.map_name == new_map.map_name for t in transients for c in t)

    # both copies share a map name, so their pickles may land in one file
    pickled = [pickle.load(p.open("rb")) for p in tmp_path.glob("*.pickle")]
    assert pickled
    for output in pickled:
        candidates = output["sifted_blind_search"].transient_candidates
        assert len(candidates) == len(sources)
        assert all(c.group_id is not None for c in candidates)

    # one map-match summary for the run, with the same groups
    (summary_path,) = (tmp_path / "match").glob("map_match_summary_*.json")
    summary = json.loads(summary_path.read_text())
    assert summary["min_arrays"] == 1
    assert {g["group_id"] for g in summary["groups"]} == {
        str(c.group_id) for c in transients[0]
    }

    # the same maps with the default min_arrays=2 are all unconfirmed, so
    # their candidates go to the noise and keep their group_id
    runner.map_matcher = MultiArrayMapMatcher()
    maps = [copy.deepcopy(new_map), copy.deepcopy(new_map)]
    for forced, sifted in runner.run(maps):
        assert sifted.transient_candidates == []
        grouped = [c for c in sifted.noise_candidates if c.group_id is not None]
        assert len(grouped) == len(sources)


def test_basic_pipeline_missing_crossmatches(
    tmp_path,
    map_with_sources: tuple[SimulatedMap, list[RegisteredSource]],
    map_with_single_source: tuple[SimulatedMap, list[RegisteredSource]],
):
    """
    Tests a complete setup of the basic pipeline run with different maps
    to verify crossmatch.
    """

    map_1, sources = map_with_sources
    map_2, _ = map_with_single_source
    source_cat = RegisteredSourceCatalog(sources=sources)
    maps = [map_1, map_2]

    runner = PipelineRunner(
        map_coadder=None,
        source_catalogs=[source_cat],
        source_injector=None,
        preprocessors=None,
        pointing_provider=None,
        pointing_residual_model=None,
        postprocessors=None,
        source_simulators=None,
        forced_photometry=TwoDGaussianFitter(mode="lmfit"),
        source_subtractor=None,
        blind_search=SigmaClipBlindSearch(),
        sifter=DefaultSifter(),
        source_outputs=[PickleSerializer(directory=tmp_path)],
        map_outputs=None,
    )

    runner.run(maps)


def test_find_single_source_blind(
    map_with_single_source: tuple[SimulatedMap, list[RegisteredSource]],
):
    """
    Tests that we can find a set of sources that we just injected with
    the blind search algorithm.
    """
    new_map, sources = map_with_single_source

    searcher = SigmaClipBlindSearch()

    found_sources, _ = searcher.search(input_map=new_map)

    assert len(found_sources) == len(sources)

    # Check we can sift them out!
    sifter = SimpleCatalogSifter(
        radius=u.Quantity(30.0, "arcsec"),
    )

    res = sifter.sift(
        sources=found_sources,
        input_map=new_map,
        catalogs=[RegisteredSourceCatalog(sources=sources)],
    )

    assert len(res.source_candidates) == 1
    assert len(res.transient_candidates) == 0


def test_crossmatch_transients(
    map_with_sources: tuple[SimulatedMap, list[RegisteredSource]],
):
    """
    Tests that we can find a set of sources that we just injected with
    the blind search algorithm.
    """
    new_map, sources = map_with_sources

    searcher = SigmaClipBlindSearch()

    found_sources, _ = searcher.search(input_map=new_map)

    assert len(found_sources) == len(sources)

    # Check we can sift them out!
    sifter = SimpleCatalogSifter(
        radius=u.Quantity(30.0, "arcsec"),
    )

    res = sifter.sift(
        sources=found_sources,
        input_map=new_map,
        catalogs=[],
    )

    assert len(res.source_candidates) == 0
    assert len(res.transient_candidates) == len(sources)

    transient_positions = np.array(
        [[src.dec.value, src.ra.value] for src in res.transient_candidates]
    )

    all_positions = combinations([transient_positions] * 2, 2)

    matches = [
        crossmatch_mask(*positions, radius=1.5, return_matches=True)[1]
        for positions in all_positions
    ]

    cross_matches = n_wise_crossmatch(
        matches,
        {"map_1": res.transient_candidates, "map_2": res.transient_candidates},
    )
    assert len(cross_matches) == len(sources)
    # verify that the sources all match to their identical copies
    for match in cross_matches.values():
        assert match[0][1] == match[1][1]


def test_crossmatch_transients_with_no_matches(
    map_with_sources: tuple[SimulatedMap, list[RegisteredSource]],
    second_map_with_sources: tuple[SimulatedMap, list[RegisteredSource]],
):
    """
    Tests that when we have two maps with different simulated sources,
    we don't get any false-positive crossmatches.
    """
    new_map, _ = map_with_sources
    other_map, _ = second_map_with_sources

    searcher = SigmaClipBlindSearch()

    found_sources, _ = searcher.search(input_map=new_map)
    other_sources, _ = searcher.search(input_map=other_map)

    # Check we can sift them out!
    sifter = SimpleCatalogSifter(
        radius=u.Quantity(30.0, "arcsec"),
    )

    res = sifter.sift(
        sources=found_sources,
        input_map=new_map,
        catalogs=[],
    )
    other_res = sifter.sift(
        sources=other_sources,
        input_map=other_map,
        catalogs=[],
    )

    assert len(res.source_candidates) == 0
    assert len(other_res.source_candidates) == 0

    transient_positions = np.array(
        [[src.dec.value, src.ra.value] for src in res.transient_candidates]
    )
    other_positions = np.array(
        [[src.dec.value, src.ra.value] for src in other_res.transient_candidates]
    )

    all_positions = combinations([transient_positions, other_positions], 2)

    matches = [
        crossmatch_mask(*positions, radius=1.5, return_matches=True)[1]
        for positions in all_positions
    ]

    cross_matches = n_wise_crossmatch(
        matches,
        {"map_1": res.transient_candidates, "map_2": other_res.transient_candidates},
    )
    assert len(cross_matches) == 0


def test_crossmatch_transients_with_partial_matches(
    map_with_sources: tuple[SimulatedMap, list[RegisteredSource]],
    second_map_with_sources: tuple[SimulatedMap, list[RegisteredSource]],
):
    """
    Tests that when we have two maps with different simulated sources,
    we don't get any false-positive crossmatches.
    """
    new_map, _ = map_with_sources
    other_map, _ = second_map_with_sources

    searcher = SigmaClipBlindSearch()

    found_sources, _ = searcher.search(input_map=new_map)
    other_sources, _ = searcher.search(input_map=other_map)

    # Check we can sift them out!
    sifter = SimpleCatalogSifter(
        radius=u.Quantity(30.0, "arcsec"),
    )

    res = sifter.sift(
        sources=found_sources,
        input_map=new_map,
        catalogs=[],
    )
    other_res = sifter.sift(
        sources=other_sources,
        input_map=other_map,
        catalogs=[],
    )

    assert len(res.source_candidates) == 0
    assert len(other_res.source_candidates) == 0

    transient_positions = np.array(
        [[src.dec.value, src.ra.value] for src in res.transient_candidates]
    )
    other_positions = np.array(
        [[src.dec.value, src.ra.value] for src in other_res.transient_candidates]
    )

    all_positions = combinations(
        [transient_positions, transient_positions, other_positions], 2
    )

    matches = [
        crossmatch_mask(*positions, radius=1.5, return_matches=True)[1]
        for positions in all_positions
    ]

    cross_matches = n_wise_crossmatch(
        matches,
        {
            "map_1": res.transient_candidates,
            "map_2": res.transient_candidates,
            "map_3": other_res.transient_candidates,
        },
    )
    assert len(cross_matches) == 4
    # verify that the sources all match to their identical copies
    for match in cross_matches.values():
        # verify match is only between the two identical sets of sources
        assert len(match) == 2
        assert match[0][1] == match[1][1]


def test_find_sources_blind(
    map_with_sources: tuple[SimulatedMap, list[RegisteredSource]],
):
    """
    Tests that we can find a set of sources that we just injected with
    the blind search algorithm.
    """
    new_map, sources = map_with_sources

    searcher = SigmaClipBlindSearch()

    found_sources, _ = searcher.search(input_map=new_map)

    assert len(found_sources) == len(sources)

    # Check we can sift them out!
    sifter = SimpleCatalogSifter(
        radius=u.Quantity(0.5, "arcmin"),
        method="closest",
    )

    res = sifter.sift(
        sources=found_sources,
        input_map=new_map,
        catalogs=[RegisteredSourceCatalog(sources=sources)],
    )

    # Crossmatches should, well, match
    source_ids = sorted([x.source_id for x in sources])
    match_ids = sorted([x.crossmatches[0].source_id for x in res.source_candidates])
    assert source_ids == match_ids
    assert len(res.source_candidates) == 4
    assert len(res.transient_candidates) == 0


def test_default_sifter(
    map_with_sources: tuple[SimulatedMap, list[RegisteredSource]],
):
    """
    Tests that we can find a set of sources that we just injected with
    the blind search algorithm.
    """
    new_map, sources = map_with_sources
    assert len(sources) > 0
    searcher = SigmaClipBlindSearch()

    found_sources, _ = searcher.search(input_map=new_map)

    assert len(found_sources) == len(sources)

    # Check we can sift them out!
    sifter = DefaultSifter()
    res = sifter.sift(
        sources=found_sources,
        input_map=new_map,
        catalogs=[RegisteredSourceCatalog(sources=sources)],
    )
    # Crossmatches should, well, match
    source_ids = sorted([x.source_id for x in sources])
    match_ids = sorted([x.crossmatches[0].source_id for x in res.source_candidates])
    assert source_ids == match_ids
    assert len(res.source_candidates) == 4
    assert len(res.transient_candidates) == 0


class _RecordingOutput(SourceOutput):
    """Records the sequence of the analyses and the source outputs."""

    def __init__(self, events: list[str]):
        self.events = events

    def output(self, map_name, **kwargs):
        self.events.append("output")


def _recording_runner(events, map_matcher, fail_on_call=None, **kwargs):
    runner = PipelineRunner(
        map_coadder=None,
        source_catalogs=[],
        source_injector=None,
        preprocessors=None,
        pointing_provider=None,
        pointing_residual_model=None,
        postprocessors=None,
        source_simulators=None,
        forced_photometry=None,
        source_subtractor=None,
        blind_search=None,
        sifter=None,
        source_outputs=[_RecordingOutput(events)],
        map_outputs=None,
        map_matcher=map_matcher,
        **kwargs,
    )
    analyze = runner.coadd_and_analyze_maps
    calls = []

    def recording_analyze(maps, simulated_sources):
        calls.append(1)
        events.append("analyze")
        if len(calls) == fail_on_call:
            raise ValueError("bad map")
        return analyze(maps, simulated_sources)

    runner.coadd_and_analyze_maps = recording_analyze
    return runner


def test_default_matcher_writes_each_map_after_its_analysis(empty_map):
    events = []
    runner = _recording_runner(events, EmptyMapMatcher())

    results = runner.run([empty_map, empty_map])

    assert events == ["analyze", "output", "analyze", "output"]
    assert results == []


def test_hold_all_maps_writes_after_all_analyses(empty_map):
    events = []
    runner = _recording_runner(events, EmptyMapMatcher(hold_all_maps=True))

    results = runner.run([empty_map, empty_map])

    assert events == ["analyze", "analyze", "output", "output"]
    assert all(isinstance(r, tuple) for r in results)


def test_failed_map_keeps_the_outputs_of_the_previous_maps(empty_map):
    events = []
    runner = _recording_runner(events, EmptyMapMatcher(), fail_on_call=2)

    with pytest.raises(ValueError, match="bad map"):
        runner.run([empty_map, empty_map])

    # The first map has its source output before the second map fails.
    assert events == ["analyze", "output", "analyze"]


def test_map_match_outputs_need_held_results(tmp_path, empty_map):
    events = []
    runner = _recording_runner(
        events,
        EmptyMapMatcher(),
        map_match_outputs=[JSONMapMatchOutput(directory=tmp_path)],
    )

    with pytest.raises(ValueError, match="hold_all_maps"):
        runner.run([empty_map])

    assert events == []


class _NewMapPreprocessor(MapPreprocessor):
    """Makes a new map, as the matched filter does."""

    def __init__(self):
        self.new_maps = []

    def preprocess(self, input_map):
        new_map = copy.deepcopy(input_map)
        self.new_maps.append(new_map)
        return new_map


def test_runner_releases_map_made_by_preprocessor(empty_map):
    preprocessor = _NewMapPreprocessor()
    runner = _recording_runner([], EmptyMapMatcher())
    runner.preprocessors = [preprocessor]

    runner.run([empty_map])

    (new_map,) = preprocessor.new_maps
    assert not any(isinstance(v, np.ndarray) for v in vars(new_map).values())
    assert not any(isinstance(v, np.ndarray) for v in vars(empty_map).values())
