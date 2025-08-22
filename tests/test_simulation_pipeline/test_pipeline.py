"""
Tests for the fully simulated pipeline setup.
"""

import structlog

from sotrplib.handlers.basic import PipelineRunner
from sotrplib.maps.core import SimulatedMap
from sotrplib.outputs.core import PickleSerializer
from sotrplib.sources.core import PhotutilsGaussianFitter, SigmaClipBlindSearch
from sotrplib.sources.forced_photometry import photutils_2D_gauss_fit
from sotrplib.sources.sources import SourceCandidate


def test_created_map(empty_map: SimulatedMap):
    assert isinstance(empty_map, SimulatedMap)


def test_injected_sources(map_with_sources: tuple[SimulatedMap, list[SourceCandidate]]):
    new_map, sources = map_with_sources

    assert new_map.finalized
    assert len(sources) > 0

    # See if we can recover them
    fits, thumbnails = photutils_2D_gauss_fit(
        flux_map=new_map.flux,
        snr_map=new_map.snr,
        source_catalog=sources,
        log=structlog.get_logger(),
        return_thumbnails=True,
    )

    # Can't look at length of fits because the fits
    # are skipped for items near the edge.
    assert len(thumbnails) == len(sources)


def test_basic_pipeline(
    tmp_path, map_with_sources: tuple[SimulatedMap, list[SourceCandidate]]
):
    """
    Tests a complete setup of the basic pipeline run.
    """

    new_map, sources = map_with_sources

    runner = PipelineRunner(
        maps=[new_map, new_map],
        preprocessors=None,
        postprocessors=None,
        forced_photometry=PhotutilsGaussianFitter(sources=sources),
        source_subtractor=None,
        blind_search=None,
        sifter=None,
        outputs=[PickleSerializer(directory=tmp_path)],
    )

    runner.run()


def test_find_sources_blind(
    map_with_sources: tuple[SimulatedMap, list[SourceCandidate]],
):
    """
    Tests that we can find a set of sources that we just injected with
    the blind search algorithm.
    """
    new_map, sources = map_with_sources

    searcher = SigmaClipBlindSearch()

    found_sources, _ = searcher.search(input_map=new_map)

    assert len(found_sources) == len(sources)
