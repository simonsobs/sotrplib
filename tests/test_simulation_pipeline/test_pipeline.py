"""
Tests for the fully simulated pipeline setup.
"""

import structlog
from astropy import units as u

from sotrplib.handlers.basic import PipelineRunner
from sotrplib.outputs.core import PickleSerializer
from sotrplib.sifter.core import SimpleCatalogSifter
from sotrplib.sims.maps import SimulatedMap
from sotrplib.source_catalog.core import RegisteredSourceCatalog
from sotrplib.sources.blind import SigmaClipBlindSearch
from sotrplib.sources.force import PhotutilsGaussianFitter, Scipy2DGaussianFitter
from sotrplib.sources.forced_photometry import (
    photutils_2D_gauss_fit,
    scipy_2d_gaussian_fit,
)
from sotrplib.sources.sources import RegisteredSource


def test_created_map(empty_map: SimulatedMap):
    assert isinstance(empty_map, SimulatedMap)


def test_injected_sources_scipy(
    map_with_sources: tuple[SimulatedMap, list[RegisteredSource]],
):
    new_map, sources = map_with_sources

    assert new_map.finalized
    assert len(sources) > 0

    # See if we can recover them
    forced_sources = scipy_2d_gaussian_fit(
        new_map, source_catalog=[s.to_forced_photometry_source() for s in sources]
    )

    assert len(forced_sources) == len(sources)


def test_basic_pipeline_scipy(
    tmp_path, map_with_sources: tuple[SimulatedMap, list[RegisteredSource]]
):
    """
    Tests a complete setup of the basic pipeline run.
    """

    new_map, sources = map_with_sources

    runner = PipelineRunner(
        maps=[new_map, new_map],
        preprocessors=None,
        postprocessors=None,
        source_simulators=None,
        forced_photometry=Scipy2DGaussianFitter(
            sources=[s.to_forced_photometry_source() for s in sources]
        ),
        source_subtractor=None,
        blind_search=None,
        sifter=None,
        outputs=[PickleSerializer(directory=tmp_path)],
    )

    runner.run()


def test_injected_sources(
    map_with_sources: tuple[SimulatedMap, list[RegisteredSource]],
):
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
    tmp_path, map_with_sources: tuple[SimulatedMap, list[RegisteredSource]]
):
    """
    Tests a complete setup of the basic pipeline run.
    """

    new_map, sources = map_with_sources

    runner = PipelineRunner(
        maps=[new_map, new_map],
        preprocessors=None,
        postprocessors=None,
        source_simulators=None,
        forced_photometry=PhotutilsGaussianFitter(sources=sources),
        source_subtractor=None,
        blind_search=None,
        sifter=None,
        outputs=[PickleSerializer(directory=tmp_path)],
    )

    runner.run()


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
        catalog=RegisteredSourceCatalog(sources=sources),
        radius=u.Quantity(30.0, "arcsec"),
    )

    res = sifter.sift(sources=found_sources, input_map=new_map)

    assert len(res.source_candidates) == 1
    assert len(res.transient_candidates) == 0


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
        catalog=RegisteredSourceCatalog(sources=sources[:32]),
        radius=u.Quantity(0.5, "arcmin"),
        method="closest",
    )

    res = sifter.sift(sources=found_sources, input_map=new_map)

    # Crossmatches should, well, match
    source_ids = sorted([x.sourceID for x in sources[:32]])
    match_ids = sorted([x.crossmatch_names[0] for x in res.source_candidates])

    assert source_ids == match_ids

    assert len(res.source_candidates) == 32
    assert len(res.transient_candidates) == 32
