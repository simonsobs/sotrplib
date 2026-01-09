"""
Tests for the fully simulated pipeline setup.
"""

from astropy import units as u

from sotrplib.handlers.basic import PipelineRunner
from sotrplib.outputs.core import PickleSerializer
from sotrplib.sifter.core import DefaultSifter, SimpleCatalogSifter
from sotrplib.sims.maps import SimulatedMap
from sotrplib.source_catalog.core import RegisteredSourceCatalog
from sotrplib.sources.blind import SigmaClipBlindSearch
from sotrplib.sources.force import Scipy2DGaussianFitter
from sotrplib.sources.forced_photometry import (
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
    forced_sources = scipy_2d_gaussian_fit(new_map, source_list=sources)

    assert len(forced_sources) == len(sources)


def test_basic_pipeline_scipy(
    tmp_path, map_with_sources: tuple[SimulatedMap, list[RegisteredSource]]
):
    """
    Tests a complete setup of the basic pipeline run.
    """

    new_map, sources = map_with_sources
    source_cat = RegisteredSourceCatalog(sources=sources)

    runner = PipelineRunner(
        maps=[new_map, new_map],
        map_coadder=None,
        source_catalogs=[source_cat],
        sso_catalogs=[],
        source_injector=None,
        preprocessors=None,
        pointing_provider=None,
        pointing_residual=None,
        postprocessors=None,
        source_simulators=None,
        forced_photometry=Scipy2DGaussianFitter(),
        source_subtractor=None,
        blind_search=SigmaClipBlindSearch(),
        sifter=DefaultSifter(),
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
        radius=u.Quantity(30.0, "arcsec"),
    )

    res = sifter.sift(
        sources=found_sources,
        input_map=new_map,
        catalogs=[RegisteredSourceCatalog(sources=sources)],
    )

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
