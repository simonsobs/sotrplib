"""
Each measured source records the map that gave it (map_id, map_name).
"""

import uuid7
from astropy import units as u

from sotrplib.source_catalog.core import RegisteredSourceCatalog
from sotrplib.sources.blind import BlindSearchParameters, SigmaClipBlindSearch
from sotrplib.sources.force import SimpleForcedPhotometry, TwoDGaussianFitter
from sotrplib.sources.sources import MeasuredSource


def _with_mapcat_id(input_map):
    map_id = uuid7.create()
    old = input_map.mapcat_id
    input_map.mapcat_id = map_id
    return map_id, old


def _catalog(sources):
    catalog = RegisteredSourceCatalog(sources=[])
    catalog.add_sources(sources=sources)
    catalog.valid_fluxes = [s.source_id for s in sources]
    return catalog


def test_set_map(map_with_single_source):
    input_map, _ = map_with_single_source
    map_id, old = _with_mapcat_id(input_map)
    try:
        source = MeasuredSource(ra=1 * u.deg, dec=2 * u.deg).set_map(input_map)
    finally:
        input_map.mapcat_id = old
    assert source.map_id == map_id
    assert source.map_name == input_map.map_name


def test_set_map_without_mapcat_id(map_with_single_source):
    input_map, _ = map_with_single_source
    old = input_map.mapcat_id
    input_map.mapcat_id = None
    try:
        source = MeasuredSource(ra=1 * u.deg, dec=2 * u.deg).set_map(input_map)
    finally:
        input_map.mapcat_id = old
    assert source.map_id is None
    assert source.map_name == input_map.map_name


def test_forced_photometry_sets_map(map_with_single_source):
    input_map, sources = map_with_single_source
    map_id, old = _with_mapcat_id(input_map)
    try:
        simple = SimpleForcedPhotometry(mode="nn").force(
            input_map=input_map, catalogs=[_catalog(sources)]
        )
        lmfit = TwoDGaussianFitter(mode="lmfit").force(
            input_map=input_map, catalogs=[_catalog(sources)]
        )
    finally:
        input_map.mapcat_id = old
    for results in (simple, lmfit):
        assert len(results) == 1
        assert results[0].map_id == map_id
        assert results[0].map_name == input_map.map_name


def test_blind_search_sets_map(map_with_single_source):
    input_map, _ = map_with_single_source
    map_id, old = _with_mapcat_id(input_map)
    try:
        found, _ = SigmaClipBlindSearch(
            parameters=BlindSearchParameters(sigma_threshold=5.0),
            thumbnail_half_width=3 * u.arcmin,
        ).search(input_map)
    finally:
        input_map.mapcat_id = old
    assert found
    assert all(s.map_id == map_id for s in found)
    assert all(s.map_name == input_map.map_name for s in found)
