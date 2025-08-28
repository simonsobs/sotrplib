import pytest

from sotrplib.sources.force import EmptyForcedPhotometry, SimpleForcedPhotometry


def test_simple_forced_photometry(map_with_single_source):
    input_map, sources = map_with_single_source

    forced_photometry = SimpleForcedPhotometry(mode="nn")

    results = forced_photometry.force(input_map=input_map, sources=[])
    assert results == []

    results = forced_photometry.force(
        input_map=input_map, sources=[s.to_forced_photometry_source() for s in sources]
    )

    assert len(results) == 1
    assert results[0].flux.value == pytest.approx(sources[0].flux, rel=2e-1)
    assert results[0].fit_method == "nearest_neighbor"


def test_empty_forced_photometry(map_with_single_source):
    input_map, sources = map_with_single_source

    forced_photometry = EmptyForcedPhotometry()
    results = forced_photometry.force(input_map=input_map, sources=sources)

    assert results == []
