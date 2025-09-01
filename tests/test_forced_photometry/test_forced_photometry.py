import pytest
from astropy import units as u

from sotrplib.sources.force import (
    EmptyForcedPhotometry,
    Scipy2DGaussianFitter,
    SimpleForcedPhotometry,
)

### sources returned from simulations need to be converted to registered sources
### for now, they have implied flux units of Jy


def test_simple_forced_photometry(map_with_single_source):
    input_map, sources = map_with_single_source

    forced_photometry = SimpleForcedPhotometry(mode="nn")

    results = forced_photometry.force(input_map=input_map, sources=[])
    assert results == []

    results = forced_photometry.force(
        input_map=input_map, sources=[s.to_forced_photometry_source() for s in sources]
    )

    assert len(results) == 1
    assert results[0].flux.to(u.Jy).value == pytest.approx(sources[0].flux, rel=2e-1)
    assert results[0].fit_method == "nearest_neighbor"


def test_empty_forced_photometry(map_with_single_source):
    input_map, sources = map_with_single_source

    forced_photometry = EmptyForcedPhotometry()
    results = forced_photometry.force(input_map=input_map, sources=sources)

    assert results == []


def test_scipy_curve_fit(map_with_single_source):
    input_map, sources = map_with_single_source

    forced_photometry = Scipy2DGaussianFitter()
    results = forced_photometry.force(
        input_map=input_map, sources=[s.to_forced_photometry_source() for s in sources]
    )
    assert len(results) == 1
    if results[0].fit_failed:
        assert results[0].fit_failure_reason == "source_near_map_edge"
        max_attempts = 10
        attempts = 0
        while results[0].fit_failed and attempts < max_attempts:
            input_map, sources = map_with_single_source
            results = forced_photometry.force(
                input_map=input_map,
                sources=[s.to_forced_photometry_source() for s in sources],
            )
            attempts += 1
        if results[0].fit_failed:
            pytest.fail(
                f"Scipy2DGaussianFitter.fit repeatedly failed due to source_near_map_edge after {max_attempts} attempts"
            )

    assert results[0].flux.to(u.Jy).value == pytest.approx(sources[0].flux, rel=2e-1)
    assert results[0].fit_method == "2d_gaussian"
