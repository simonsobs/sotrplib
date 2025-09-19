import numpy as np
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

    results = forced_photometry.force(input_map=input_map, sources=sources)

    assert len(results) == 1
    assert results[0].flux.to(u.Jy).value == pytest.approx(
        sources[0].flux.to(u.Jy).value, rel=2e-1
    )
    assert results[0].fit_method == "nearest_neighbor"


def test_empty_forced_photometry(map_with_single_source):
    input_map, sources = map_with_single_source

    forced_photometry = EmptyForcedPhotometry()
    results = forced_photometry.force(input_map=input_map, sources=sources)

    assert results == []


def test_scipy_curve_fit(map_with_single_source):
    input_map, sources = map_with_single_source

    forced_photometry = Scipy2DGaussianFitter()
    results = forced_photometry.force(input_map=input_map, sources=sources)
    assert len(results) == 1
    ntries = 10
    if results[0].fit_failed:
        assert (
            results[0].fit_failure_reason
            == "source_outside_map_bounds" | "flux_thumb_has_nan"
        )
        while results[0].fit_failed and ntries > 0:
            results = forced_photometry.force(
                input_map=input_map,
                sources=sources,
            )
            ntries -= 1

    assert results[0].flux.to(u.Jy).value == pytest.approx(
        sources[0].flux.to(u.Jy).value, rel=2e-1
    )
    assert results[0].fit_method == "2d_gaussian"


def test_failed_fit(map_with_single_source):
    input_map, sources = map_with_single_source
    input_map.flux *= np.nan
    forced_photometry = Scipy2DGaussianFitter(reproject_thumbnails=True)
    results = forced_photometry.force(input_map=input_map, sources=sources)
    assert len(results) == 1
    assert results[0].fit_failed
    assert results[0].fit_failure_reason == "flux_thumb_has_nan"
