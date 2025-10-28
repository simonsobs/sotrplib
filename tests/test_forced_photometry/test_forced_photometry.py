import numpy as np
import pytest
from astropy import units as u

from sotrplib.source_catalog.core import RegisteredSourceCatalog
from sotrplib.source_catalog.socat import SOCatFITSCatalog
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

    results = forced_photometry.force(input_map=input_map, catalogs=[])
    assert results == []
    source_cat = RegisteredSourceCatalog(sources=sources)
    forced_photometry = SimpleForcedPhotometry(mode="nn")
    results = forced_photometry.force(input_map=input_map, catalogs=[source_cat])

    assert len(results) == 1
    assert results[0].flux.to(u.Jy).value == pytest.approx(
        sources[0].flux.to(u.Jy).value, rel=2e-1
    )
    assert results[0].fit_method == "nearest_neighbor"


def test_empty_forced_photometry(map_with_single_source):
    input_map, _ = map_with_single_source
    forced_photometry = EmptyForcedPhotometry()
    results = forced_photometry.force(input_map=input_map, catalogs=[])

    assert results == []


def test_scipy_curve_fit(map_with_single_source):
    input_map, sources = map_with_single_source
    forced_photometry = Scipy2DGaussianFitter(thumbnail_half_width=0.1 * u.deg)
    source_cat = SOCatFITSCatalog()
    source_cat.add_sources(sources=sources)
    source_cat.valid_fluxes = [s.source_id for s in sources]
    results = forced_photometry.force(input_map=input_map, catalogs=[source_cat])
    assert len(results) == 1
    ntries = 10
    if results[0].fit_failed:
        while results[0].fit_failed and ntries > 0:
            results = forced_photometry.force(
                input_map=input_map, catalogs=[source_cat]
            )
            ntries -= 1

    assert results[0].flux.to(u.Jy).value == pytest.approx(
        sources[0].flux.to(u.Jy).value, rel=2e-1
    )
    assert results[0].offset_ra.to(u.arcmin).value < 0.5
    assert results[0].offset_dec.to(u.arcmin).value < 0.5
    assert results[0].fit_method == "2d_gaussian"


def test_failed_fit(map_with_single_source):
    input_map, sources = map_with_single_source
    input_map.flux *= np.nan
    forced_photometry = Scipy2DGaussianFitter(reproject_thumbnails=True)
    source_cat = RegisteredSourceCatalog(sources=sources)
    results = forced_photometry.force(input_map=input_map, catalogs=[source_cat])
    assert len(results) == 1
    assert results[0].fit_failed


def test_source_offset(map_with_single_source):
    input_map, sources = map_with_single_source
    forced_photometry = Scipy2DGaussianFitter(reproject_thumbnails=False)

    ra_offset = 0.02 * u.deg
    new_sources = [x.model_copy() for x in sources]
    new_sources[0].ra += ra_offset

    catalog = SOCatFITSCatalog()
    catalog.add_sources(sources=new_sources)
    catalog.valid_fluxes = [s.source_id for s in new_sources]
    results = forced_photometry.force(input_map=input_map, catalogs=[catalog])
    assert len(results) == 1
    assert not results[0].fit_failed
    assert results[0].offset_ra.to(u.arcmin).value == pytest.approx(
        ra_offset.to(u.arcmin).value, abs=0.5
    )
