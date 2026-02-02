import pytest
from astropy import units as u

from sotrplib.source_catalog.core import RegisteredSourceCatalog
from sotrplib.sources.force import (
    EmptyForcedPhotometry,
    Lmfit2DGaussianFitter,
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
    assert results[0].crossmatches
    assert results[0].fit_method == "nearest_neighbor"


def test_empty_forced_photometry(map_with_single_source):
    input_map, _ = map_with_single_source
    forced_photometry = EmptyForcedPhotometry()
    results = forced_photometry.force(input_map=input_map, catalogs=[])

    assert results == []


def test_scipy_curve_fit(map_with_single_source):
    input_map, sources = map_with_single_source
    forced_photometry = Scipy2DGaussianFitter(thumbnail_half_width=0.1 * u.deg)
    source_cat = RegisteredSourceCatalog(sources=[])
    source_cat.add_sources(sources=sources)
    source_cat.valid_fluxes = [s.source_id for s in sources]
    results = forced_photometry.force(input_map=input_map, catalogs=[source_cat])
    assert results[0].crossmatches
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


def test_source_offset(map_with_single_source):
    input_map, sources = map_with_single_source
    forced_photometry = Scipy2DGaussianFitter(reproject_thumbnails=False)

    ra_offset = 0.02 * u.deg
    new_sources = [x.model_copy() for x in sources]
    new_sources[0].ra += ra_offset

    catalog = RegisteredSourceCatalog(sources=[])
    catalog.add_sources(sources=new_sources)
    catalog.valid_fluxes = [s.source_id for s in new_sources]
    results = forced_photometry.force(input_map=input_map, catalogs=[catalog])
    assert len(results) == 1
    assert not results[0].fit_failed
    assert results[0].offset_ra.to(u.arcmin).value == pytest.approx(
        ra_offset.to(u.arcmin).value, abs=0.5
    )


def test_lmfit(map_with_single_source):
    input_map, sources = map_with_single_source
    forced_photometry = Lmfit2DGaussianFitter()
    source_cat = RegisteredSourceCatalog(sources=[])
    source_cat.add_sources(sources=sources)
    source_cat.valid_fluxes = [s.source_id for s in sources]
    results = forced_photometry.force(input_map=input_map, catalogs=[source_cat])
    assert results[0].crossmatches
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
    assert results[0].fit_method == "2d_gaussian_lmfit"


def test_lmfit_source_offset(map_with_single_source):
    input_map, sources = map_with_single_source
    forced_photometry = Lmfit2DGaussianFitter(reproject_thumbnails=False)

    ra_offset = 0.02 * u.deg
    new_sources = [x.model_copy() for x in sources]
    new_sources[0].ra += ra_offset

    catalog = RegisteredSourceCatalog(sources=[])
    catalog.add_sources(sources=new_sources)
    catalog.valid_fluxes = [s.source_id for s in new_sources]
    results = forced_photometry.force(input_map=input_map, catalogs=[catalog])
    assert len(results) == 1
    assert not results[0].fit_failed
    assert results[0].offset_ra.to(u.arcmin).value == pytest.approx(
        ra_offset.to(u.arcmin).value, abs=0.5
    )
    assert results[0].fit_method == "2d_gaussian_lmfit"


def test_lmfit_rotation(map_with_single_asymmetric_source):
    input_map, sources = map_with_single_asymmetric_source

    forced_photometry = Lmfit2DGaussianFitter(thumbnail_half_width=0.2 * u.deg)
    source_cat = RegisteredSourceCatalog(sources=[])
    source_cat.add_sources(sources=sources)
    source_cat.valid_fluxes = [s.source_id for s in sources]
    results = forced_photometry.force(input_map=input_map, catalogs=[source_cat])

    assert len(results) == 1
    assert not results[0].fit_failed

    fitted_params = results[0].fit_params
    assert fitted_params is not None

    amplitude_fit = fitted_params.get("amplitude")
    theta_fit = fitted_params.get("theta")

    assert amplitude_fit is not None
    assert theta_fit is not None

    ## TODO : include information about injected source rotation once available
