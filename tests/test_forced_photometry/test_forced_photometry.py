import numpy as np
import pytest
from astropy import units as u
from pixell import enmap

from sotrplib.maps.weights import POOR_WEIGHTS_CENTER, PoorWeightsCriteria
from sotrplib.source_catalog.core import RegisteredSourceCatalog
from sotrplib.sources.force import (
    EmptyForcedPhotometry,
    SimpleForcedPhotometry,
    TwoDGaussianFitter,
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


def test_lmfit(map_with_single_source):
    input_map, sources = map_with_single_source
    forced_photometry = TwoDGaussianFitter(mode="lmfit")
    source_cat = RegisteredSourceCatalog(sources=[])
    source_cat.add_sources(sources=sources)
    source_cat.valid_fluxes = [s.source_id for s in sources]
    results = forced_photometry.force(input_map=input_map, catalogs=[source_cat])
    assert results[0].crossmatches
    assert len(results) == 1
    results = forced_photometry.force(input_map=input_map, catalogs=[source_cat])
    assert results[0].flux.to(u.Jy).value == pytest.approx(
        sources[0].flux.to(u.Jy).value, rel=2e-1
    )
    assert results[0].offset_ra.to(u.arcmin).value < 0.5
    assert results[0].offset_dec.to(u.arcmin).value < 0.5
    assert results[0].fit_method == "lmfit_2d_gaussian"


def test_lmfit_source_offset(map_with_single_source):
    input_map, sources = map_with_single_source
    forced_photometry = TwoDGaussianFitter(mode="lmfit", reproject_thumbnails=False)

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
    assert results[0].fit_method == "lmfit_2d_gaussian"


def test_lmfit_rotation(map_with_single_asymmetric_source):
    input_map, sources = map_with_single_asymmetric_source

    forced_photometry = TwoDGaussianFitter(
        mode="lmfit", thumbnail_half_width=0.2 * u.deg
    )
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


def _catalog(sources):
    catalog = RegisteredSourceCatalog(sources=[])
    catalog.add_sources(sources=sources)
    catalog.valid_fluxes = [s.source_id for s in sources]
    return catalog


def _weights_with_hole(input_map, source, half_width_pix=3):
    weights = enmap.ndmap(np.ones(input_map.flux.shape), input_map.flux.wcs)
    y, x = (
        int(np.round(p))
        for p in weights.sky2pix(
            [source.dec.to_value(u.rad), source.ra.to_value(u.rad)]
        )
    )
    weights[
        y - half_width_pix : y + half_width_pix + 1,
        x - half_width_pix : x + half_width_pix + 1,
    ] = 0.0
    return weights


def test_lmfit_poor_weights_flagged(map_with_single_source):
    input_map, sources = map_with_single_source
    input_map.weights = _weights_with_hole(input_map, sources[0])

    forced_photometry = TwoDGaussianFitter(
        mode="lmfit", poor_weights=PoorWeightsCriteria()
    )
    results = forced_photometry.force(input_map=input_map, catalogs=[_catalog(sources)])

    assert len(results) == 1
    assert POOR_WEIGHTS_CENTER in results[0].flags
    ## flag only; the fit is still done
    assert not results[0].fit_failed


def test_lmfit_poor_weights_disabled(map_with_single_source):
    input_map, sources = map_with_single_source
    input_map.weights = _weights_with_hole(input_map, sources[0])

    forced_photometry = TwoDGaussianFitter(mode="lmfit")
    results = forced_photometry.force(input_map=input_map, catalogs=[_catalog(sources)])

    assert len(results) == 1
    assert not any(f.startswith("poor_weights") for f in results[0].flags)


def test_lmfit_poor_weights_without_weight_map(map_with_single_source):
    input_map, sources = map_with_single_source
    assert input_map.weights is None

    forced_photometry = TwoDGaussianFitter(
        mode="lmfit", poor_weights=PoorWeightsCriteria()
    )
    results = forced_photometry.force(input_map=input_map, catalogs=[_catalog(sources)])

    assert len(results) == 1
    assert not any(f.startswith("poor_weights") for f in results[0].flags)
