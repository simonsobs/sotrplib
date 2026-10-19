import numpy as np
import pytest
from astropy import units as u

from sotrplib.source_catalog.core import RegisteredSourceCatalog
from sotrplib.sources.force import (
    EmptyForcedPhotometry,
    SimpleForcedPhotometry,
    TwoDGaussianFitter,
)
from sotrplib.sources.forced_photometry import LmFitGaussian2DFitter
from sotrplib.sources.sources import MeasuredSource

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


def _thumbnail_source(
    peaks: list[tuple[float, float, float]], noise: float = 1.0
) -> MeasuredSource:
    """
    Make a source with a 33x33 thumbnail (0.5 arcmin pixels, source position
    at the center) that has white noise and Gaussian beams of 2.2 arcmin FWHM.
    Each peak is (amplitude, x offset in arcmin, y offset in arcmin).
    """

    res = 0.5
    x = (np.arange(33) - 16) * res
    X, Y = np.meshgrid(x, x)
    sigma = 2.2 / np.sqrt(8 * np.log(2))
    thumb = np.random.default_rng(1).normal(0.0, noise, X.shape)
    for amplitude, dx, dy in peaks:
        thumb += amplitude * np.exp(-0.5 * ((X - dx) ** 2 + (Y - dy) ** 2) / sigma**2)
    return MeasuredSource(
        ra=10 * u.deg,
        dec=0 * u.deg,
        thumbnail=thumb,
        thumbnail_res=[-res, res] * u.arcmin,
        thumbnail_unit=u.mJy,
    )


def test_lmfit_forced_center_ignores_peak_away_from_source():
    ## no source at the center, a bright peak 3 arcmin away
    source = _thumbnail_source([(50.0, 3.0, 0.0)])
    fitter = LmFitGaussian2DFitter(source, force_center=True)
    fitter.initialize_model()
    fit = fitter.fit()

    assert not fit.failed
    assert abs(fit.amplitude.to_value(u.mJy)) < 5.0


def test_lmfit_free_center_stays_near_source():
    ## a source 0.5 arcmin from the center and a brighter peak 3 arcmin away
    source = _thumbnail_source([(20.0, 0.5, 0.0), (50.0, 3.0, 0.0)])
    fitter = LmFitGaussian2DFitter(
        source, force_center=False, allowable_center_offset=1.0 * u.arcmin
    )
    fitter.initialize_model()
    fit = fitter.fit()

    assert not fit.failed
    assert fit.amplitude.to_value(u.mJy) == pytest.approx(20.0, rel=0.5)
    assert abs(fit.ra_offset.to_value(u.arcmin)) <= 1.0
