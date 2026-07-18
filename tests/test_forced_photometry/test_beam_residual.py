"""
Tests for the fixed-beam Gaussian model residual used to reject
detections that are not beam shaped (e.g. planet-masking artifacts).
"""

import numpy as np
from astropy import units as u

from sotrplib.sources.forced_photometry import fit_beam_model_residual
from sotrplib.sources.sources import MeasuredSource

BEAM_FWHM = 2.2 * u.arcmin
RES_ARCMIN = 0.5
NPIX = 41


def make_source(image):
    return MeasuredSource(
        ra=10.0 * u.deg,
        dec=0.0 * u.deg,
        thumbnail=image,
        thumbnail_res=u.Quantity([RES_ARCMIN, RES_ARCMIN], "arcmin"),
        thumbnail_unit=u.Jy,
    )


def pixel_grid():
    x = np.arange(NPIX) - (NPIX - 1) / 2
    return np.meshgrid(x, x)


def beam_image(amplitude=1.0, x0=0.0, y0=0.0):
    sigma_pix = BEAM_FWHM.to_value("arcmin") / RES_ARCMIN / 2.355
    X, Y = pixel_grid()
    return amplitude * np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2 * sigma_pix**2))


def test_beam_shaped_source_has_small_residual():
    rng = np.random.default_rng(0)
    image = beam_image(amplitude=1.0) + rng.normal(0, 0.01, (NPIX, NPIX))
    residual = fit_beam_model_residual(make_source(image), BEAM_FWHM)
    assert residual is not None
    assert residual < 0.1


def test_subpixel_offset_still_fits():
    image = beam_image(amplitude=1.0, x0=0.4, y0=-0.3)
    residual = fit_beam_model_residual(make_source(image), BEAM_FWHM)
    assert residual is not None
    assert residual < 0.05


def test_sharp_edged_artifact_has_large_residual():
    # a top-hat disk (e.g. the rim of a masked planet) is far from beam shaped
    X, Y = pixel_grid()
    image = np.where(np.hypot(X, Y) <= 6.0, 1.0, 0.0)
    residual = fit_beam_model_residual(make_source(image), BEAM_FWHM)
    assert residual is not None
    assert residual > 0.2


def test_artifact_scores_worse_than_source():
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 0.01, (NPIX, NPIX))
    source_res = fit_beam_model_residual(
        make_source(beam_image(amplitude=1.0) + noise), BEAM_FWHM
    )
    X, Y = pixel_grid()
    crescent = np.where(
        (np.hypot(X, Y) <= 8.0) & (np.hypot(X - 3.0, Y) > 6.0), 1.0, 0.0
    )
    artifact_res = fit_beam_model_residual(make_source(crescent + noise), BEAM_FWHM)
    assert source_res is not None and artifact_res is not None
    assert artifact_res > 3 * source_res


def test_missing_thumbnail_returns_none():
    source = MeasuredSource(ra=10.0 * u.deg, dec=0.0 * u.deg)
    assert fit_beam_model_residual(source, BEAM_FWHM) is None


def test_nan_thumbnail_returns_none():
    image = beam_image()
    image[0, 0] = np.nan
    assert fit_beam_model_residual(make_source(image), BEAM_FWHM) is None


def test_gaussian_fit_populates_beam_residual(map_with_single_source):
    from sotrplib.source_catalog.core import RegisteredSourceCatalog
    from sotrplib.sources.force import TwoDGaussianFitter

    input_map, sources = map_with_single_source
    source_cat = RegisteredSourceCatalog(sources=[])
    source_cat.add_sources(sources=sources)
    source_cat.valid_fluxes = [s.source_id for s in sources]
    results = TwoDGaussianFitter(mode="lmfit").force(
        input_map=input_map, catalogs=[source_cat]
    )
    assert len(results) == 1
    assert results[0].beam_model_residual is not None
    assert results[0].beam_model_residual < 0.3
