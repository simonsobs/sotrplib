import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord

from sotrplib.maps.pointing import ConstantPointingOffset, PolynomialPointingOffset
from sotrplib.source_catalog.core import RegisteredSourceCatalog
from sotrplib.sources.force import (
    TwoDGaussianFitter,
    TwoDGaussianPointingFitter,
)
from sotrplib.sources.sources import MeasuredSource


def test_median_residual_empty(map_with_single_source):
    input_map, sources = map_with_single_source
    forced_photometry = TwoDGaussianFitter(
        mode="scipy",
        flux_limit_centroid=0.1 * u.Jy,
        reproject_thumbnails=False,
        thumbnail_half_width=10.0 * u.arcmin,
    )

    ra_offset = 1.2 * u.arcmin
    dec_offset = -2.0 * u.arcmin
    new_sources = [x.model_copy(deep=True) for x in sources]
    new_sources[0].ra += ra_offset
    new_sources[0].dec += dec_offset

    catalog = RegisteredSourceCatalog(sources=[])
    catalog.add_sources(sources=new_sources)
    catalog.valid_fluxes = [s.source_id for s in new_sources]
    _ = forced_photometry.force(input_map=input_map, catalogs=[catalog])
    pointing_residual_generator = ConstantPointingOffset(
        min_num=5, min_snr=5, avg_method="median"
    )
    _ = pointing_residual_generator.build_model(pointing_sources=[])
    assert pointing_residual_generator.pointing_model is None


def test_median_residual_notenough_sources(map_with_single_source):
    input_map, sources = map_with_single_source
    forced_photometry = TwoDGaussianFitter(
        mode="scipy",
        flux_limit_centroid=0.1 * u.Jy,
        reproject_thumbnails=False,
        thumbnail_half_width=10.0 * u.arcmin,
    )

    ra_offset = 1.2 * u.arcmin
    dec_offset = -2.0 * u.arcmin
    new_sources = [x.model_copy(deep=True) for x in sources]
    new_sources[0].ra += ra_offset
    new_sources[0].dec += dec_offset

    catalog = RegisteredSourceCatalog(sources=[])
    catalog.add_sources(sources=new_sources)
    catalog.valid_fluxes = [s.source_id for s in new_sources]
    results = forced_photometry.force(input_map=input_map, catalogs=[catalog])
    pointing_residual_generator = ConstantPointingOffset(
        min_num=5, min_snr=5, avg_method="median"
    )
    _ = pointing_residual_generator.build_model(pointing_sources=results)
    assert pointing_residual_generator.pointing_model is None


def test_median_residual(map_with_sources):
    input_map, sources = map_with_sources
    pointing_fitter = TwoDGaussianPointingFitter(
        mode="scipy",
        min_flux=0.1 * u.Jy,
        reproject_thumbnails=True,
        thumbnail_half_width=10.0 * u.arcmin,
    )

    ra_offset = 1.2 * u.arcmin
    dec_offset = -2.0 * u.arcmin
    new_sources = [x.model_copy(deep=True) for x in sources]
    for i in range(len(new_sources)):
        new_sources[i].ra += ra_offset
        new_sources[i].dec += dec_offset

    catalog = RegisteredSourceCatalog(sources=[])
    catalog.add_sources(sources=new_sources)
    catalog.valid_fluxes = [s.source_id for s in new_sources]
    results = pointing_fitter.force(input_map=input_map, catalogs=[catalog])
    pointing_residual_generator = ConstantPointingOffset(
        min_num=5, min_snr=5, sigma_clip_level=3, avg_method="median"
    )
    pointing_model, _ = pointing_residual_generator.build_model(
        pointing_sources=results
    )

    assert abs(pointing_model.ra_offset - ra_offset) < 0.25 * u.arcmin
    assert abs(pointing_model.dec_offset - dec_offset) < 0.25 * u.arcmin

    for og_src, new_src in zip(sources, new_sources):
        old_pos = SkyCoord(ra=og_src.ra, dec=og_src.dec)
        new_pos = SkyCoord(ra=new_src.ra, dec=new_src.dec)
        new_pos = pointing_model.predict(new_pos)
        assert abs(old_pos.ra - new_pos.ra) < 0.25 * u.arcmin
        assert abs(old_pos.dec - new_pos.dec) < 0.25 * u.arcmin


def _make_source(ra_deg, dec_deg, offset_ra_deg, offset_dec_deg, snr):
    return MeasuredSource(
        ra=ra_deg * u.deg,
        dec=dec_deg * u.deg,
        offset_ra=offset_ra_deg * u.deg,
        offset_dec=offset_dec_deg * u.deg,
        snr=snr,
    )


def test_polynomial_pointing_not_enough_sources():
    """With fewer sources than min_num the model should be None."""
    sources = [_make_source(20.0, -1.0, 0.02, -0.03, 10.0)]
    fitter = PolynomialPointingOffset(min_snr=5, min_num=5, poly_order=1)
    result = fitter.build_model(pointing_sources=sources)
    assert result is None
    assert fitter.pointing_model is None


def test_polynomial_pointing_order1_constant_offset():
    """
    A spatially uniform offset should be recovered by a poly_order=1 fit.
    Sources are spread across a 4-deg patch; all have the same dra/ddec.
    The model's predict() should undo the offset to within 0.1 arcmin.
    """
    ra_deg_offset = (1.2 * u.arcmin).to_value(u.deg)
    dec_deg_offset = (-2.0 * u.arcmin).to_value(u.deg)

    rng = np.random.default_rng(42)
    ras = rng.uniform(18.0, 22.0, 20)
    decs = rng.uniform(-3.0, 1.0, 20)

    sources = [
        _make_source(ra, dec, ra_deg_offset, dec_deg_offset, snr=20.0)
        for ra, dec in zip(ras, decs)
    ]

    fitter = PolynomialPointingOffset(min_snr=5, min_num=5, poly_order=1)
    fitter.build_model(pointing_sources=sources)
    assert fitter.pointing_model is not None

    for ra, dec in zip(ras, decs):
        measured = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
        corrected = fitter.pointing_model.predict(measured)
        assert abs(corrected.ra - (ra - ra_deg_offset) * u.deg) < 0.1 * u.arcmin
        assert abs(corrected.dec - (dec - dec_deg_offset) * u.deg) < 0.1 * u.arcmin


def test_polynomial_pointing_order1_linear_offset():
    """
    A spatially varying offset that is linear in ra and dec should be recovered by a poly_order=1 fit.
    The model's predict() should undo the offset to within 0.1 arcmin.
    """
    rng = np.random.default_rng(42)
    ras = rng.uniform(18.0, 22.0, 20)
    decs = rng.uniform(-3.0, 1.0, 20)

    sources = []
    for ra, dec in zip(ras, decs):
        ra_offset = (1.2 * u.arcmin).to_value(u.deg) + 0.01 * (ra - 20.0)
        dec_offset = (-2.0 * u.arcmin).to_value(u.deg) + 0.01 * (dec + 1.0)
        sources.append(_make_source(ra, dec, ra_offset, dec_offset, snr=20.0))

    fitter = PolynomialPointingOffset(min_snr=5, min_num=5, poly_order=1)
    fitter.build_model(pointing_sources=sources)
    assert fitter.pointing_model is not None

    for ra, dec in zip(ras, decs):
        measured = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
        corrected = fitter.pointing_model.predict(measured)
        true_ra_offset = (1.2 * u.arcmin).to_value(u.deg) + 0.01 * (ra - 20.0)
        true_dec_offset = (-2.0 * u.arcmin).to_value(u.deg) + 0.01 * (dec + 1.0)
        assert abs(corrected.ra - (ra - true_ra_offset) * u.deg) < 0.1 * u.arcmin
        assert abs(corrected.dec - (dec - true_dec_offset) * u.deg) < 0.1 * u.arcmin


def test_mean_residual(map_with_sources):
    input_map, sources = map_with_sources
    pointing_fitter = TwoDGaussianPointingFitter(
        mode="scipy",
        min_flux=0.1 * u.Jy,
        reproject_thumbnails=True,
        thumbnail_half_width=10.0 * u.arcmin,
    )

    ra_offset = 1.2 * u.arcmin
    dec_offset = -2.0 * u.arcmin
    new_sources = [x.model_copy(deep=True) for x in sources]
    for source in new_sources:
        source.ra += ra_offset
        source.dec += dec_offset

    catalog = RegisteredSourceCatalog(sources=[])
    catalog.add_sources(sources=new_sources)
    catalog.valid_fluxes = [s.source_id for s in new_sources]
    results = pointing_fitter.force(input_map=input_map, catalogs=[catalog])
    pointing_residual_generator = ConstantPointingOffset(
        min_num=5, min_snr=5, avg_method="mean"
    )
    pointing_model, _ = pointing_residual_generator.build_model(
        pointing_sources=results
    )

    assert abs(pointing_model.ra_offset - ra_offset) < 0.25 * u.arcmin
    assert abs(pointing_model.dec_offset - dec_offset) < 0.25 * u.arcmin

    for og_src, new_src in zip(sources, new_sources):
        old_pos = SkyCoord(ra=og_src.ra, dec=og_src.dec)
        new_pos = SkyCoord(ra=new_src.ra, dec=new_src.dec)
        new_pos = pointing_model.predict(new_pos)
        assert abs(old_pos.ra - new_pos.ra) < 0.25 * u.arcmin
        assert abs(old_pos.dec - new_pos.dec) < 0.25 * u.arcmin
