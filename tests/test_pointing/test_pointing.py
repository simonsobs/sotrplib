from astropy import units as u

from sotrplib.maps.pointing import LinearMeanPointingOffset, LinearMedianPointingOffset
from sotrplib.source_catalog.socat import SOCatFITSCatalog
from sotrplib.sources.force import (
    Scipy2DGaussianFitter,
)


def test_median_residual_notenough_sources(map_with_single_source):
    input_map, sources = map_with_single_source
    forced_photometry = Scipy2DGaussianFitter(
        flux_limit_centroid=0.1 * u.Jy,
        reproject_thumbnails=False,
        thumbnail_half_width=10.0 * u.arcmin,
    )

    ra_offset = 1.2 * u.arcmin
    dec_offset = -2.0 * u.arcmin
    new_sources = [x.model_copy() for x in sources]
    new_sources[0].ra += ra_offset
    new_sources[0].dec += dec_offset

    catalog = SOCatFITSCatalog()
    catalog.add_sources(sources=new_sources)
    catalog.valid_fluxes = [s.source_id for s in new_sources]
    results = forced_photometry.force(input_map=input_map, catalogs=[catalog])
    pointing_residual_generator = LinearMedianPointingOffset(min_num=5, min_snr=5)
    pointing_residuals = pointing_residual_generator.get_offset(
        pointing_sources=results
    )
    assert pointing_residual_generator.ra_offset == 0 * u.arcsec
    assert pointing_residual_generator.dec_offset == 0 * u.arcsec


def test_median_residual(map_with_sources):
    input_map, sources = map_with_sources
    forced_photometry = Scipy2DGaussianFitter(
        flux_limit_centroid=0.1 * u.Jy,
        reproject_thumbnails=True,
        thumbnail_half_width=10.0 * u.arcmin,
    )

    ra_offset = 1.2 * u.arcmin
    dec_offset = -2.0 * u.arcmin
    new_sources = [x.model_copy() for x in sources]
    for source in new_sources:
        source.ra += ra_offset
        source.dec += dec_offset

    catalog = SOCatFITSCatalog()
    catalog.add_sources(sources=new_sources)
    catalog.valid_fluxes = [s.source_id for s in new_sources]
    results = forced_photometry.force(input_map=input_map, catalogs=[catalog])
    pointing_residual_generator = LinearMedianPointingOffset(min_num=5, min_snr=5)
    pointing_residuals = pointing_residual_generator.get_offset(
        pointing_sources=results
    )

    assert abs(pointing_residual_generator.ra_offset - ra_offset) < 0.25 * u.arcmin
    assert abs(pointing_residual_generator.dec_offset - dec_offset) < 0.25 * u.arcmin


def test_mean_residual(map_with_sources):
    input_map, sources = map_with_sources
    forced_photometry = Scipy2DGaussianFitter(
        flux_limit_centroid=0.1 * u.Jy,
        reproject_thumbnails=True,
        thumbnail_half_width=10.0 * u.arcmin,
    )

    ra_offset = 1.2 * u.arcmin
    dec_offset = -2.0 * u.arcmin
    new_sources = [x.model_copy() for x in sources]
    for source in new_sources:
        source.ra += ra_offset
        source.dec += dec_offset

    catalog = SOCatFITSCatalog()
    catalog.add_sources(sources=new_sources)
    catalog.valid_fluxes = [s.source_id for s in new_sources]
    results = forced_photometry.force(input_map=input_map, catalogs=[catalog])
    pointing_residual_generator = LinearMeanPointingOffset(min_num=5, min_snr=5)
    pointing_residuals = pointing_residual_generator.get_offset(
        pointing_sources=results
    )

    assert abs(pointing_residual_generator.ra_offset - ra_offset) < 0.25 * u.arcmin
    assert abs(pointing_residual_generator.dec_offset - dec_offset) < 0.25 * u.arcmin
    assert pointing_residual_generator.ra_offset_rms < 0.25 * u.arcmin
    assert pointing_residual_generator.dec_offset_rms < 0.25 * u.arcmin
