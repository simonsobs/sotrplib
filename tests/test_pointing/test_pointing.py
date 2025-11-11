from astropy import units as u
from astropy.coordinates import SkyCoord

from sotrplib.maps.pointing import ConstantPointingOffset
from sotrplib.source_catalog.socat import SOCatFITSCatalog
from sotrplib.sources.force import (
    Scipy2DGaussianFitter,
    Scipy2DGaussianPointingFitter,
)


def test_median_residual_empty(map_with_single_source):
    input_map, sources = map_with_single_source
    forced_photometry = Scipy2DGaussianFitter(
        flux_limit_centroid=0.1 * u.Jy,
        reproject_thumbnails=False,
        thumbnail_half_width=10.0 * u.arcmin,
    )

    ra_offset = 1.2 * u.arcmin
    dec_offset = -2.0 * u.arcmin
    new_sources = [x.model_copy(deep=True) for x in sources]
    new_sources[0].ra += ra_offset
    new_sources[0].dec += dec_offset

    catalog = SOCatFITSCatalog()
    catalog.add_sources(sources=new_sources)
    catalog.valid_fluxes = [s.source_id for s in new_sources]
    _ = forced_photometry.force(input_map=input_map, catalogs=[catalog])
    pointing_residual_generator = ConstantPointingOffset(
        min_num=5, min_snr=5, avg_method="median"
    )
    _ = pointing_residual_generator.get_offset(pointing_sources=[])
    assert pointing_residual_generator.ra_offset == 0 * u.arcsec
    assert pointing_residual_generator.dec_offset == 0 * u.arcsec


def test_median_residual_notenough_sources(map_with_single_source):
    input_map, sources = map_with_single_source
    forced_photometry = Scipy2DGaussianFitter(
        flux_limit_centroid=0.1 * u.Jy,
        reproject_thumbnails=False,
        thumbnail_half_width=10.0 * u.arcmin,
    )

    ra_offset = 1.2 * u.arcmin
    dec_offset = -2.0 * u.arcmin
    new_sources = [x.model_copy(deep=True) for x in sources]
    new_sources[0].ra += ra_offset
    new_sources[0].dec += dec_offset

    catalog = SOCatFITSCatalog()
    catalog.add_sources(sources=new_sources)
    catalog.valid_fluxes = [s.source_id for s in new_sources]
    results = forced_photometry.force(input_map=input_map, catalogs=[catalog])
    pointing_residual_generator = ConstantPointingOffset(
        min_num=5, min_snr=5, avg_method="median"
    )
    _ = pointing_residual_generator.get_offset(pointing_sources=results)
    assert pointing_residual_generator.ra_offset == 0 * u.arcsec
    assert pointing_residual_generator.dec_offset == 0 * u.arcsec


def test_median_residual(map_with_sources):
    input_map, sources = map_with_sources
    pointing_fitter = Scipy2DGaussianPointingFitter(
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

    catalog = SOCatFITSCatalog()
    catalog.add_sources(sources=new_sources)
    catalog.valid_fluxes = [s.source_id for s in new_sources]
    results = pointing_fitter.force(input_map=input_map, catalogs=[catalog])
    pointing_residual_generator = ConstantPointingOffset(
        min_num=5, min_snr=5, sigma_clip_level=3, avg_method="median"
    )
    _ = pointing_residual_generator.get_offset(pointing_sources=results)

    assert pointing_residual_generator.ra_offset_rms < 0.25 * u.arcmin
    assert pointing_residual_generator.dec_offset_rms < 0.25 * u.arcmin

    assert abs(pointing_residual_generator.ra_offset - ra_offset) < 0.25 * u.arcmin
    assert abs(pointing_residual_generator.dec_offset - dec_offset) < 0.25 * u.arcmin

    for og_src, new_src in zip(sources, new_sources):
        old_pos = SkyCoord(ra=og_src.ra, dec=og_src.dec)
        new_pos = SkyCoord(ra=new_src.ra, dec=new_src.dec)
        new_pos = pointing_residual_generator.apply_offset_at_position(new_pos)
        assert abs(old_pos.ra - new_pos.ra) < 0.25 * u.arcmin
        assert abs(old_pos.dec - new_pos.dec) < 0.25 * u.arcmin


def test_mean_residual(map_with_sources):
    input_map, sources = map_with_sources
    pointing_fitter = Scipy2DGaussianPointingFitter(
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

    catalog = SOCatFITSCatalog()
    catalog.add_sources(sources=new_sources)
    catalog.valid_fluxes = [s.source_id for s in new_sources]
    results = pointing_fitter.force(input_map=input_map, catalogs=[catalog])
    pointing_residual_generator = ConstantPointingOffset(
        min_num=5, min_snr=5, avg_method="mean"
    )
    _ = pointing_residual_generator.get_offset(pointing_sources=results)

    assert pointing_residual_generator.ra_offset_rms < 0.25 * u.arcmin
    assert pointing_residual_generator.dec_offset_rms < 0.25 * u.arcmin

    assert abs(pointing_residual_generator.ra_offset - ra_offset) < 0.25 * u.arcmin
    assert abs(pointing_residual_generator.dec_offset - dec_offset) < 0.25 * u.arcmin

    for og_src, new_src in zip(sources, new_sources):
        old_pos = SkyCoord(ra=og_src.ra, dec=og_src.dec)
        new_pos = SkyCoord(ra=new_src.ra, dec=new_src.dec)
        new_pos = pointing_residual_generator.apply_offset_at_position(new_pos)
        assert abs(old_pos.ra - new_pos.ra) < 0.25 * u.arcmin
        assert abs(old_pos.dec - new_pos.dec) < 0.25 * u.arcmin
