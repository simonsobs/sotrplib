import numpy as np
import pytest
from astropy import units as u

from sotrplib.sims.sources.core import (
    RandomSourceSimulation,
    RandomSourceSimulationParameters,
)
from sotrplib.source_catalog.core import RegisteredSourceCatalog
from sotrplib.sources.force import (
    EmptyForcedPhotometry,
    SimpleForcedPhotometry,
    TwoDGaussianFitter,
    TwoDGaussianPointingFitter,
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
    ## forced photometry fits at the source position: no centroid offsets
    assert results[0].fit_params["forced_center"]
    assert results[0].offset_ra is None
    assert results[0].offset_dec is None
    assert results[0].fit_method == "lmfit_2d_gaussian"


def test_lmfit_forced_at_catalog_position(map_with_single_source):
    input_map, sources = map_with_single_source
    forced_photometry = TwoDGaussianFitter(mode="lmfit", reproject_thumbnails=False)

    ## the catalog position is 3 arcmin from the source: the forced flux is
    ## the flux at the catalog position, not the flux of the source
    new_sources = [x.model_copy() for x in sources]
    new_sources[0].dec += 0.05 * u.deg

    catalog = RegisteredSourceCatalog(sources=[])
    catalog.add_sources(sources=new_sources)
    catalog.valid_fluxes = [s.source_id for s in new_sources]
    results = forced_photometry.force(input_map=input_map, catalogs=[catalog])
    assert len(results) == 1
    assert not results[0].fit_failed
    assert results[0].flux.to(u.Jy).value < 0.1 * sources[0].flux.to(u.Jy).value


def test_lmfit_pointing_source_offset(map_with_single_source):
    input_map, sources = map_with_single_source
    forced_photometry = TwoDGaussianPointingFitter(
        mode="lmfit", reproject_thumbnails=False
    )

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
    peaks: list[tuple[float, float, float]], noise: float = 1.0, seed: int = 1
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
    thumb = np.random.default_rng(seed).normal(0.0, noise, X.shape)
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


def test_lmfit_forced_center_noise_is_not_biased():
    ## no sources: forced fluxes are noise around zero, not noise peaks
    fluxes = []
    for seed in range(40):
        source = _thumbnail_source([], seed=seed)
        fitter = LmFitGaussian2DFitter(source, force_center=True)
        fitter.initialize_model()
        fluxes.append(fitter.fit().amplitude.to_value(u.mJy))
    assert 0.25 < np.mean(np.array(fluxes) < 0) < 0.75


def test_lmfit_free_center_starts_at_thumbnail_center():
    ## the brightest pixel is 5 arcmin from the center, and it is not the start
    source = _thumbnail_source([(50.0, 1.5, 0.0), (200.0, -5.0, 5.0)])
    fitter = LmFitGaussian2DFitter(
        source, force_center=False, allowable_center_offset=3.0 * u.arcmin
    )
    fitter.initialize_model()

    assert fitter.initial_guess["x0"].value == 0.0
    assert fitter.initial_guess["y0"].value == 0.0
    assert fitter.initial_guess["x0"].max == pytest.approx(6.0)
    assert fitter.initial_guess["y0"].min == pytest.approx(-6.0)


@pytest.mark.parametrize("dx,dy", [(1.5, 0.0), (-1.0, 1.0), (2.0, 0.0), (0.0, 0.0)])
def test_lmfit_free_center_finds_offset_source(dx, dy):
    ## a pointing source off the center: the fit moves from the center to it
    source = _thumbnail_source([(50.0, dx, dy)])
    fitter = LmFitGaussian2DFitter(
        source, force_center=False, allowable_center_offset=3.0 * u.arcmin
    )
    fitter.initialize_model()
    fit = fitter.fit()

    assert not fit.failed
    assert fit.amplitude.to_value(u.mJy) == pytest.approx(50.0, rel=0.1)
    assert abs(fit.ra_offset.to_value(u.arcmin)) == pytest.approx(abs(dx), abs=0.1)
    assert abs(fit.dec_offset.to_value(u.arcmin)) == pytest.approx(abs(dy), abs=0.1)


def test_lmfit_free_center_stays_in_allowed_offset():
    ## a peak far outside the limit: the center does not go past the limit
    source = _thumbnail_source([(50.0, 6.0, 0.0)])
    limit = 1.0 * u.arcmin
    fitter = LmFitGaussian2DFitter(
        source, force_center=False, allowable_center_offset=limit
    )
    fitter.initialize_model()
    fit = fitter.fit()

    if not fit.failed:
        assert abs(fit.ra_offset) <= limit * 1.0001
        assert abs(fit.dec_offset) <= limit * 1.0001


def test_lmfit_config_ignores_centroid_options():
    from sotrplib.config.forced_photometry import LmfitGaussianFitterConfig

    with pytest.warns(DeprecationWarning, match="flux_limit_centroid"):
        config = LmfitGaussianFitterConfig(
            flux_limit_centroid="0.1 Jy", allowable_center_offset="1 arcmin"
        )
    assert isinstance(config.to_forced_photometry(), TwoDGaussianFitter)


def test_lmfit_error_is_map_noise(map_with_single_source):
    input_map, sources = map_with_single_source
    forced_photometry = TwoDGaussianFitter(mode="lmfit")
    catalog = RegisteredSourceCatalog(sources=[])
    catalog.add_sources(sources=sources)
    catalog.valid_fluxes = [s.source_id for s in sources]
    results = forced_photometry.force(input_map=input_map, catalogs=[catalog])

    ## the flux error is the map noise (flux / SNR) at the source position,
    ## not the lmfit error
    pos = np.array([[results[0].dec.to_value(u.rad)], [results[0].ra.to_value(u.rad)]])
    noise = input_map.flux.at(pos, mode="nn")[0] / input_map.snr.at(pos, mode="nn")[0]
    assert results[0].err_flux.to_value(input_map.flux_units) == pytest.approx(noise)
    assert results[0].snr == pytest.approx(
        (results[0].flux / results[0].err_flux).to_value(u.dimensionless_unscaled)
    )


def test_lmfit_fits_at_source_position(empty_map):
    ## the fit coordinates are centered at the source position, also if the
    ## thumbnail has an even number of pixels or the source is not at the
    ## center of a pixel
    parameters = RandomSourceSimulationParameters(
        n_sources=10,
        min_flux=u.Quantity(9.999, "Jy"),
        max_flux=u.Quantity(10.001, "Jy"),
        fwhm_uncertainty_frac=0.0,
        fraction_return=1.0,
    )
    input_map, sources = RandomSourceSimulation(parameters=parameters).simulate(
        input_map=empty_map
    )
    catalog = RegisteredSourceCatalog(sources=[])
    catalog.add_sources(sources=sources)
    catalog.valid_fluxes = [s.source_id for s in sources]

    free = TwoDGaussianPointingFitter(
        mode="lmfit", min_flux=1.0 * u.Jy, near_source_rel_flux_limit=100.0
    ).force(input_map=input_map, catalogs=[catalog])
    free = {r.source_id: r for r in free if not r.fit_failed}
    forced = TwoDGaussianFitter(mode="lmfit").force(
        input_map=input_map, catalogs=[catalog]
    )
    forced = {r.source_id: r for r in forced if not r.fit_failed}

    assert len(free) >= 5
    assert next(iter(free.values())).thumbnail.shape[0] % 2 == 0
    for source_id, fit in free.items():
        ## 0.05 arcmin is 0.1 pixels
        assert abs(fit.fit_params["ra_offset"]) < 0.05 * u.arcmin
        assert abs(fit.fit_params["dec_offset"]) < 0.05 * u.arcmin
        assert forced[source_id].flux.to_value(u.Jy) == pytest.approx(
            fit.flux.to_value(u.Jy), rel=0.005
        )
