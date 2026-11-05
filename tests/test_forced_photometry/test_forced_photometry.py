import numpy as np
import pytest
import structlog
from astropy import units as u
from astropy.time import Time, TimeDelta

from sotrplib.maps.maps import make_model_source_map
from sotrplib.sims.maps import SimulatedMap, SimulationParameters
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
from sotrplib.sources.forced_photometry import LmFitGaussian2DFitter, gaussian_fit
from sotrplib.sources.sources import MeasuredSource, RegisteredSource

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
    ## a bright source gets a free fit, and the offset is small
    assert not results[0].fit_params["forced_center"]
    assert abs(results[0].offset_ra.to_value(u.arcmin)) < 0.1
    assert abs(results[0].offset_dec.to_value(u.arcmin)) < 0.1
    assert results[0].fit_method == "lmfit_2d_gaussian"


def test_lmfit_forced_at_catalog_position(map_with_single_source):
    input_map, sources = map_with_single_source
    forced_photometry = TwoDGaussianFitter(
        mode="lmfit", reproject_thumbnails=False, centroid_snr_limit=None
    )

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


@pytest.mark.parametrize("force_center", [True, False])
def test_lmfit_flux_is_height_above_background(force_center):
    ## the fitted background is not part of the flux
    source = _thumbnail_source([(50.0, 0.0, 0.0)])
    source.thumbnail = source.thumbnail - 10.0
    fitter = LmFitGaussian2DFitter(source, force_center=force_center)
    fitter.initialize_model()
    fit = fitter.fit()

    assert not fit.failed
    assert fit.amplitude.to_value(u.mJy) == pytest.approx(50.0, rel=0.1)


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
            flux_limit_centroid="0.1 Jy",
            allowable_center_offset="2 arcmin",
            centroid_snr_limit=7.0,
        )
    fitter = config.to_forced_photometry()
    assert isinstance(fitter, TwoDGaussianFitter)
    assert fitter.allowable_center_offset == 2 * u.arcmin
    assert fitter.centroid_snr_limit == 7.0


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


def _catalog(sources):
    catalog = RegisteredSourceCatalog(sources=[])
    catalog.add_sources(sources=sources)
    catalog.valid_fluxes = [s.source_id for s in sources]
    return catalog


def test_lmfit_bright_source_gets_free_fit(map_with_single_source):
    ## a bright source moved from its catalog position: the fit gives the
    ## offset and the width, and it does not go past the limit
    input_map, sources = map_with_single_source
    moved = [x.model_copy() for x in sources]
    moved[0].dec = moved[0].dec + 0.5 * u.arcmin
    results = TwoDGaussianFitter(mode="lmfit").force(
        input_map=input_map, catalogs=[_catalog(moved)]
    )
    assert not results[0].fit_failed
    assert not results[0].fit_params["forced_center"]
    assert abs(results[0].offset_dec.to_value(u.arcmin)) == pytest.approx(0.5, abs=0.05)
    assert results[0].fwhm_ra.to_value(u.arcmin) == pytest.approx(2.2, abs=0.2)


def test_lmfit_faint_source_stays_at_position(map_with_single_source):
    ## a limit that this source does not reach: the fit stays at the position
    input_map, sources = map_with_single_source
    results = TwoDGaussianFitter(mode="lmfit", centroid_snr_limit=1e12).force(
        input_map=input_map, catalogs=[_catalog(sources)]
    )
    assert not results[0].fit_failed
    assert results[0].fit_params["forced_center"]
    assert results[0].offset_ra is None


def test_lmfit_nearby_source_stays_at_position(map_with_single_source):
    input_map, sources = map_with_single_source
    results = gaussian_fit(
        input_map,
        source_list=[x.model_copy() for x in sources],
        thumbnail_half_width=0.1 * u.deg,
        fwhm=2.2 * u.arcmin,
        centroid_snr_limit=5.0,
        flags={"nearby_source": [True]},
    )
    assert results[0].fit_params["forced_center"]
    assert results[0].offset_ra is None


def test_subtraction_model_is_at_fitted_position(map_with_single_source):
    ## the catalog position is wrong, but the fit finds the source: the model
    ## must be at the source, not at the catalog position
    input_map, sources = map_with_single_source
    true_pixel = np.asarray(
        input_map.flux.sky2pix(
            [sources[0].dec.to_value(u.rad), sources[0].ra.to_value(u.rad)]
        )
    )
    moved = [x.model_copy() for x in sources]
    moved[0].ra = moved[0].ra + 0.5 * u.arcmin
    moved[0].dec = moved[0].dec - 0.5 * u.arcmin
    results = TwoDGaussianFitter(mode="lmfit").force(
        input_map=input_map, catalogs=[_catalog(moved)]
    )
    assert not results[0].fit_params["forced_center"]

    model = np.asarray(
        make_model_source_map(
            input_map.flux,
            results,
            nominal_fwhm=2.2 * u.arcmin,
            matched_filtered=True,
            log=structlog.get_logger(),
        )
    )
    y, x = np.indices(model.shape)
    centroid = np.array([(y * model).sum(), (x * model).sum()]) / model.sum()
    assert centroid == pytest.approx(true_pixel, abs=0.1)


def _map_with_gaussian(
    center_dec: u.Quantity,
    fwhm_major: float,
    fwhm_minor: float,
    position_angle: float,
) -> tuple[SimulatedMap, RegisteredSource, np.ndarray]:
    """
    Make a map with one 10 Jy source. The widths are in arcmin on the sky. The
    position angle is in degrees, from map +x to map +y, in pixel coordinates.
    The source is not at the center of a pixel. Returns the map, a catalog
    source, and the source image without noise.
    """
    input_map = SimulatedMap(
        observation_start=Time.now() - TimeDelta(1, format="jd"),
        observation_end=Time.now(),
        simulation_parameters=SimulationParameters(center_dec=center_dec),
    )
    input_map.build()
    input_map.finalize()
    noise = np.asarray(input_map.flux).copy()
    y_source = input_map.flux.shape[0] / 2 + 0.2
    x_source = input_map.flux.shape[1] / 2 - 0.3
    dec, ra = input_map.flux.pix2sky([y_source, x_source])
    y, x = np.indices(input_map.flux.shape)
    ## pixels in RA are smaller on the sky than in the map coordinates
    dx = (x - x_source) * 0.5 * np.cos(dec)
    dy = (y - y_source) * 0.5
    angle = np.deg2rad(position_angle)
    along = dx * np.cos(angle) + dy * np.sin(angle)
    across = -dx * np.sin(angle) + dy * np.cos(angle)
    to_sigma = 1 / np.sqrt(8 * np.log(2))
    image = 10.0 * np.exp(
        -0.5
        * (
            (along / (fwhm_major * to_sigma)) ** 2
            + (across / (fwhm_minor * to_sigma)) ** 2
        )
    )
    input_map.flux[...] = noise + image
    input_map.snr[...] = (noise + image) / np.std(noise)
    source = RegisteredSource(
        ra=np.rad2deg(ra) * u.deg,
        dec=np.rad2deg(dec) * u.deg,
        flux=10 * u.Jy,
        source_id="source",
    )
    return input_map, source, image


def _fit_and_model(input_map, source, centroid_snr_limit=5.0, reproject=False):
    catalog = RegisteredSourceCatalog(sources=[])
    catalog.add_sources(sources=[source])
    catalog.valid_fluxes = [source.source_id]
    result = TwoDGaussianFitter(
        mode="lmfit",
        centroid_snr_limit=centroid_snr_limit,
        reproject_thumbnails=reproject,
    ).force(input_map=input_map, catalogs=[catalog])[0]
    model = make_model_source_map(
        input_map.flux,
        [result],
        nominal_fwhm=2.2 * u.arcmin,
        matched_filtered=True,
        log=structlog.get_logger(),
    )
    return result, np.asarray(model)


@pytest.mark.parametrize(
    "center_dec,fwhm_major,fwhm_minor,position_angle",
    [
        (0.0, 2.2, 2.2, 0.0),  # point source
        (0.0, 3.0, 3.0, 0.0),  # extended source
        (0.0, 4.0, 2.2, 0.0),  # elongated source
        (0.0, 4.0, 2.2, 90.0),
        (0.0, 4.0, 2.2, 30.0),
        (0.0, 4.0, 2.2, 60.0),
        (0.0, 4.0, 2.2, 120.0),
        (0.0, 4.0, 2.2, 150.0),
        (-30.0, 2.2, 2.2, 0.0),  # pixels in RA are small on the sky
        (-50.0, 2.2, 2.2, 0.0),
    ],
)
def test_subtraction_model_matches_source(
    center_dec, fwhm_major, fwhm_minor, position_angle
):
    input_map, source, image = _map_with_gaussian(
        center_dec * u.deg, fwhm_major, fwhm_minor, position_angle
    )
    result, model = _fit_and_model(input_map, source)
    assert not result.fit_params["forced_center"]
    ## 2% of the peak; a model with the wrong width or angle leaves 10-45%
    assert np.abs(image - model).max() < 0.2


@pytest.mark.parametrize("center_dec", [0.0, -30.0, -50.0])
def test_fit_widths_and_flux_are_on_the_sky(center_dec):
    input_map, source, _ = _map_with_gaussian(center_dec * u.deg, 2.2, 2.2, 0.0)

    free, _ = _fit_and_model(input_map, source)
    assert free.fwhm_ra.to_value(u.arcmin) == pytest.approx(2.2, abs=0.05)
    assert free.fwhm_dec.to_value(u.arcmin) == pytest.approx(2.2, abs=0.05)
    assert free.flux.to_value(u.Jy) == pytest.approx(10.0, abs=0.1)

    ## the fixed beam is round on the sky, not round in map pixels
    fixed, _ = _fit_and_model(input_map, source, centroid_snr_limit=None)
    assert fixed.fit_params["forced_center"]
    assert fixed.flux.to_value(u.Jy) == pytest.approx(10.0, abs=0.1)


@pytest.mark.parametrize("center_dec", [0.0, -60.0])
@pytest.mark.parametrize("position_angle", [0.0, 30.0, 90.0, 120.0])
@pytest.mark.parametrize("reproject", [False, True])
def test_rotated_fit_widths_are_on_the_sky(center_dec, position_angle, reproject):
    input_map, source, image = _map_with_gaussian(
        center_dec * u.deg, 4.0, 2.2, position_angle
    )
    result, model = _fit_and_model(input_map, source, reproject=reproject)
    widths = sorted(
        [result.fwhm_ra.to_value(u.arcmin), result.fwhm_dec.to_value(u.arcmin)]
    )
    assert widths == pytest.approx([2.2, 4.0], abs=0.1)
    assert np.abs(image - model).max() < 0.2
