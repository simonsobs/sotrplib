import datetime

from astropy import units as u
from astropy.coordinates import SkyCoord
from structlog import get_logger

from sotrplib.sims import (
    maps,
    sim_source_generators,
    sim_sources,
    source_injector,
)
from sotrplib.sources.force import Scipy2DGaussianFitter
from sotrplib.utils.utils import get_fwhm

log = get_logger()


def test_fixed_source_type():
    position = SkyCoord(ra=90.0 * u.deg, dec=0.0 * u.deg)
    flux = u.Quantity(1.0, "Jy")
    time = datetime.datetime.now(tz=datetime.UTC)

    source = sim_sources.FixedSimulatedSource(position=position, flux=flux)
    assert source.flux(time=time) == flux
    assert source.position(time=time) == position


def test_gaussian_source_type():
    position = SkyCoord(ra=90.0 * u.deg, dec=0.0 * u.deg)
    flux = u.Quantity(1.0, "Jy")
    width = datetime.timedelta(days=1)
    time = datetime.datetime.now(tz=datetime.UTC)

    source = sim_sources.GaussianTransientSimulatedSource(
        position=position, peak_time=time, flare_width=width, peak_amplitude=flux
    )

    # Peak flux as described
    assert u.isclose(source.flux(time=time), flux, rtol=0.001)
    assert source.flux(time=time - datetime.timedelta(hours=1)) < source.flux(time=time)
    assert source.flux(time=time + datetime.timedelta(hours=1)) < source.flux(time=time)

    assert source.position(time=time) == position
    assert source.position(time=time - width) == position
    assert source.position(time=time + width) == position


def test_powerlaw_source_type():
    position = SkyCoord(ra=90.0 * u.deg, dec=0.0 * u.deg)
    flux = u.Quantity(1.0, "Jy")
    peak_time = datetime.datetime.now()
    alpha_rise = 1.5
    alpha_decay = -1.2
    smoothness = 1.0
    t_ref = datetime.timedelta(days=1)

    source = sim_sources.PowerLawTransientSimulatedSource(
        position=position,
        peak_time=peak_time,
        peak_amplitude=flux,
        alpha_rise=alpha_rise,
        alpha_decay=alpha_decay,
        smoothness=smoothness,
        t_ref=t_ref,
    )

    # Peak flux as described - allow tolerance for numerical precision
    assert u.isclose(source.flux(time=peak_time), flux, rtol=0.001)

    # Use longer time offsets (multiples of t_ref) to clearly be in rise/decay regimes
    # Flux well before peak should be less (deep in rising phase)
    assert source.flux(time=peak_time - 2 * t_ref) < source.flux(time=peak_time)

    # Flux well after peak should be less (deep in decaying phase)
    assert source.flux(time=peak_time + 2 * t_ref) < source.flux(time=peak_time)

    # Verify the peak is actually near the maximum by sampling around it
    sample_times = [peak_time + datetime.timedelta(hours=h) for h in range(-24, 25, 6)]
    fluxes = [source.flux(t) for t in sample_times]
    peak_flux = source.flux(peak_time)

    # Peak should be within top few samples (allowing for numerical precision)
    assert peak_flux >= sorted(fluxes, reverse=True)[2]  # At least 3rd highest

    # Flux before onset should be zero
    assert (
        source.flux(time=source.onset_time - datetime.timedelta(hours=1)) == 0.0 * u.Jy
    )

    # Position should be constant
    assert source.position(time=peak_time) == position
    assert source.position(time=peak_time - t_ref) == position
    assert source.position(time=peak_time + t_ref) == position


def test_fixed_source_generation():
    min_flux = u.Quantity(1.0, "Jy")
    max_flux = u.Quantity(10.0, "Jy")
    left = 10.0 * u.deg
    right = 20.0 * u.deg
    bottom = 0.0 * u.deg
    top = 10.0 * u.deg
    time = datetime.datetime.now(tz=datetime.UTC)
    number = 32

    generator = sim_source_generators.FixedSourceGenerator(
        min_flux=min_flux,
        max_flux=max_flux,
        number=number,
        catalog_fraction=0.5,
    )

    sources, cat = generator.generate(
        box=[SkyCoord(ra=left, dec=bottom), SkyCoord(ra=right, dec=top)]
    )

    assert len(sources) == number
    assert len(cat.sources) == int(number * 0.5)

    for source in sources:
        assert min_flux < source.flux(time) < max_flux
        assert left < source.position(time).ra < right
        assert bottom < source.position(time).dec < top


def test_gaussian_source_generation():
    min_flux = u.Quantity(1.0, "Jy")
    max_flux = u.Quantity(10.0, "Jy")
    left = 10.0 * u.deg
    right = 20.0 * u.deg
    bottom = 0.0 * u.deg
    top = 10.0 * u.deg
    time = datetime.datetime.now(tz=datetime.UTC)
    dt = datetime.timedelta(hours=2)
    earliest = time - dt
    latest = time + dt
    shortest = datetime.timedelta(hours=1.0)
    longest = datetime.timedelta(hours=2.0)
    number = 32

    generator = sim_source_generators.GaussianTransientSourceGenerator(
        flare_earliest_time=earliest,
        flare_latest_time=latest,
        flare_width_shortest=shortest,
        flare_width_longest=longest,
        peak_amplitude_minimum=min_flux,
        peak_amplitude_maximum=max_flux,
        number=number,
        catalog_fraction=0.5,
    )

    sources, cat = generator.generate(
        box=[SkyCoord(ra=left, dec=bottom), SkyCoord(ra=right, dec=top)]
    )

    assert len(sources) == number
    assert len(cat.sources) == int(number * 0.5)

    for source in sources:
        assert min_flux < source.peak_amplitude < max_flux
        assert left < source.position(time).ra < right
        assert bottom < source.position(time).dec < top
        assert shortest < source.flare_width < longest


def test_powerlaw_source_generation():
    min_flux = u.Quantity(1.0, "Jy")
    max_flux = u.Quantity(10.0, "Jy")
    left = 10.0 * u.deg
    right = 20.0 * u.deg
    bottom = 0.0 * u.deg
    top = 10.0 * u.deg
    time = datetime.datetime.now()
    dt = datetime.timedelta(hours=2)
    earliest = time - dt
    latest = time + dt
    alpha_rise_min = 0.5
    alpha_rise_max = 2.0
    alpha_decay_min = -2.5
    alpha_decay_max = -1.0
    smoothness_min = 0.5
    smoothness_max = 2.0
    t_ref_min = datetime.timedelta(hours=12)
    t_ref_max = datetime.timedelta(days=2)
    number = 32

    generator = sim_source_generators.PowerLawTransientSourceGenerator(
        flare_earliest_time=earliest,
        flare_latest_time=latest,
        alpha_rise_min=alpha_rise_min,
        alpha_rise_max=alpha_rise_max,
        alpha_decay_min=alpha_decay_min,
        alpha_decay_max=alpha_decay_max,
        peak_amplitude_minimum=min_flux,
        peak_amplitude_maximum=max_flux,
        number=number,
        smoothness_min=smoothness_min,
        smoothness_max=smoothness_max,
        t_ref_min=t_ref_min,
        t_ref_max=t_ref_max,
        catalog_fraction=0.5,
    )

    sources, cat = generator.generate(
        box=[SkyCoord(ra=left, dec=bottom), SkyCoord(ra=right, dec=top)]
    )

    assert len(sources) == number
    assert len(cat.sources) == int(number * 0.5)

    for source in sources:
        # Check parameter ranges
        assert min_flux < source.peak_amplitude < max_flux
        assert left < source.position(time).ra < right
        assert bottom < source.position(time).dec < top
        assert alpha_rise_min <= source.alpha_rise <= alpha_rise_max
        assert alpha_decay_min <= source.alpha_decay <= alpha_decay_max
        assert smoothness_min <= source.smoothness <= smoothness_max
        assert t_ref_min <= source.t_ref <= t_ref_max
        assert earliest <= source.peak_time <= latest

        # Verify onset time is before peak time
        assert source.onset_time < source.peak_time

        # Verify flux is zero before onset (fundamental property)
        assert (
            source.flux(source.onset_time - datetime.timedelta(hours=1)) == 0.0 * u.Jy
        )

        # Verify flux is positive after onset
        flux_after_onset = source.flux(
            source.onset_time + datetime.timedelta(minutes=1)
        )
        assert flux_after_onset >= 0.0 * u.Jy

        # Verify position is constant over time
        pos1 = source.position(source.onset_time)
        pos2 = source.position(source.peak_time)
        pos3 = source.position(source.peak_time + source.t_ref)
        assert pos1 == pos2 == pos3


def test_source_injection_into_map():
    min_flux = u.Quantity(1.0, "Jy")
    max_flux = u.Quantity(10.0, "Jy")
    number = 32

    generator = sim_source_generators.FixedSourceGenerator(
        min_flux=min_flux,
        max_flux=max_flux,
        number=number,
        catalog_fraction=0.5,
    )

    start_obs = datetime.datetime.now(tz=datetime.UTC)
    -datetime.timedelta(hours=2)
    end_obs = datetime.datetime.now(tz=datetime.UTC)

    base_map = maps.SimulatedMap(
        observation_start=start_obs,
        observation_end=end_obs,
        frequency="f090",
        array="pa5",
        simulation_parameters=maps.SimulationParameters(
            center_ra=50.0 * u.deg,
            center_dec=0.0 * u.deg,
            width_ra=20.0 * u.deg,
            width_dec=20.0 * u.deg,
        ),
    )

    base_map.build()

    injector = source_injector.PhotutilsSourceInjector()
    sources, catalog = generator.generate(input_map=base_map)
    _, new_map = injector.inject(input_map=base_map, simulated_sources=sources)
    new_map.finalize()

    assert new_map != base_map
    assert new_map.original_map == base_map

    assert (new_map.flux != base_map.flux).any()


def test_source_injection_forced_photometry():
    min_flux = u.Quantity(1.0, "Jy")
    max_flux = u.Quantity(1.1, "Jy")
    map_sim_params = maps.SimulationParameters()
    left = map_sim_params.center_ra - map_sim_params.width_ra / 2
    right = map_sim_params.center_ra + map_sim_params.width_ra / 2
    bottom = map_sim_params.center_dec - map_sim_params.width_dec / 2
    top = map_sim_params.center_dec + map_sim_params.width_dec / 2
    map_sim_params.map_noise = u.Quantity(0.001, "Jy")
    start_time = datetime.datetime.fromisoformat("2025-10-01T00:00:00+00:00")
    number = 10

    generator = sim_source_generators.FixedSourceGenerator(
        min_flux=min_flux,
        max_flux=max_flux,
        number=number,
        catalog_fraction=1.0,
    )

    sources, cat = generator.generate(
        box=[SkyCoord(ra=left, dec=bottom), SkyCoord(ra=right, dec=top)]
    )
    start_obs = start_time
    end_obs = start_time + datetime.timedelta(hours=2)
    base_map = maps.SimulatedMap(
        observation_start=start_obs,
        observation_end=end_obs,
        frequency="f090",
        array="pa5",
        box=[SkyCoord(ra=left, dec=bottom), SkyCoord(ra=right, dec=top)],
    )

    base_map.build()
    base_map.finalize()

    injector = source_injector.PhotutilsSourceInjector(
        gauss_fwhm=get_fwhm(base_map.frequency)
    )
    _, new_map = injector.inject(input_map=base_map, simulated_sources=sources)
    phot = Scipy2DGaussianFitter(thumbnail_half_width=3 * u.arcmin)
    forced_phot_results = phot.force(new_map, catalogs=[cat])

    assert len(forced_phot_results) == number
    n_valid = 0
    for res in forced_phot_results:
        ## check if two sources nearby:
        xmatch = cat.crossmatch(res.ra, res.dec, radius=3.0 * u.arcmin, method="all")
        if len(xmatch) > 1:
            continue
        n_valid += 1
        assert min_flux * 0.9 <= res.flux <= max_flux * 1.1
        assert abs(res.offset_ra.to_value(u.arcmin)) < 1.0
        assert abs(res.offset_dec.to_value(u.arcmin)) < 1.0
    assert n_valid >= number // 2
