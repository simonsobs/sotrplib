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
from sotrplib.sources.blind import BlindSearchParameters, SigmaClipBlindSearch
from sotrplib.sources.force import TwoDGaussianFitter
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
    phot = TwoDGaussianFitter(mode="lmfit", thumbnail_half_width=4 * u.arcmin)
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


def test_source_injection_blind_search():
    min_flux = u.Quantity(1.0, "Jy")
    max_flux = u.Quantity(1.1, "Jy")
    map_sim_params = maps.SimulationParameters()
    left = map_sim_params.center_ra - map_sim_params.width_ra / 2
    right = map_sim_params.center_ra + map_sim_params.width_ra / 2
    bottom = map_sim_params.center_dec - map_sim_params.width_dec / 2
    top = map_sim_params.center_dec + map_sim_params.width_dec / 2
    map_sim_params.map_noise = u.Quantity(0.001, "Jy")
    start_time = datetime.datetime.fromisoformat("2025-10-01T00:00:00+00:00")
    number = 4

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
    blind_search = SigmaClipBlindSearch(
        parameters=BlindSearchParameters(
            sigma_threshold=5.0,
            minimum_separation=[1.5 * u.arcmin],
            sigma_threshold_for_minimum_separation=[2.0],
        ),
        thumbnail_half_width=3 * u.arcmin,
    )
    found_sources, _ = blind_search.search(new_map)
    assert len(found_sources) == number

    blind_search = SigmaClipBlindSearch(
        parameters=BlindSearchParameters(
            sigma_threshold=5.0,
            minimum_separation=[1.5 * u.arcmin],
            sigma_threshold_for_minimum_separation=[2.0],
        ),
    )
    found_sources, _ = blind_search.search(new_map)
    assert len(found_sources) == number


def test_source_injection_blind_search_SAT():
    min_flux = u.Quantity(1.0, "Jy")
    max_flux = u.Quantity(1.1, "Jy")
    map_sim_params = maps.SimulationParameters()
    left = map_sim_params.center_ra - map_sim_params.width_ra / 2
    right = map_sim_params.center_ra + map_sim_params.width_ra / 2
    bottom = map_sim_params.center_dec - map_sim_params.width_dec / 2
    top = map_sim_params.center_dec + map_sim_params.width_dec / 2
    map_sim_params.map_noise = u.Quantity(0.1, "Jy")
    start_time = datetime.datetime.fromisoformat("2025-10-01T00:00:00+00:00")
    number = 1

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
        array=None,
        instrument="SOSAT",
        simulation_parameters=map_sim_params,
        box=[SkyCoord(ra=left, dec=bottom), SkyCoord(ra=right, dec=top)],
    )

    base_map.build()
    base_map.finalize()

    fwhm = get_fwhm(
        arr=base_map.array, freq=base_map.frequency, instrument=base_map.instrument
    )

    injector = source_injector.PhotutilsSourceInjector(gauss_fwhm=fwhm)
    _, new_map = injector.inject(input_map=base_map, simulated_sources=sources)

    blind_search = SigmaClipBlindSearch(
        parameters=BlindSearchParameters(
            sigma_threshold=5.0,
            minimum_separation=[1.5 * fwhm],
            sigma_threshold_for_minimum_separation=[2.0],
        ),
        thumbnail_half_width=1 * u.deg,
    )
    found_sources, _ = blind_search.search(new_map)
    assert len(found_sources) == number

    blind_search = SigmaClipBlindSearch(
        parameters=BlindSearchParameters(
            sigma_threshold=5.0,
            minimum_separation=[1.5 * fwhm],
            sigma_threshold_for_minimum_separation=[2.0],
            thumbnail_half_width=None,
        ),
    )
    found_sources, _ = blind_search.search(new_map)
    assert len(found_sources) == number
