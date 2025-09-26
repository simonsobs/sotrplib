import datetime

import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from pixell import enmap
from structlog import get_logger

from sotrplib.sims import (
    maps,
    sim_maps,
    sim_source_generators,
    sim_sources,
    source_injector,
)

log = get_logger()


def test_fixed_source_type():
    position = SkyCoord(ra=90.0 * u.deg, dec=0.0 * u.deg)
    flux = u.Quantity(1.0, "Jy")
    time = datetime.datetime.now()

    source = sim_sources.FixedSimulatedSource(position=position, flux=flux)
    assert source.flux(time=time) == flux
    assert source.position(time=time) == position


def test_gaussian_source_type():
    position = SkyCoord(ra=90.0 * u.deg, dec=0.0 * u.deg)
    flux = u.Quantity(1.0, "Jy")
    width = datetime.timedelta(days=1)
    time = datetime.datetime.now()

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
    time = datetime.datetime.now()
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
    time = datetime.datetime.now()
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

    start_obs = datetime.datetime.now() - datetime.timedelta(hours=2)
    end_obs = datetime.datetime.now()

    base_map = maps.SimulatedMap(
        observation_start=start_obs,
        observation_end=end_obs,
        frequency="f090",
        array="pa5",
    )

    base_map.build()

    # Grab the geometry -- TODO actually have this as a property
    # of maps...
    sp = base_map.simulation_parameters
    left = sp.center_ra - sp.width_ra
    right = sp.center_ra + sp.width_ra
    bottom = sp.center_dec - sp.width_dec
    top = sp.center_dec + sp.width_dec

    box = [
        SkyCoord(ra=left, dec=bottom),
        SkyCoord(ra=right, dec=top),
    ]

    injector = source_injector.PhotutilsSourceInjector()
    sources, catalog = generator.generate(box=box)
    new_map = injector.inject(input_map=base_map, simulated_sources=sources)

    new_map.finalize()

    assert new_map != base_map
    assert new_map.original_map == base_map

    assert (new_map.flux != base_map.flux).any()

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2)

    ax[0].imshow(new_map.flux, vmin=0.0, vmax=0.1)
    ax[1].imshow(base_map.flux, vmin=0.0, vmax=0.1)

    fig.savefig("test.png")


def test_photutils_sim_n_sources_output_a(sim_map_params, log=log):
    test_map = sim_maps.make_enmap(**sim_map_params["maps"], log=log)
    n_sources = 5
    out_map, injected_sources = sim_maps.photutils_sim_n_sources(
        test_map,
        n_sources=n_sources,
        min_flux_Jy=0.2 * u.Jy,
        max_flux_Jy=0.5 * u.Jy,
        map_noise_Jy=0.01 * u.Jy,
        seed=8675309,
        log=log,
    )
    # Check output types
    assert isinstance(out_map, enmap.ndmap)
    assert isinstance(injected_sources, list)
    assert len(injected_sources) == n_sources


def test_photutils_sim_n_sources_output_b(dummy_map, log=log):
    n_sources = 5
    out_map, injected_sources = sim_maps.photutils_sim_n_sources(
        dummy_map,
        n_sources=n_sources,
        min_flux_Jy=0.2 * u.Jy,
        max_flux_Jy=0.5 * u.Jy,
        map_noise_Jy=0.01 * u.Jy,
        seed=8675309,
        log=log,
    )
    # Check output types
    assert isinstance(out_map, type(dummy_map))
    assert isinstance(injected_sources, list)
    assert len(injected_sources) == n_sources


def test_photutils_sim_n_sources_flux_range(sim_map_params, log=log):
    test_map = sim_maps.make_enmap(**sim_map_params["maps"], log=log)
    n_sources = 3
    min_flux = 1.0 * u.Jy
    max_flux = 1.01 * u.Jy
    map_noise = 0.001 * u.Jy
    _, injected_sources = sim_maps.photutils_sim_n_sources(
        test_map,
        n_sources=n_sources,
        min_flux_Jy=min_flux,
        max_flux_Jy=max_flux,
        map_noise_Jy=map_noise,
        seed=8675309,
        log=log,
    )
    for src in injected_sources:
        assert min_flux <= src.flux <= max_flux


def test_photutils_sim_n_sources_invalid_flux(sim_map_params, log=log):
    test_map = sim_maps.make_enmap(**sim_map_params["maps"], log=log)
    with pytest.raises(ValueError):
        sim_maps.photutils_sim_n_sources(
            test_map,
            n_sources=2,
            min_flux_Jy=1.0 * u.Jy,
            max_flux_Jy=0.5 * u.Jy,
            log=log,
        )


def test_inject_sources_empty(sim_map_params, log=log):
    test_map = sim_maps.make_enmap(**sim_map_params["maps"], log=log)
    out_map, injected = sim_maps.inject_sources(test_map, [], 0.0, log=log)
    assert isinstance(out_map, enmap.ndmap)
    assert injected == []


def test_inject_sources_bad_maptype(log=log):
    from sotrplib.sims import sim_maps

    with pytest.raises(ValueError):
        sim_maps.inject_sources("not_a_map", ["fake_source"], 0.0, log=log)


def test_inject_sources_out_of_bounds(sim_map_params, dummy_source, log=log):
    test_map = sim_maps.make_enmap(**sim_map_params["maps"], log=log)
    oob_source = dummy_source
    center_ra, center_dec = (
        sim_map_params["maps"]["center_ra"],
        sim_map_params["maps"]["center_dec"],
    )
    ra_width = test_map.shape[-2] * test_map.wcs.wcs.cdelt[0] * 60  # arcmin
    dec_width = test_map.shape[-1] * test_map.wcs.wcs.cdelt[1] * 60  # arcmin
    oob_source.ra = center_ra + ra_width * u.arcmin  # Out of bounds
    oob_source.dec = center_dec + dec_width * u.arcmin  # Out of bounds
    _, injected = sim_maps.inject_sources(test_map, [oob_source], 0.0, log=log)
    assert injected == []


def test_inject_sources_not_flaring(sim_map_params, dummy_source, log=log):
    test_map = sim_maps.make_enmap(**sim_map_params["maps"], log=log)
    # observation_time far from peak_time
    quiescent_source = dummy_source
    quiescent_source.ra = sim_map_params["maps"]["center_ra"]
    quiescent_source.dec = sim_map_params["maps"]["center_dec"]
    ## inject source with width 1. but 10 days after peak time
    _, injected = sim_maps.inject_sources(test_map, [quiescent_source], 10.0, log=log)
    assert injected == []


def test_inject_simulated_sources_all_none(log=log):
    from sotrplib.sims import sim_maps

    # Dummy mapdata object
    class DummyMap:
        pass

    mapdata = DummyMap()
    # All inputs None/empty
    result = sim_maps.inject_simulated_sources(mapdata, log=log)
    assert result == ([], [])


def test_inject_simulated_sources_random_only(
    dummy_map, sim_map_params, sim_source_params, log=log
):
    from sotrplib.sims import sim_maps

    # Minimal sim_params for random injection
    sim_params = {
        **sim_map_params,
        **sim_source_params,
    }
    # Should return a list of injected sources, no transients
    catalog_sources, injected_sources = sim_maps.inject_simulated_sources(
        dummy_map, sim_params=sim_params, verbose=False, log=log
    )
    assert isinstance(catalog_sources, list)
    assert isinstance(injected_sources, list)
    assert (
        len(catalog_sources) == sim_params["injected_sources"]["n_sources"]
    )  # From random injection
    assert len(injected_sources) == 0  # No transients injected


def test_inject_simulated_sources_use_geometry(
    dummy_map, sim_map_params, sim_source_params, log=log
):
    from sotrplib.sims import sim_maps

    # Minimal sim_params for random injection
    sim_params = {
        **sim_map_params,
        **sim_source_params,
    }
    # Should return a list of injected sources, no transients
    catalog_sources, injected_sources = sim_maps.inject_simulated_sources(
        dummy_map,
        sim_params=sim_params,
        use_map_geometry=True,
        verbose=False,
        log=log,
    )
    assert isinstance(catalog_sources, list)
    assert isinstance(injected_sources, list)
    assert (
        len(catalog_sources) == sim_params["injected_sources"]["n_sources"]
    )  # From random injection
    assert len(injected_sources) == 0  # No transients injected


def test_generate_transients_empty(log=log):
    from sotrplib.sims import sim_sources

    # Should return an empty list if no transients are specified
    transients = sim_sources.generate_transients(
        n=1,
        ra_lims=(1, 3) * u.deg,
        dec_lims=(-3, -1) * u.deg,
        peak_amplitudes=None,
        peak_times=None,
        flare_widths=None,
        imap=None,
        uniform_on_sky=True,
        log=log,
    )
    assert isinstance(transients, list)
    assert len(transients) == 1


def test_generate_transients_imap(sim_map_params, log=log):
    from sotrplib.sims import sim_sources

    # Should return an empty list if no transients are specified
    m = sim_maps.make_enmap(**sim_map_params["maps"], log=log)
    transients = sim_sources.generate_transients(
        n=1,
        imap=m,
        log=log,
    )
    assert isinstance(transients, list)
    assert len(transients) == 1


def test_generate_transients_value_error(log=log):
    from sotrplib.sims import sim_sources

    # Should raise ValueError if peak_amplitudes is None but n > 0
    with pytest.raises(ValueError):
        sim_sources.generate_transients(
            n=None,
            positions=None,
            log=log,
        )
        sim_sources.generate_transients(
            n=1,
            positions=[1 * u.deg, 2 * u.deg],
            log=log,
        )


def test_inject_simulated_sources_with_db_and_transients(
    dummy_map, sim_map_params, sim_transient_params, sim_source_params, log=log
):
    from sotrplib.sims import sim_maps

    # Minimal sim_params for random injection
    sim_params = {**sim_map_params, **sim_transient_params, **sim_source_params}

    catalog_sources, injected_sources = sim_maps.inject_simulated_sources(
        dummy_map,
        injected_source_db=None,
        sim_params=sim_params,
        inject_transients=True,
        use_map_geometry=False,
        verbose=False,
        log=log,
    )
    assert isinstance(catalog_sources, list)
    assert isinstance(injected_sources, list)
    assert len(catalog_sources) == 2  # From injected_source_db
    assert len(injected_sources) == 1  # From random injection
