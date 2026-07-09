import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from pixell import enmap
from structlog import get_logger

from sotrplib.sims import sim_maps

log = get_logger("test_sim_maps")


def test_make_enmap_defaults(log=log):
    # Test default parameters
    test_map = sim_maps.make_enmap(log=log)
    assert isinstance(test_map, enmap.ndmap)
    assert np.all(test_map == 0.0)  ## noise is None by default


def test_make_enmap_shape_and_type(sim_map_params, log=log):
    test_map = sim_maps.make_enmap(**sim_map_params["maps"], log=log)
    assert isinstance(test_map, enmap.ndmap)
    # Check shape: width/resolution*2 (since width is +/-)
    expected_npix = int(
        2
        * sim_map_params["maps"]["width_ra"].to_value(u.deg)
        / sim_map_params["maps"]["resolution"].to_value(u.deg)
    )
    # Allow for rounding
    assert abs(test_map.shape[-2] - expected_npix) < 5
    expected_npix = int(
        2
        * sim_map_params["maps"]["width_dec"].to_value(u.deg)
        / sim_map_params["maps"]["resolution"].to_value(u.deg)
    )
    assert abs(test_map.shape[-1] - expected_npix) < 5


def test_make_enmap_wcs(sim_map_params, log=log):
    test_map = sim_maps.make_enmap(**sim_map_params["maps"], log=log)
    # Check WCS center
    y, x = np.array(test_map.shape) // 2
    dec, ra = enmap.pix2sky(test_map.shape, test_map.wcs, [y, x], safe=False)
    assert np.isclose(
        ra, sim_map_params["maps"]["center_ra"].to_value(u.rad), atol=1e-4
    )
    assert np.isclose(
        dec, sim_map_params["maps"]["center_dec"].to_value(u.rad), atol=1e-4
    )
    assert (
        u.Quantity(abs(test_map.wcs.wcs.cdelt[0]), test_map.wcs.wcs.cunit[0])
        == sim_map_params["maps"]["resolution"]
    )
    assert (
        u.Quantity(abs(test_map.wcs.wcs.cdelt[1]), test_map.wcs.wcs.cunit[1])
        == sim_map_params["maps"]["resolution"]
    )


def test_make_enmap_invalid_width(log=log):
    with pytest.raises(ValueError):
        sim_maps.make_enmap(width_ra=-1 * u.deg, log=log)
    with pytest.raises(ValueError):
        sim_maps.make_enmap(width_dec=-1 * u.deg, log=log)
    with pytest.raises(ValueError):
        sim_maps.make_enmap(width_dec=0 * u.deg, log=log)
    with pytest.raises(ValueError):
        sim_maps.make_enmap(width_ra=0 * u.deg, log=log)


def test_make_enmap_invalid_dec(log=log):
    with pytest.raises(ValueError):
        sim_maps.make_enmap(center_dec=89 * u.deg, width_dec=2 * u.deg, log=log)
    with pytest.raises(ValueError):
        sim_maps.make_enmap(center_dec=-89 * u.deg, width_dec=2 * u.deg, log=log)


def test_make_noise_map_properties(sim_map_params, log=log):
    test_map = sim_maps.make_enmap(**sim_map_params["maps"], log=log)
    noise_std = 1.0 * u.Jy
    noise_mean = 0.0 * u.Jy
    noise_map = sim_maps.make_noise_map(
        test_map, map_noise_Jy=noise_std, map_mean_Jy=noise_mean, seed=8675309, log=log
    )
    assert isinstance(noise_map, enmap.ndmap)
    assert noise_map.shape == test_map.shape
    # Check mean and std are within 1%
    assert np.isclose(np.mean(noise_map) * u.Jy, noise_mean, atol=0.01)
    assert np.isclose(np.std(noise_map) * u.Jy, noise_std, atol=0.01)


def test_make_noise_map_seed_behavior(sim_map_params, log=log):
    # Test that seed is used and warning is printed if not provided
    test_map = sim_maps.make_enmap(**sim_map_params["maps"], log=log)
    # Should print warning and still return a map
    noise_map = sim_maps.make_noise_map(
        test_map, map_noise_Jy=0.1 * u.Jy, map_mean_Jy=0.0 * u.Jy, seed=None, log=log
    )
    assert isinstance(noise_map, enmap.ndmap)


def test_photutils_sim_n_sources_invalid_input(log=log):
    # Test error on invalid sim_map type
    with pytest.raises(ValueError):
        sim_maps.photutils_sim_n_sources(sim_map="not_a_map", n_sources=1, log=log)
    # Test error on min_flux >= max_flux
    test_map = enmap.zeros((10, 10))
    with pytest.raises(ValueError):
        sim_maps.photutils_sim_n_sources(
            test_map,
            n_sources=1,
            min_flux_Jy=1.0 * u.Jy,
            max_flux_Jy=0.5 * u.Jy,
            log=log,
        )


def test_photutils_sim_n_sources_seed_behavior(sim_map_params, log=log):
    # Should print warning and still run if seed is None
    test_map = sim_maps.make_enmap(**sim_map_params["maps"], log=log)
    sim_maps.photutils_sim_n_sources(test_map, n_sources=1, seed=None, log=log)


def test_inject_sources_empty_and_not_flaring(
    sim_map_params, dummy_gaussian_source, log=log
):
    test_map = sim_maps.make_enmap(**sim_map_params["maps"], log=log)
    # Empty sources list
    out_map, injected = sim_maps.inject_sources(
        test_map, [], Time(0.0, format="unix"), log=log
    )
    assert isinstance(out_map, enmap.ndmap)
    assert injected == []
    # Not flaring: peak_time far from observation_time

    out_map, injected = sim_maps.inject_sources(
        test_map, [dummy_gaussian_source], Time(0.0, format="unix"), log=log
    )
    assert injected == []


def test_inject_sources_out_of_bounds(sim_map_params, dummy_fixed_source, log=log):
    test_map = sim_maps.make_enmap(**sim_map_params["maps"], log=log)
    # Place source out of bounds
    dummy_fixed_source._position = SkyCoord(ra=0.0 * u.deg, dec=90.0 * u.deg)
    _, injected = sim_maps.inject_sources(
        test_map, [dummy_fixed_source], Time(0.0, format="unix"), log=log
    )
    assert injected == []


def test_inject_sources_nan_mapval(sim_map_params, dummy_fixed_source, log=log):
    test_map = sim_maps.make_enmap(**sim_map_params["maps"], log=log)
    # Set a pixel to nan and inject source there
    dummy_fixed_source._position = SkyCoord(
        ra=sim_map_params["maps"]["center_ra"], dec=sim_map_params["maps"]["center_dec"]
    )
    x, y = test_map.sky2pix(
        [
            dummy_fixed_source.position().dec.to(u.rad).value,
            dummy_fixed_source.position().ra.to(u.rad).value,
        ]
    )
    test_map[int(x), int(y)] = np.nan
    _, injected = sim_maps.inject_sources(
        test_map, [dummy_fixed_source], Time(0.0, format="unix"), log=log
    )
    assert injected == []


def test_inject_random_sources_no_sim_params(sim_map_params, log=log):
    test_map = sim_maps.make_enmap(**sim_map_params["maps"], log=log)
    # Should return empty list if sim_params is None or empty
    assert sim_maps.inject_random_sources(test_map, None, "id", log=log) == []
    assert sim_maps.inject_random_sources(test_map, {}, "id", log=log) == []


def test_inject_simulated_sources_all_none(log=log):
    # Should return two empty lists if all inputs are None/empty
    from sotrplib.sims import sim_maps

    mapdata = type("MapData", (), {})()
    assert sim_maps.inject_simulated_sources(mapdata, log=log) == ([], [])


class TestSimulatedMapBboxSkyBox:
    """
    Tests for SimulatedMap.bbox / sky_box property interactions.

    Regression coverage for the infinite-recursion bug where bbox called
    self.sky_box which called self.bbox (fixed by reading _sky_box directly).
    """

    def _make_map(self, obs_times, sky_box=None, simulation_parameters=None):
        from sotrplib.sims.maps import SimulatedMap

        return SimulatedMap(
            observation_start=obs_times[0],
            observation_end=obs_times[1],
            sky_box=sky_box,
            simulation_parameters=simulation_parameters,
        )

    def test_no_flux_no_sky_box_no_recursion(self, obs_times):
        """Regression: accessing bbox and sky_box on a fresh map must not recurse."""
        sim = self._make_map(obs_times)
        _ = sim.bbox
        _ = sim.sky_box

    def test_no_flux_no_sky_box_bbox_uses_sim_params(self, obs_times):
        from sotrplib.sims.maps import SimulationParameters

        params = SimulationParameters(
            center_ra=16.0 * u.deg,
            center_dec=-2.0 * u.deg,
            width_ra=2.0 * u.deg,
            width_dec=1.0 * u.deg,
        )
        sim = self._make_map(obs_times, simulation_parameters=params)
        box = sim.bbox  # [[dec_min, ra_max], [dec_max, ra_min]] in radians
        assert np.isclose(box[0][0], np.deg2rad(-3.0), atol=1e-6)  # dec_min
        assert np.isclose(box[0][1], np.deg2rad(18.0), atol=1e-6)  # ra_max
        assert np.isclose(box[1][0], np.deg2rad(-1.0), atol=1e-6)  # dec_max
        assert np.isclose(box[1][1], np.deg2rad(14.0), atol=1e-6)  # ra_min

    def test_no_flux_no_sky_box_sky_box_derived_from_sim_params(self, obs_times):
        from sotrplib.sims.maps import SimulationParameters

        params = SimulationParameters(
            center_ra=16.0 * u.deg,
            center_dec=-2.0 * u.deg,
            width_ra=2.0 * u.deg,
            width_dec=1.0 * u.deg,
        )
        sim = self._make_map(obs_times, simulation_parameters=params)
        sky_box = sim.sky_box
        assert sky_box is not None
        # sky_box[0] = (ra_min, dec_min), sky_box[1] = (ra_max, dec_max)
        assert np.isclose(sky_box[0].ra.to_value(u.deg), 14.0, atol=1e-4)
        assert np.isclose(sky_box[0].dec.to_value(u.deg), -3.0, atol=1e-4)
        assert np.isclose(sky_box[1].ra.to_value(u.deg), 18.0, atol=1e-4)
        assert np.isclose(sky_box[1].dec.to_value(u.deg), -1.0, atol=1e-4)

    def test_explicit_sky_box_no_flux_bbox(self, obs_times):
        explicit = (
            SkyCoord(ra=10.0 * u.deg, dec=-5.0 * u.deg),
            SkyCoord(ra=20.0 * u.deg, dec=5.0 * u.deg),
        )
        sim = self._make_map(obs_times, sky_box=explicit)
        box = sim.bbox
        assert np.isclose(box[0][0], np.deg2rad(-5.0), atol=1e-6)  # dec_min
        assert np.isclose(box[0][1], np.deg2rad(20.0), atol=1e-6)  # ra_max
        assert np.isclose(box[1][0], np.deg2rad(5.0), atol=1e-6)  # dec_max
        assert np.isclose(box[1][1], np.deg2rad(10.0), atol=1e-6)  # ra_min

    def test_explicit_sky_box_no_flux_sky_box_returns_explicit(self, obs_times):
        explicit = (
            SkyCoord(ra=10.0 * u.deg, dec=-5.0 * u.deg),
            SkyCoord(ra=20.0 * u.deg, dec=5.0 * u.deg),
        )
        sim = self._make_map(obs_times, sky_box=explicit)
        sky_box = sim.sky_box
        assert np.isclose(sky_box[0].ra.to_value(u.deg), 10.0, atol=1e-4)
        assert np.isclose(sky_box[0].dec.to_value(u.deg), -5.0, atol=1e-4)
        assert np.isclose(sky_box[1].ra.to_value(u.deg), 20.0, atol=1e-4)
        assert np.isclose(sky_box[1].dec.to_value(u.deg), 5.0, atol=1e-4)

    def test_flux_set_bbox_uses_flux(self, obs_times, flux_enmap):
        sim = self._make_map(obs_times)
        sim.flux = flux_enmap
        expected = enmap.box(flux_enmap.shape, flux_enmap.wcs)
        assert np.allclose(sim.bbox, expected)

    def test_flux_set_sky_box_derived_from_flux(self, obs_times, flux_enmap):
        sim = self._make_map(obs_times)
        sim.flux = flux_enmap
        flux_box = enmap.box(flux_enmap.shape, flux_enmap.wcs)
        sky_box = sim.sky_box
        assert sky_box is not None
        # ra_min lives at flux_box[1][1], dec_min at flux_box[0][0]
        assert np.isclose(sky_box[0].ra.to_value(u.rad), flux_box[1][1], atol=1e-4)
        assert np.isclose(sky_box[0].dec.to_value(u.rad), flux_box[0][0], atol=1e-4)
        assert np.isclose(sky_box[1].ra.to_value(u.rad), flux_box[0][1], atol=1e-4)
        assert np.isclose(sky_box[1].dec.to_value(u.rad), flux_box[1][0], atol=1e-4)

    def test_flux_set_explicit_sky_box_preserved(self, obs_times, flux_enmap):
        """Explicit sky_box is not overridden when flux is set."""
        explicit = (
            SkyCoord(ra=10.0 * u.deg, dec=-5.0 * u.deg),
            SkyCoord(ra=20.0 * u.deg, dec=5.0 * u.deg),
        )
        sim = self._make_map(obs_times, sky_box=explicit)
        sim.flux = flux_enmap
        # bbox should use flux
        assert np.allclose(sim.bbox, enmap.box(flux_enmap.shape, flux_enmap.wcs))
        # sky_box should still be the explicit one
        sky_box = sim.sky_box
        assert np.isclose(sky_box[0].ra.to_value(u.deg), 10.0, atol=1e-4)
        assert np.isclose(sky_box[1].ra.to_value(u.deg), 20.0, atol=1e-4)
