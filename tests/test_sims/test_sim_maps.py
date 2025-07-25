import numpy as np
import pytest
from pixell import enmap

from sotrplib.sims import sim_maps


def test_make_enmap_defaults():
    # Test default parameters
    test_map = sim_maps.make_enmap()
    assert isinstance(test_map, enmap.ndmap)
    assert np.all(test_map == 0.0)  ## noise is None by default


def test_make_enmap_shape_and_type(sim_map_params):
    test_map = sim_maps.make_enmap(**sim_map_params["maps"])
    assert isinstance(test_map, enmap.ndmap)
    # Check shape: width/resolution*2 (since width is +/-)
    expected_npix = int(
        2
        * sim_map_params["maps"]["width_ra"]
        / sim_map_params["maps"]["resolution"]
        * 60
    )
    # Allow for rounding
    assert abs(test_map.shape[-2] - expected_npix) < 5
    expected_npix = int(
        2
        * sim_map_params["maps"]["width_dec"]
        / sim_map_params["maps"]["resolution"]
        * 60
    )
    assert abs(test_map.shape[-1] - expected_npix) < 5


def test_make_enmap_wcs(sim_map_params):
    from pixell.utils import arcmin, degree

    test_map = sim_maps.make_enmap(**sim_map_params["maps"])
    # Check WCS center
    y, x = np.array(test_map.shape) // 2
    dec, ra = enmap.pix2sky(test_map.shape, test_map.wcs, [y, x], safe=False)
    assert np.isclose(ra, sim_map_params["maps"]["center_ra"] * degree, atol=1e-4)
    assert np.isclose(dec, sim_map_params["maps"]["center_dec"] * degree, atol=1e-4)
    assert (
        abs(test_map.wcs.wcs.cdelt[0])
        == sim_map_params["maps"]["resolution"] * arcmin / degree
    )
    assert (
        abs(test_map.wcs.wcs.cdelt[1])
        == sim_map_params["maps"]["resolution"] * arcmin / degree
    )


def test_make_enmap_invalid_width():
    with pytest.raises(ValueError):
        sim_maps.make_enmap(width_ra=-1)
    with pytest.raises(ValueError):
        sim_maps.make_enmap(width_dec=-1)
    with pytest.raises(ValueError):
        sim_maps.make_enmap(width_dec=0)
    with pytest.raises(ValueError):
        sim_maps.make_enmap(width_ra=0)


def test_make_enmap_invalid_dec():
    with pytest.raises(ValueError):
        sim_maps.make_enmap(center_dec=89, width_dec=2)
    with pytest.raises(ValueError):
        sim_maps.make_enmap(center_dec=-89, width_dec=2)


def test_make_noise_map_properties(sim_map_params):
    test_map = sim_maps.make_enmap(**sim_map_params["maps"])
    noise_std = 1.0
    noise_mean = 0.0
    noise_map = sim_maps.make_noise_map(
        test_map, map_noise_Jy=noise_std, map_mean_Jy=noise_mean, seed=8675309
    )
    assert isinstance(noise_map, enmap.ndmap)
    assert noise_map.shape == test_map.shape
    # Check mean and std are within 1%
    assert np.isclose(np.mean(noise_map), noise_mean, atol=0.01)
    assert np.isclose(np.std(noise_map), noise_std, atol=0.01)


def test_make_noise_map_seed_behavior(sim_map_params):
    # Test that seed is used and warning is printed if not provided
    test_map = sim_maps.make_enmap(**sim_map_params["maps"])
    # Should print warning and still return a map
    noise_map = sim_maps.make_noise_map(
        test_map, map_noise_Jy=0.1, map_mean_Jy=0.0, seed=None
    )
    assert isinstance(noise_map, enmap.ndmap)


def test_photutils_sim_n_sources_invalid_input():
    # Test error on invalid sim_map type
    with pytest.raises(ValueError):
        sim_maps.photutils_sim_n_sources(sim_map="not_a_map", n_sources=1)
    # Test error on min_flux >= max_flux
    test_map = enmap.zeros((10, 10))
    with pytest.raises(ValueError):
        sim_maps.photutils_sim_n_sources(
            test_map, n_sources=1, min_flux_Jy=1.0, max_flux_Jy=0.5
        )


def test_photutils_sim_n_sources_seed_behavior(sim_map_params):
    # Should print warning and still run if seed is None
    test_map = sim_maps.make_enmap(**sim_map_params["maps"])
    sim_maps.photutils_sim_n_sources(test_map, n_sources=1, seed=None)


def test_inject_sources_empty_and_not_flaring(sim_map_params, dummy_source):
    test_map = sim_maps.make_enmap(**sim_map_params["maps"])
    # Empty sources list
    out_map, injected = sim_maps.inject_sources(test_map, [], 0.0)
    assert isinstance(out_map, enmap.ndmap)
    assert injected == []
    # Not flaring: peak_time far from observation_time
    dummy_source.peak_time = 0.0
    dummy_source.flare_width = 1.0
    out_map, injected = sim_maps.inject_sources(test_map, [dummy_source], 100.0)
    assert injected == []


def test_inject_sources_out_of_bounds(sim_map_params, dummy_source):
    test_map = sim_maps.make_enmap(**sim_map_params["maps"])
    # Place source out of bounds
    dummy_source.ra = 999.0
    dummy_source.dec = 999.0
    _, injected = sim_maps.inject_sources(test_map, [dummy_source], 0.0)
    assert injected == []


def test_inject_sources_nan_mapval(sim_map_params, dummy_source):
    from pixell.utils import degree

    test_map = sim_maps.make_enmap(**sim_map_params["maps"])
    # Set a pixel to nan and inject source there
    dummy_source.ra = sim_map_params["maps"]["center_ra"]
    dummy_source.dec = sim_map_params["maps"]["center_dec"]
    x, y = test_map.sky2pix([dummy_source.dec * degree, dummy_source.ra * degree])
    test_map[int(x), int(y)] = np.nan
    _, injected = sim_maps.inject_sources(test_map, [dummy_source], 0.0)
    assert injected == []


def test_inject_sources_from_db_empty(tmp_path, sim_map_params):
    # Create a dummy SourceCatalogDatabase with no sources
    class DummyDB:
        def read_database(self):
            return []

    test_map = sim_maps.make_enmap(**sim_map_params["maps"])
    mapdata = type("MapData", (), {})()
    mapdata.time_map = test_map
    mapdata.freq = "f090"
    mapdata.wafer_name = "sim"
    mapdata.map_id = "test"
    injected = sim_maps.inject_sources_from_db(mapdata, DummyDB())
    assert injected == []


def test_inject_random_sources_no_sim_params(sim_map_params):
    test_map = sim_maps.make_enmap(**sim_map_params["maps"])
    # Should return empty list if sim_params is None or empty
    assert sim_maps.inject_random_sources(test_map, None, "id") == []
    assert sim_maps.inject_random_sources(test_map, {}, "id") == []


def test_inject_simulated_sources_all_none():
    # Should return two empty lists if all inputs are None/empty
    from sotrplib.sims import sim_maps

    mapdata = type("MapData", (), {})()
    assert sim_maps.inject_simulated_sources(mapdata) == ([], [])
