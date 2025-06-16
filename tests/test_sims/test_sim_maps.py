import pytest
import numpy as np
from pixell import enmap
from sotrplib.sims import sim_maps

@pytest.fixture
def sim_map_params():
    return {
        "center_ra": 32.0,
        "center_dec": -51.0,
        "width_ra": 1.0,
        "width_dec": 1.0,
        "resolution": 0.5,
    }

def test_make_enmap_shape_and_type(sim_map_params):
    test_map = sim_maps.make_enmap(**sim_map_params)
    assert isinstance(test_map, enmap.ndmap)
    # Check shape: width/resolution*2 (since width is +/-)
    expected_npix = int(2 * sim_map_params["width_ra"] / sim_map_params["resolution"] * 60)
    # Allow for rounding
    assert abs(test_map.shape[-2] - expected_npix) < 5
    expected_npix = int(2 * sim_map_params["width_dec"] / sim_map_params["resolution"] * 60)
    assert abs(test_map.shape[-1] - expected_npix) < 5
    

def test_make_enmap_wcs(sim_map_params):
    from pixell.utils import degree,arcmin
    test_map = sim_maps.make_enmap(**sim_map_params)
    # Check WCS center
    y, x = np.array(test_map.shape) // 2
    dec, ra = enmap.pix2sky(test_map.shape, test_map.wcs, [y, x], safe=False)
    assert np.isclose(ra, sim_map_params["center_ra"]*degree, atol=1e-4)
    assert np.isclose(dec, sim_map_params["center_dec"]*degree, atol=1e-4)
    assert abs(test_map.wcs.wcs.cdelt[0]) == sim_map_params["resolution"] * arcmin / degree
    assert abs(test_map.wcs.wcs.cdelt[1]) == sim_map_params["resolution"] * arcmin / degree


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
    test_map = sim_maps.make_enmap(**sim_map_params)
    noise_std = 1.0
    noise_mean = 0.0
    noise_map = sim_maps.make_noise_map(test_map, 
                                        map_noise_Jy=noise_std, 
                                        map_mean_Jy=noise_mean, 
                                        seed=8675309
                                        )
    assert isinstance(noise_map, enmap.ndmap)
    assert noise_map.shape == test_map.shape
    # Check mean and std are within 1%
    assert np.isclose(np.mean(noise_map), noise_mean, atol=0.01)
    assert np.isclose(np.std(noise_map), noise_std, atol=0.01)




