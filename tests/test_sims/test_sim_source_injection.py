import pytest
import numpy as np
from pixell import enmap
from sotrplib.sims import sim_maps
    
#from .conftest import sim_map_params, dummy_source


def test_photutils_sim_n_sources_output(sim_map_params):
    test_map = sim_maps.make_enmap(**sim_map_params)
    n_sources = 5
    out_map, injected_sources = sim_maps.photutils_sim_n_sources(test_map, 
                                                                 n_sources=n_sources, 
                                                                 min_flux_Jy=0.2, 
                                                                 max_flux_Jy=0.5, 
                                                                 map_noise_Jy=0.01, 
                                                                 seed=8675309
                                                                )
    # Check output types
    assert isinstance(out_map, enmap.ndmap)
    assert isinstance(injected_sources, list)
    assert len(injected_sources) == n_sources


def test_photutils_sim_n_sources_flux_range(sim_map_params):
    test_map = sim_maps.make_enmap(**sim_map_params)
    n_sources = 3
    min_flux = 1.0
    max_flux = 1.01
    _, injected_sources = sim_maps.photutils_sim_n_sources(test_map,
                                                           n_sources=n_sources, 
                                                           min_flux_Jy=min_flux,
                                                           max_flux_Jy=max_flux, 
                                                           map_noise_Jy=0.001, 
                                                           seed=8675309
                                                          )
    for src in injected_sources:
        assert min_flux <= src.flux <= max_flux


def test_photutils_sim_n_sources_invalid_flux(sim_map_params):
    test_map = sim_maps.make_enmap(**sim_map_params)
    with pytest.raises(ValueError):
        sim_maps.photutils_sim_n_sources(test_map, 
                                         n_sources=2, 
                                         min_flux_Jy=1.0, 
                                         max_flux_Jy=0.5
                                        )
        

def test_inject_sources_empty(sim_map_params):
    test_map = sim_maps.make_enmap(**sim_map_params)
    out_map, injected = sim_maps.inject_sources(test_map, [], 0.0)
    assert isinstance(out_map, enmap.ndmap)
    assert injected == []




def test_inject_sources_out_of_bounds(sim_map_params, dummy_source):
    from pixell.utils import degree
    test_map = sim_maps.make_enmap(**sim_map_params)
    oob_source = dummy_source
    center_ra,center_dec = test_map.wcs.wcs.crval
    ra_width = test_map.shape[-2] * test_map.wcs.wcs.cdelt[0] * 60  # arcmin
    dec_width = test_map.shape[-1] * test_map.wcs.wcs.cdelt[1] * 60  # arcmin
    oob_source.ra = (center_ra+ra_width)/degree   # Out of bounds
    oob_source.dec = (center_dec+dec_width)/degree  # Out of bounds
    _, injected = sim_maps.inject_sources(test_map, [oob_source], 0.0)
    assert injected == []


def test_inject_sources_not_flaring(sim_map_params, dummy_source):
    from pixell.utils import degree
    test_map = sim_maps.make_enmap(**sim_map_params)
    # observation_time far from peak_time
    quiescent_source = dummy_source
    center_ra,center_dec = test_map.wcs.wcs.crval
    quiescent_source.ra = center_ra/degree   # Out of bounds
    quiescent_source.dec = center_dec/degree  # Out of bounds
    ## inject source with width 1. but 10 days after peak time
    _, injected = sim_maps.inject_sources(test_map, [quiescent_source], 10.0)
    assert injected == []
