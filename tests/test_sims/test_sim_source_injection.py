import pytest
import numpy as np
from pixell import enmap
from sotrplib.sims import sim_maps
    
#from .conftest import sim_map_params, dummy_source


def test_photutils_sim_n_sources_output(sim_map_params):
    test_map = sim_maps.make_enmap(**sim_map_params['maps'])
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
    test_map = sim_maps.make_enmap(**sim_map_params['maps'])
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
    test_map = sim_maps.make_enmap(**sim_map_params['maps'])
    with pytest.raises(ValueError):
        sim_maps.photutils_sim_n_sources(test_map, 
                                         n_sources=2, 
                                         min_flux_Jy=1.0, 
                                         max_flux_Jy=0.5
                                        )
        

def test_inject_sources_empty(sim_map_params):
    test_map = sim_maps.make_enmap(**sim_map_params['maps'])
    out_map, injected = sim_maps.inject_sources(test_map, [], 0.0)
    assert isinstance(out_map, enmap.ndmap)
    assert injected == []


def test_inject_sources_out_of_bounds(sim_map_params, dummy_source):
    from pixell.utils import degree
    test_map = sim_maps.make_enmap(**sim_map_params['maps'])
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
    test_map = sim_maps.make_enmap(**sim_map_params['maps'])
    # observation_time far from peak_time
    quiescent_source = dummy_source
    center_ra,center_dec = test_map.wcs.wcs.crval
    quiescent_source.ra = center_ra/degree   # Out of bounds
    quiescent_source.dec = center_dec/degree  # Out of bounds
    ## inject source with width 1. but 10 days after peak time
    _, injected = sim_maps.inject_sources(test_map, [quiescent_source], 10.0)
    assert injected == []


def test_inject_simulated_sources_all_none():
    from sotrplib.sims import sim_maps
    # Dummy mapdata object
    class DummyMap:
        pass
    mapdata = DummyMap()
    # All inputs None/empty
    result = sim_maps.inject_simulated_sources(mapdata)
    assert result == ([], [])


def test_inject_simulated_sources_random_only(dummy_depth1_map,sim_map_params,sim_source_params):
    from sotrplib.sims import sim_maps
    # Minimal sim_params for random injection
    sim_params = {**sim_map_params,
                  **sim_source_params,
    }
    # Should return a list of injected sources, no transients
    catalog_sources, injected_sources = sim_maps.inject_simulated_sources(dummy_depth1_map, 
                                                                          sim_params=sim_params,
                                                                          verbose=False)
    assert isinstance(catalog_sources, list)
    assert isinstance(injected_sources, list)
    assert len(catalog_sources) == sim_params["injected_sources"]["n_sources"]  # From random injection
    assert len(injected_sources) == 0  # No transients injected


def test_generate_transients_empty():
    from sotrplib.sims import sim_sources
    # Should return an empty list if no transients are specified
    transients = sim_sources.generate_transients(n=1, 
                                                 ra_lims=(1,3), 
                                                 dec_lims=(-3,-1), 
                                                 peak_amplitudes=None, 
                                                 peak_times=None, 
                                                 flare_widths=None, 
                                                 imap=None, 
                                                 uniform_on_sky=True
                                                 )
    assert isinstance(transients, list)
    assert len(transients) == 1


def test_inject_simulated_sources_with_db_and_transients( dummy_depth1_map,sim_map_params,sim_transient_params,sim_source_params):
    from sotrplib.sims import sim_maps
    
    # Minimal sim_params for random injection
    sim_params = {**sim_map_params,
                  **sim_transient_params,
                  **sim_source_params
                 }

    catalog_sources, injected_sources = sim_maps.inject_simulated_sources(
        dummy_depth1_map,
        injected_source_db=None,
        sim_params=sim_params,
        inject_transients=True,
        use_map_geometry=False,
        verbose=False
    )
    assert isinstance(catalog_sources, list)
    assert isinstance(injected_sources, list)
    assert len(catalog_sources) == 2  # From injected_source_db
    assert len(injected_sources) == 1  # From random injection


