import pytest
from astropy import units as u


@pytest.fixture
def sim_map_params():
    return {
        "maps": {
            "map_noise": 0.01,
            "center_ra": 2.0,
            "center_dec": -2.0,
            "width_ra": 1.0,
            "width_dec": 1.0,
            "resolution": 0.5,
        }
    }


@pytest.fixture
def sim_source_params():
    return {
        "injected_sources": {
            "n_sources": 2,
            "min_flux": 0.1,
            "max_flux": 0.2,
            "fwhm_uncert_frac": 0.01,
        }
    }


@pytest.fixture
def sim_transient_params():
    return {
        "injected_transients": {
            "n_transients": 1,
            "min_flux": 0.1,
            "max_flux": 0.2,
            "min_width": 1.0,
            "max_width": 2.0,
            "min_peak_time": 0.0,
            "max_peak_time": 0.0,
        }
    }


@pytest.fixture
def dummy_source():
    from sotrplib.sims.sim_sources import SimTransient

    ds = SimTransient()
    ds.ra = 0.4 * u.deg
    ds.dec = -0.4 * u.deg
    ds.flux = 1.0 * u.Jy
    ds.peak_time = 0.0
    ds.flare_width = 1.0
    return ds


@pytest.fixture
def dummy_depth1_map(sim_map_params):
    from structlog import get_logger

    from sotrplib.sims import sim_maps

    log = get_logger("dummy_depth1_map")
    DummyMap = sim_maps.Depth1Map()
    DummyMap.flux = sim_maps.make_enmap(**sim_map_params["maps"], log=log)
    DummyMap.snr = 10 + abs(
        sim_maps.make_enmap(**sim_map_params["maps"], log=log)
    )  ## just keep above 0
    DummyMap.time_map = 0 * sim_maps.make_enmap(**sim_map_params["maps"], log=log)
    DummyMap.freq = "f090"
    DummyMap.wafer_name = "sim"
    DummyMap.map_id = "test"
    DummyMap.map_ctime = 0.0
    return DummyMap
