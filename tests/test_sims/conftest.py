import pytest

@pytest.fixture
def sim_map_params():
    return {
        "center_ra": 32.0,
        "center_dec": -51.0,
        "width_ra": 1.0,
        "width_dec": 1.0,
        "resolution": 0.5,
    }

@pytest.fixture
def dummy_source():
    from sotrplib.sims.sim_sources import SimTransient
    ds = SimTransient()
    ds.ra = 0.0
    ds.dec = 0.0
    ds.flux = 1.0
    ds.peak_time = 0.0
    ds.flare_width = 1.0
    return ds