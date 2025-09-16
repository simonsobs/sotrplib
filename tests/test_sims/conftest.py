import datetime
import os

import pytest
from astropy import units as u
from pixell import enmap


@pytest.fixture
def sim_map_params():
    return {
        "maps": {
            "map_noise": 0.01 * u.Jy,
            "center_ra": 1.0 * u.deg,
            "center_dec": -2.0 * u.deg,
            "width_ra": 1.0 * u.deg,
            "width_dec": 1.0 * u.deg,
            "resolution": 0.5 * u.arcmin,
        }
    }


@pytest.fixture
def sim_source_params():
    return {
        "injected_sources": {
            "n_sources": 2,
            "min_flux": 0.1 * u.Jy,
            "max_flux": 0.2 * u.Jy,
            "fwhm_uncert_frac": 0.01,
        }
    }


@pytest.fixture
def sim_transient_params():
    return {
        "injected_transients": {
            "n_transients": 1,
            "min_flux": 0.1 * u.Jy,
            "max_flux": 0.2 * u.Jy,
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
    ds.ra = 1.1 * u.deg
    ds.dec = -2.2 * u.deg
    ds.flux = 1.0 * u.Jy
    ds.peak_time = 0.0
    ds.flare_width = 1.0
    return ds


def build_wcs(sim_map_params):
    from pixell import enmap

    nx, ny = (
        int(sim_map_params["maps"]["width_ra"] / sim_map_params["maps"]["resolution"]),
        int(sim_map_params["maps"]["width_dec"] / sim_map_params["maps"]["resolution"]),
    )
    nshape = (ny, nx)
    wcs = enmap.wcsutils.car(
        pos=[
            [
                sim_map_params["maps"]["center_ra"].to(u.deg).value
                - 0.5 * sim_map_params["maps"]["width_ra"].to(u.deg).value,
                sim_map_params["maps"]["center_dec"].to(u.deg).value
                - 0.5 * sim_map_params["maps"]["width_dec"].to(u.deg).value,
            ],
            [
                sim_map_params["maps"]["center_ra"].to(u.deg).value
                + 0.5 * sim_map_params["maps"]["width_ra"].to(u.deg).value,
                sim_map_params["maps"]["center_dec"].to(u.deg).value
                + 0.5 * sim_map_params["maps"]["width_dec"].to(u.deg).value,
            ],
        ],
        shape=nshape,
        res=sim_map_params["maps"]["resolution"].to(u.deg).value,
    )
    return nshape, wcs


def create_map(
    full_path,
    sim_map_params,
    on: bool = False,
):
    shape, wcs = build_wcs(sim_map_params)
    if not on:
        enmap.zeros(shape, wcs=wcs).write(full_path, fmt="fits")
    else:
        enmap.ones(shape, wcs=wcs).write(full_path, fmt="fits")


@pytest.fixture
def ones_map_set(tmp_path, sim_map_params):
    """
    Creates a set of five maps: map, ivar, rho, kappa and time.
    time is just a map of zeros.
    """
    map_paths = {
        x: tmp_path / f"{x}.fits" for x in ["map", "ivar", "rho", "kappa", "time"]
    }

    create_map(map_paths["map"], on=True, sim_map_params=sim_map_params)
    create_map(map_paths["ivar"], on=True, sim_map_params=sim_map_params)
    create_map(map_paths["rho"], on=True, sim_map_params=sim_map_params)
    create_map(map_paths["kappa"], on=True, sim_map_params=sim_map_params)
    create_map(map_paths["time"], on=True, sim_map_params=sim_map_params)

    yield map_paths

    for x in map_paths.values():
        os.unlink(x)


@pytest.fixture
def dummy_map(sim_map_params, ones_map_set):
    from sotrplib.maps.core import RhoAndKappaMap

    DummyMap = RhoAndKappaMap(
        rho_filename=ones_map_set["rho"],
        kappa_filename=ones_map_set["kappa"],
        time_filename=ones_map_set["time"],
        start_time=datetime.datetime.now() - 1 * datetime.timedelta(days=1),
        end_time=datetime.datetime.now(),
    )
    DummyMap.build()
    DummyMap.add_time_offset(DummyMap.observation_start)
    DummyMap.finalize()
    DummyMap.frequency = "f090"
    DummyMap.array = "sim"
    return DummyMap
