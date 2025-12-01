"""
Map testing
"""

import datetime
import os

import pytest
from astropy import units as u
from pixell import enmap


@pytest.fixture
def map_params():
    return {
        "map1": {
            "map_noise": 0.01 * u.Jy,
            "center_ra": 1.0 * u.deg,
            "center_dec": -2.0 * u.deg,
            "width_ra": 1.0 * u.deg,
            "width_dec": 1.0 * u.deg,
            "resolution": 0.5 * u.arcmin,
        },
        "map2": {
            "map_noise": 0.01 * u.Jy,
            "center_ra": 3.0 * u.deg,
            "center_dec": -2.0 * u.deg,
            "width_ra": 1.0 * u.deg,
            "width_dec": 1.0 * u.deg,
            "resolution": 0.5 * u.arcmin,
        },
    }


def build_wcs(map_params, mapkey):
    nx, ny = (
        int(map_params[mapkey]["width_ra"] / map_params[mapkey]["resolution"]),
        int(map_params[mapkey]["width_dec"] / map_params[mapkey]["resolution"]),
    )
    nshape = (ny, nx)
    wcs = enmap.wcsutils.car(
        pos=[
            [
                map_params[mapkey]["center_ra"].to(u.deg).value
                - 0.5 * map_params[mapkey]["width_ra"].to(u.deg).value,
                map_params[mapkey]["center_dec"].to(u.deg).value
                - 0.5 * map_params[mapkey]["width_dec"].to(u.deg).value,
            ],
            [
                map_params[mapkey]["center_ra"].to(u.deg).value
                + 0.5 * map_params[mapkey]["width_ra"].to(u.deg).value,
                map_params[mapkey]["center_dec"].to(u.deg).value
                + 0.5 * map_params[mapkey]["width_dec"].to(u.deg).value,
            ],
        ],
        shape=nshape,
        res=map_params[mapkey]["resolution"].to(u.deg).value,
    )
    return nshape, wcs


def create_map(
    full_path,
    map_params,
    mapkey="map1",
    on: bool = False,
):
    shape, wcs = build_wcs(map_params, mapkey)
    if not on:
        enmap.zeros(shape, wcs=wcs).write(full_path, fmt="fits")
    else:
        enmap.ones(shape, wcs=wcs).write(full_path, fmt="fits")


def create_time_map(
    full_path,
    map_params,
    mapkey="map1",
):
    shape, wcs = build_wcs(map_params, mapkey)
    time_map = enmap.zeros(shape, wcs=wcs)
    time_map += datetime.datetime.now().timestamp()
    time_map.write(full_path, fmt="fits")


@pytest.fixture
def ones_map_set(tmp_path, map_params, mapkey="map1"):
    """
    Creates a set of five maps: map, ivar, rho, kappa and time.
    time is just a map of zeros.
    """
    map_paths = {
        x: tmp_path / f"{x}.fits" for x in ["map", "ivar", "rho", "kappa", "time"]
    }

    create_map(map_paths["map"], map_params, mapkey=mapkey, on=True)
    create_map(map_paths["ivar"], map_params, mapkey=mapkey, on=True)
    create_map(map_paths["rho"], map_params, mapkey=mapkey, on=True)
    create_map(map_paths["kappa"], map_params, mapkey=mapkey, on=True)
    create_time_map(map_paths["time"], map_params, mapkey=mapkey)
    yield map_paths

    for x in map_paths.values():
        os.unlink(x)


@pytest.fixture
def ones_unfiltered_partial_map_set(tmp_path, map_params, mapkey="map1"):
    """
    Creates a set of three maps: map, ivar, and time.
    time is just a map of zeros.

    rho and kappa are not created.
    """
    map_paths = {x: tmp_path / f"{x}.fits" for x in ["map", "ivar", "time"]}

    create_map(map_paths["map"], map_params, mapkey=mapkey, on=True)
    create_map(map_paths["ivar"], map_params, mapkey=mapkey, on=True)
    create_time_map(map_paths["time"], map_params, mapkey=mapkey)

    yield map_paths

    for x in map_paths.values():
        os.unlink(x)


@pytest.fixture
def ones_filtered_partial_map_set(tmp_path, map_params, mapkey="map1"):
    """
    Creates a set of three maps: rho, kappa and time.
    time is just a map of zeros.

    map and ivar are not created.
    """
    map_paths = {x: tmp_path / f"{x}.fits" for x in ["rho", "kappa", "time"]}

    create_map(map_paths["rho"], map_params, mapkey=mapkey, on=True)
    create_map(map_paths["kappa"], map_params, mapkey=mapkey, on=True)
    create_time_map(map_paths["time"], map_params, mapkey=mapkey)

    yield map_paths

    for x in map_paths.values():
        os.unlink(x)


@pytest.fixture
def map_set_1(tmp_path, map_params, mapkey="map1"):
    """
    Creates a set of five maps: map, ivar, rho, kappa and time.
    time is just a map of zeros.
    """
    map_paths = {
        x: tmp_path / f"{x}_set1.fits" for x in ["map", "ivar", "rho", "kappa", "time"]
    }

    create_map(map_paths["map"], map_params, mapkey=mapkey, on=True)
    create_map(map_paths["ivar"], map_params, mapkey=mapkey, on=True)
    create_map(map_paths["rho"], map_params, mapkey=mapkey, on=True)
    create_map(map_paths["kappa"], map_params, mapkey=mapkey, on=True)
    create_time_map(map_paths["time"], map_params, mapkey=mapkey)
    yield map_paths

    for x in map_paths.values():
        os.unlink(x)


@pytest.fixture
def map_set_2(tmp_path, map_params, mapkey="map2"):
    """
    Creates a set of five maps: map, ivar, rho, kappa and time.
    time is just a map of zeros.
    """
    map_paths = {
        x: tmp_path / f"{x}_set2.fits" for x in ["map", "ivar", "rho", "kappa", "time"]
    }

    create_map(map_paths["map"], map_params, mapkey=mapkey, on=True)
    create_map(map_paths["ivar"], map_params, mapkey=mapkey, on=True)
    create_map(map_paths["rho"], map_params, mapkey=mapkey, on=True)
    create_map(map_paths["kappa"], map_params, mapkey=mapkey, on=True)
    create_time_map(map_paths["time"], map_params, mapkey=mapkey)
    yield map_paths

    for x in map_paths.values():
        os.unlink(x)
