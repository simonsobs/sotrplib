"""
Map testing
"""

import datetime
import os

import numpy as np
import pytest
from astropy import units as u
from pixell import enmap


@pytest.fixture
def separate_map_params():
    return {
        "map1": {
            "map_noise": 0.001 * u.Jy,
            "center_ra": 1.0 * u.deg,
            "center_dec": -2.0 * u.deg,
            "width_ra": 1.0 * u.deg,
            "width_dec": 1.0 * u.deg,
            "resolution": 0.5 * u.arcmin,
        },
        "map2": {
            "map_noise": 0.001 * u.Jy,
            "center_ra": 3.0 * u.deg,
            "center_dec": -2.0 * u.deg,
            "width_ra": 1.0 * u.deg,
            "width_dec": 1.0 * u.deg,
            "resolution": 0.5 * u.arcmin,
        },
    }


@pytest.fixture
def overlapping_map_params():
    return {
        "map1": {
            "map_noise": 0.001 * u.Jy,
            "center_ra": 1.0 * u.deg,
            "center_dec": -2.0 * u.deg,
            "width_ra": 2.0 * u.deg,
            "width_dec": 2.0 * u.deg,
            "resolution": 0.5 * u.arcmin,
        },
        "map2": {
            "map_noise": 0.001 * u.Jy,
            "center_ra": 2.0 * u.deg,
            "center_dec": -3.0 * u.deg,
            "width_ra": 2.0 * u.deg,
            "width_dec": 2.0 * u.deg,
            "resolution": 0.5 * u.arcmin,
        },
    }


@pytest.fixture
def separate_source_params():
    return {
        "source1": {
            "ra": 1.0 * u.deg,
            "dec": -2.0 * u.deg,
            "flux": 10.0 * u.Jy,
        },
        "source2": {
            "ra": 3.0 * u.deg,
            "dec": -2.0 * u.deg,
            "flux": 10.0 * u.Jy,
        },
    }


@pytest.fixture
def overlapping_source_params():
    return {
        "source1": {
            "ra": 1.5 * u.deg,
            "dec": -2.5 * u.deg,
            "flux": 10.0 * u.Jy,
        },
        "source2": {
            "ra": 2.5 * u.deg,
            "dec": -3.5 * u.deg,
            "flux": 10.0 * u.Jy,
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
    mapkey,
    on: bool = False,
):
    shape, wcs = build_wcs(map_params, mapkey)
    if not on:
        enmap.zeros(shape, wcs=wcs).write(full_path, fmt="fits")
    else:
        m = enmap.ones(shape, wcs=wcs)
        m += np.random.normal(
            scale=map_params[mapkey]["map_noise"].to(u.Jy).value, size=shape
        )
        m.write(full_path, fmt="fits")


def create_time_map(
    full_path,
    map_params,
    mapkey,
):
    shape, wcs = build_wcs(map_params, mapkey)
    time_map = enmap.zeros(shape, wcs=wcs)
    time_map += datetime.datetime(2025, 10, 10, 0, 0, 0).timestamp()
    time_map.write(full_path, fmt="fits")


def map_set(tmp_path, map_params, mapkey="map1"):
    """
    Creates a set of five maps: map, ivar, rho, kappa and time.
    rho and map are zeros , ivar and kappa are ones.
    time is just a map of a constant given by the current timestamp.
    """
    map_paths = {
        x: tmp_path / f"{mapkey}_{x}.fits"
        for x in ["map", "ivar", "rho", "kappa", "time"]
    }

    create_map(map_paths["map"], map_params, mapkey=mapkey, on=True)
    create_map(map_paths["ivar"], map_params, mapkey=mapkey, on=True)
    create_map(map_paths["rho"], map_params, mapkey=mapkey, on=True)
    create_map(map_paths["kappa"], map_params, mapkey=mapkey, on=True)
    create_time_map(map_paths["time"], map_params, mapkey=mapkey)
    return map_paths


@pytest.fixture
def separate_map_set_1(tmp_path, separate_map_params):
    """
    Creates a set of five maps: map, ivar, rho, kappa and time.
    time is just a map of a constant given by the current timestamp.
    """
    map_paths = map_set(tmp_path, separate_map_params, mapkey="map1")
    yield map_paths
    for x in map_paths.values():
        os.unlink(x)


@pytest.fixture
def separate_map_set_2(tmp_path, separate_map_params):
    """
    Creates a set of five maps: map, ivar, rho, kappa and time.
    time is just a map of a constant given by the current timestamp.
    """
    map_paths = map_set(tmp_path, separate_map_params, mapkey="map2")
    yield map_paths
    for x in map_paths.values():
        os.unlink(x)


@pytest.fixture
def overlapping_map_set_1(tmp_path, overlapping_map_params):
    """
    Creates a set of five maps: map, ivar, rho, kappa and time.
    time is just a map of a constant given by the current timestamp.
    """
    map_paths = map_set(tmp_path, overlapping_map_params, mapkey="map1")
    yield map_paths
    for x in map_paths.values():
        os.unlink(x)


@pytest.fixture
def overlapping_map_set_2(tmp_path, overlapping_map_params):
    """
    Creates a set of five maps: map, ivar, rho, kappa and time.
    time is just a map of a constant given by the current timestamp.
    """
    map_paths = map_set(tmp_path, overlapping_map_params, mapkey="map2")
    yield map_paths
    for x in map_paths.values():
        os.unlink(x)
