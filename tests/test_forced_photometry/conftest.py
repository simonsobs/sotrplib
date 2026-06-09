"""
Fixtures for the dependency-injected completely simulated pipeline.
"""

import datetime
import os
from pathlib import Path

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from mapcat import alembic_location
from mapcat.database import DepthOneMapTable
from pixell import enmap
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from sotrplib.sims import (
    maps,
    sim_source_generators,
    source_injector,
)

# from sotrplib.outputs.core import MapOutputSerializer
from sotrplib.sims.maps import SimulatedMap
from sotrplib.sims.sources.core import (
    RandomSourceSimulation,
    RandomSourceSimulationParameters,
)
from sotrplib.utils.utils import get_fwhm


@pytest.fixture(scope="session", autouse=True)
def empty_map():
    sim_map = SimulatedMap(
        observation_start=datetime.datetime.now(tz=datetime.UTC)
        - datetime.timedelta(days=1),
        observation_end=datetime.datetime.now(tz=datetime.UTC),
    )

    sim_map.build()
    sim_map.finalize()

    assert sim_map.finalized

    yield sim_map


@pytest.fixture
def map_with_single_source(empty_map):
    parameters = RandomSourceSimulationParameters(
        n_sources=1,
        # Use bright sources so we can guarantee recovery
        min_flux=u.Quantity(9.9999, "Jy"),
        max_flux=u.Quantity(10.0001, "Jy"),
        fwhm_uncertainty_frac=0.01,
        fraction_return=1.0,
    )

    simulator = RandomSourceSimulation(parameters=parameters)

    new_map, sources = simulator.simulate(input_map=empty_map)

    yield new_map, sources


@pytest.fixture
def map_with_sources(empty_map):
    parameters = RandomSourceSimulationParameters(
        n_sources=10,
        # Use bright sources so we can guarantee recovery
        min_flux=u.Quantity(9.999, "Jy"),
        max_flux=u.Quantity(10.001, "Jy"),
        fwhm_uncertainty_frac=0.01,
        fraction_return=1.0,
    )

    simulator = RandomSourceSimulation(parameters=parameters)

    new_map, sources = simulator.simulate(input_map=empty_map)

    yield new_map, sources


@pytest.fixture
def map_with_single_asymmetric_source(empty_map):
    parameters = RandomSourceSimulationParameters(
        n_sources=1,
        # Use bright sources so we can guarantee recovery
        min_flux=u.Quantity(0.9999, "Jy"),
        max_flux=u.Quantity(1.0001, "Jy"),
        fwhm_uncertainty_frac=0.5,
        fraction_return=1.0,
    )

    simulator = RandomSourceSimulation(parameters=parameters)

    new_map, sources = simulator.simulate(input_map=empty_map)

    yield new_map, sources


@pytest.fixture(scope="session", autouse=True)
def mapset_with_sources():
    min_flux = u.Quantity(1.0, "Jy")
    max_flux = u.Quantity(1.1, "Jy")
    map_sim_params = maps.SimulationParameters()
    ## not all the way to the edge to avoid edge effects in photometry
    left = map_sim_params.center_ra - map_sim_params.width_ra / 2.5
    right = map_sim_params.center_ra + map_sim_params.width_ra / 2.5
    bottom = map_sim_params.center_dec - map_sim_params.width_dec / 2.5
    top = map_sim_params.center_dec + map_sim_params.width_dec / 2.5
    map_sim_params.map_noise = u.Quantity(0.001, "Jy")
    start_time = datetime.datetime.fromisoformat("2025-01-01T00:00:00+00:00")
    number = 3

    generator = sim_source_generators.FixedSourceGenerator(
        min_flux=min_flux,
        max_flux=max_flux,
        number=number,
        catalog_fraction=1.0,
    )

    sources, _ = generator.generate(
        box=[SkyCoord(ra=left, dec=bottom), SkyCoord(ra=right, dec=top)]
    )
    ret_maps = []
    for i in range(10):
        start_obs = start_time + datetime.timedelta(days=i)
        end_obs = start_time + datetime.timedelta(hours=8)
        base_map = maps.SimulatedMap(
            observation_start=start_obs,
            observation_end=end_obs,
            frequency="f090",
            array="pa5",
            sky_box=[SkyCoord(ra=left, dec=bottom), SkyCoord(ra=right, dec=top)],
        )

        base_map.build()

        injector = source_injector.PhotutilsSourceInjector(
            gauss_fwhm=get_fwhm(base_map.frequency)
        )
        _, new_map = injector.inject(input_map=base_map, simulated_sources=sources)
        ret_maps.append(new_map)
    for i, source in enumerate(sources):
        sources[i] = source._to_registered_fixed()
    yield ret_maps, sources


@pytest.fixture(scope="session")
def create_test_maps(mapset_with_sources):
    """
    Create a set of test maps for use in the forced photometry tests.
    If the maps already exist, this function will not recreate them.

    Parameters
    ----------
    mapset_with_sources : tuple
        A tuple containing a list of simulated maps and the sources in them.

    """
    from mapcat.helper import settings as mapcat_settings

    cached_dir = mapcat_settings.depth_one_parent / "simmaps"
    cached_dir.mkdir(parents=True, exist_ok=True)
    # Not sure I need to check if maps already exist, I don't think making them takes so much time#
    # map_names = [f"simmap_{i}" for i in range(len(maps))]

    maps, sources = mapset_with_sources
    d1tables = []
    for cur_map in maps:
        tstamp = str(int(np.mean(cur_map.time_mean)))
        freq = cur_map.frequency
        subdir = Path(tstamp[:5])
        map_name = Path("depth1_{tstamp}_{freq}".format(tstamp=tstamp, freq=freq))
        base = cached_dir / subdir / map_name
        base = str(base)
        for map_type in ["map", "rho", "kappa", "time", "flux", "snr"]:
            cur_map_name = base + f"_{map_type}.fits"
            if map_type == "time":
                imap = getattr(cur_map, "time_mean")
            elif map_type == "map":
                imap = getattr(cur_map, "rho") / getattr(cur_map, "kappa")
            else:
                imap = getattr(cur_map, map_type)
            if not Path(cur_map_name).exists():
                enmap.write_map(cur_map_name, imap)

        data = DepthOneMapTable(
            map_name=base,
            map_path=base + "_map.fits",
            rho_path=base + "_rho.fits",
            kappa_path=base + "_kappa.fits",
            flux_path=base + "_flux.fits",
            snr_path=base + "_snr.fits",
            mean_time_path=base + "_time.fits",
            tube_slot="pa5",
            frequency=freq,
            ctime=tstamp,
            start_time=int(np.min(cur_map.time_mean)),
            stop_time=int(np.max(cur_map.time_mean)),
        )
        d1tables.append(data)

    return d1tables, sources


def run_migration(database_path: str):
    from alembic import command
    from alembic.config import Config

    alembic_cfg = Config(alembic_location)
    database_url = f"sqlite:///{database_path}"
    alembic_cfg.set_main_option("sqlalchemy.url", database_url)
    command.upgrade(alembic_cfg, "head")


@pytest.fixture(scope="session", autouse=True)
def database_sessionmaker(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("mapcat")
    # Create a temporary SQLite database for testing.
    database_path = tmp_path / "stacking_test.db"

    # Run the migration on the database. This is blocking.
    run_migration(database_path)

    os.environ["MAPCAT_DATABASE_NAME"] = str(database_path)
    os.environ["MAPCAT_DEPTH_ONE_PARENT"] = str(tmp_path / "depth_one_parent")

    # TODO: Double check this crap
    from mapcat import helper as mapcat_helper

    mapcat_helper.settings.database_name = str(database_path)
    mapcat_helper.settings.depth_one_parent = Path(
        os.environ["MAPCAT_DEPTH_ONE_PARENT"]
    )
    mapcat_helper.settings._engine = None
    mapcat_helper.settings._sessionmaker = None

    engine = create_engine(
        mapcat_helper.settings.sync_database_url, echo=True, future=True
    )

    yield sessionmaker(bind=engine, expire_on_commit=False)

    # Clean up the database (don't do this in case we want to inspect)
    database_path.unlink()
