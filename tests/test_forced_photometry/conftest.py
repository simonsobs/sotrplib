"""
Fixtures for the dependency-injected completely simulated pipeline.
"""

import datetime

import pytest
from astropy import units as u
from mapcat import alembic_location
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# from sotrplib.outputs.core import MapOutputSerializer
from sotrplib.sims.maps import SimulatedMap
from sotrplib.sims.sources.core import (
    RandomSourceSimulation,
    RandomSourceSimulationParameters,
)


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
def mapset_with_sources(empty_map):
    parameters = RandomSourceSimulationParameters(
        n_sources=5,
        # Use bright sources so we can guarantee recovery
        min_flux=u.Quantity(9.999, "Jy"),
        max_flux=u.Quantity(10.001, "Jy"),
        fwhm_uncertainty_frac=0.01,
        fraction_return=1.0,
    )

    simulator = RandomSourceSimulation(parameters=parameters)
    maps = []
    for i in range(10):
        new_map, sources = simulator.simulate(input_map=empty_map)
        maps.append(new_map)

    yield maps, sources


def run_migration(database_path: str):
    from alembic import command
    from alembic.config import Config

    alembic_cfg = Config(alembic_location)
    database_url = f"sqlite:///{database_path}"
    alembic_cfg.set_main_option("sqlalchemy.url", database_url)
    command.upgrade(alembic_cfg, "head")


"""
@pytest.fixture(scope="session")
def simmed_data_file(request, mapset_with_sources):
    
    Fixture to download depth 1 maps for testing.

    Parameters
    ----------
    request : pytest.FixtureRequest
        Factory to create temporary directory for downloaded files
    Returns
    -------
    file_path : str
        Path to the downloaded file
    
    mapset, sources = mapset_with_sources

    for i, cur_map in enumerate(mapset):
        cache_dir = Path("../.pytest_cache/simmaps")
        filename = "simmap_{}".format(i)

        file_path = cache_dir / filename
        mapoutputter = MapOutputSerializer(file_path, field_ids=["flux", "snr"])
        enmap.save_map(cur_map, file_path)  # Unsure what correct function is

    return cache_dir, sources
"""


@pytest.fixture(scope="session", autouse=True)
def database_sessionmaker(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("mapcat")
    # Create a temporary SQLite database for testing.
    database_path = tmp_path / "stacking_test.db"

    # Run the migration on the database. This is blocking.
    run_migration(database_path)

    database_url = f"sqlite:///{database_path}"

    engine = create_engine(database_url, echo=True, future=True)

    yield sessionmaker(bind=engine, expire_on_commit=False)

    # Clean up the database (don't do this in case we want to inspect)
    database_path.unlink()
