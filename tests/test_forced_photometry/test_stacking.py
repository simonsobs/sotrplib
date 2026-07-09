import numpy as np
from astropy.time import Time, TimeDelta
from mapcat.toolkit import update_sky_coverage

from sotrplib.maps.database import FluxMapReader, RhoKappaMapReader
from sotrplib.sources.stacking import Stacker


def test_stacker(database_sessionmaker, create_test_maps):
    d1tables, sources = create_test_maps

    with database_sessionmaker() as session:
        session.add_all(d1tables)
        session.commit()

    update_sky_coverage.core(session=database_sessionmaker, convention="standard")

    end_time = Time("2025-01-10T00:00:00", scale="utc")
    start_time = end_time - TimeDelta(11, format="jd")

    db_reader = RhoKappaMapReader(
        number_to_read=None,
        start_time=start_time,
        end_time=end_time,
        sources=sources,
        frequency="f090",
        array="pa5",
        rerun=True,
    )

    stacker = Stacker(reader=db_reader)
    Stamps = stacker.stamp(sources)
    for i in range(3):
        simple_stack = np.sum(
            Stamps[i].flux_stamps / np.square(Stamps[i].ivar_stamps), axis=0
        ) / np.sum(1 / np.square(Stamps[i].ivar_stamps), axis=0)
        assert np.amax(simple_stack) <= 1.1
        assert np.amax(simple_stack) >= 0.95

    db_reader = FluxMapReader(
        number_to_read=None,
        start_time=start_time,
        end_time=end_time,
        sources=sources,
        frequency="f090",
        array="pa5",
        rerun=True,
    )

    stacker = Stacker(reader=db_reader)
    Stamps = stacker.stamp(sources)
    for i in range(3):
        simple_stack = np.sum(
            Stamps[i].flux_stamps / np.square(Stamps[i].ivar_stamps), axis=0
        ) / np.sum(1 / np.square(Stamps[i].ivar_stamps), axis=0)
        assert np.amax(simple_stack) <= 1.1
        assert np.amax(simple_stack) >= 0.95

    for i in range(10):
        assert np.isclose(
            Stamps[0].times[i] - start_time.unix,
            float(d1tables[0].ctime),
            rtol=1e-3,
        )
