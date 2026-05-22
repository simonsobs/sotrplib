import datetime

import numpy as np
from mapcat.toolkit import update_sky_coverage

from sotrplib.sources.stacking import FluxSNRStacker, RhoKappaStacker


def test_RhoKappa(database_sessionmaker, create_test_maps):
    d1tables, sources = create_test_maps

    with database_sessionmaker() as session:
        session.add_all(d1tables)
        session.commit()

    update_sky_coverage.core(session=database_sessionmaker, convention="standard")

    date_string = "2025-01-10 00:00:00"
    format_code = "%Y-%m-%d %H:%M:%S"
    end_time = datetime.datetime.strptime(date_string, format_code)
    start_time = end_time - datetime.timedelta(days=11)

    stacker = RhoKappaStacker(
        sources=sources,
        start_time=start_time,
        end_time=end_time,
        freq="f090",
        array="pa5",
    )

    stacker.get_maps()
    stacker.get_stamps()

    for i in range(3):
        simple_stack = np.sum(
            stacker.stamps[i] / np.square(stacker.ivar_stamps[i]), axis=0
        ) / np.sum(1 / np.square(stacker.ivar_stamps[i]), axis=0)
        assert np.amax(simple_stack) <= 1.1
        assert (
            np.amax(simple_stack) >= 0.95
        )  # Note can be slightly less than 1.0 due to not actually measuring peak height and just using amax


def test_FluxSNR(database_sessionmaker, create_test_maps):
    d1tables, sources = create_test_maps

    with database_sessionmaker() as session:
        session.add_all(d1tables)
        session.commit()

    update_sky_coverage.core(session=database_sessionmaker, convention="standard")

    date_string = "2025-01-10 00:00:00"
    format_code = "%Y-%m-%d %H:%M:%S"
    end_time = datetime.datetime.strptime(date_string, format_code)
    start_time = end_time - datetime.timedelta(days=11)
    stacker = FluxSNRStacker(
        sources=sources,
        start_time=start_time,
        end_time=end_time,
        freq="f090",
        array="pa5",
    )

    stacker.get_maps()
    stacker.get_stamps()
    for i in range(3):
        simple_stack = np.sum(
            stacker.stamps[i] / np.square(stacker.ivar_stamps[i]), axis=0
        ) / np.sum(1 / np.square(stacker.ivar_stamps[i]), axis=0)
        assert np.amax(simple_stack) <= 1.1
        assert (
            np.amax(simple_stack) >= 0.95
        )  # Note can be slightly less than 1.0 due to not actually measuring peak height and just using amax
