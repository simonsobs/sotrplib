import datetime

import numpy as np

from sotrplib.sources.stacking import FluxAndSNRStacker


def test_stacking(database_sessionmaker, create_test_maps):
    d1tables, sources = create_test_maps

    with database_sessionmaker() as session:
        session.add_all(d1tables)
        session.commit()

    date_string = "2025-01-10 00:00:00"
    format_code = "%Y-%m-%d %H:%M:%S"
    end_time = datetime.datetime.strptime(date_string, format_code)
    start_time = end_time - datetime.timedelta(days=11)
    stacker = FluxAndSNRStacker(
        sources=sources,
        start_time=start_time,
        end_time=end_time,
        freq="f090",
        array="OTi1",
    )
    stacker.get_maps()
    stacker.get_stamps()

    source = sources[0]
    simple_stack = np.sum(stacker[0].stamps / stacker[0].ivar_stamps ** 2) / np.sum(
        1 / stacker[0].ivar_stamps ** 2
    )
