import numpy as np
from astropy.time import Time

from sotrplib.sources.stacking import ForcedPhotometryStacker


def test_stacking(database_sessionmaker, mapset_with_sources):
    imaps, sources = mapset_with_sources

    with database_sessionmaker() as session:
        session.add_all(imaps)
        session.commit()

    start_time = Time("2025-01-01T00:00:00")
    end_time = Time("2025-01-10T00:00:00")
    stacker = ForcedPhotometryStacker(
        catalogs=[sources],
        start_time=start_time,
        end_time=end_time,
        freq="f090",
        array="OTi1",
    )
    stacker.get_maps(session=database_sessionmaker())
    stacker.filter_maps()
    stacker.get_stamps()

    source = sources[0]
    simple_stack = np.sum(stacker[0].stamps / stacker[0].ivar_stamps ** 2) / np.sum(
        1 / stacker[0].ivar_stamps ** 2
    )
