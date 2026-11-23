import numpy as np
from astropy.time import Time, TimeDelta
from mapcat.toolkit import update_sky_coverage

from sotrplib.maps.database import FluxMapReader, RhoKappaMapReader
from sotrplib.sources.stacking import InverseVarianceStacking, Stacker


def check_stacked_peaks(stacked):
    """Each test source has unit flux, so each stack should peak near 1."""
    assert len(stacked) == 3
    for per_source in stacked:
        assert len(per_source) == 1, "every stamp of a source was dropped"
        (stack,) = per_source
        assert np.all(stack.ivar > 0)
        assert 0.95 <= np.amax(stack.flux) <= 1.1


def test_stacker(database_sessionmaker, create_test_maps):
    d1tables, sources = create_test_maps

    with database_sessionmaker() as session:
        session.add_all(d1tables)
        session.commit()

    update_sky_coverage.core(session=database_sessionmaker)

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
    check_stacked_peaks(stacker.stack(sources, InverseVarianceStacking()))

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
    check_stacked_peaks(stacker.stack(sources, InverseVarianceStacking()))

    Stamps = stacker.stamps
    for i in range(10):
        assert np.isclose(
            Stamps[0].times[i] - start_time.unix,
            float(Time(d1tables[0].ctime).unix),
            rtol=1e-3,
        )
