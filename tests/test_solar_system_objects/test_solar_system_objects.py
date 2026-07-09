from datetime import datetime, timezone

import pytest
from skyfield.api import load

from sotrplib.solar_system.solar_system import create_observer, get_sso_ephems_at_time


def test_planet_positions():
    # Define times
    times = [
        datetime(2025, 9, 15, 0, 0, tzinfo=timezone.utc),
        datetime(2025, 9, 15, 0, 1, tzinfo=timezone.utc),
    ]
    ts = load.timescale()
    skyfield_time = ts.from_datetimes(times)

    # Define observer location
    observer = create_observer()
    eph = load("de440s.bsp")
    earth = eph["earth"]
    observer_topo = earth + observer
    # Call the function
    mars = eph["Mars Barycenter"]
    neptune = eph["Neptune Barycenter"]

    ra, dec, distance = observer_topo.at(skyfield_time).observe(mars).radec()

    # Check results according to JPL Horizons for 2025-09-15 00:00 UTC
    assert ra.degrees[0] == pytest.approx(202.96055, abs=1e-4)
    assert dec.degrees[0] == pytest.approx(-9.49651, abs=1e-4)
    assert distance.km[0] == pytest.approx(3.4584582437e08, abs=1e4)

    ra, dec, distance = observer_topo.at(skyfield_time).observe(neptune).radec()

    # Check results according to JPL Horizons for 2025-09-15 00:00 UTC
    assert ra.degrees[0] == pytest.approx(1.12167, abs=1e-4)
    assert dec.degrees[0] == pytest.approx(-1.00995, abs=1e-4)
    assert distance.km[0] == pytest.approx(4.3223260267e09, abs=1e4)


def test_asteroid_positions(mock_jpl_ephem_db):
    # Define times
    times = [
        datetime(2019, 4, 10, 4, 0, tzinfo=timezone.utc),
    ]
    sso_ephem = get_sso_ephems_at_time(mock_jpl_ephem_db, times, planets=[])
    sso = "1 Ceres"
    # Check results according to JPL Horizons for 2019-04-10 04:00 UTC
    assert sso_ephem[sso]["pos"].ra.degree[0] == pytest.approx(253.365458, abs=1e-4)
    assert sso_ephem[sso]["pos"].dec.degree[0] == pytest.approx(-16.708611, abs=1e-4)
    assert sso_ephem[sso]["pos"].distance.au[0] == pytest.approx(2.018541, abs=1e-5)
