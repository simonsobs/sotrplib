from datetime import datetime, timezone

import pandas as pd
import pytest
from skyfield.api import load

from sotrplib.solar_system.solar_system import create_observer, get_sso_ephems_at_time


def test_get_sso_ephems_at_time_single(orbital_params):
    orbital_df = pd.DataFrame(orbital_params)

    # Define a single time
    time = datetime(2025, 9, 15, 0, 0, tzinfo=timezone.utc)
    # Define observer location
    observer = create_observer()
    # Call the function
    sso_ephems = get_sso_ephems_at_time(orbital_df, time, observer)

    # Check results
    assert "(1) Ceres" in sso_ephems
    assert "(2) Pallas" in sso_ephems
    assert "pos" in sso_ephems["(1) Ceres"]
    assert len(sso_ephems["(1) Ceres"]["pos"]) == 1


def test_get_sso_ephems_at_time_multiple(orbital_params):
    orbital_df = pd.DataFrame(orbital_params)

    # Define a single time
    time = [
        datetime(2025, 9, 15, 0, 0, tzinfo=timezone.utc),
        datetime(2025, 9, 16, 0, 0, tzinfo=timezone.utc),
    ]
    # Define observer location
    observer = create_observer()
    # Call the function
    sso_ephems = get_sso_ephems_at_time(orbital_df, time, observer)

    # Check results
    assert "(1) Ceres" in sso_ephems
    assert "(2) Pallas" in sso_ephems
    assert "pos" in sso_ephems["(1) Ceres"]
    assert len(sso_ephems["(1) Ceres"]["pos"]) == 2


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
    sun = eph["sun"]
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
