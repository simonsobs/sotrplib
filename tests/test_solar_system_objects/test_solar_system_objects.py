from datetime import datetime, timezone

import pandas as pd

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
    assert "ra" in sso_ephems["(1) Ceres"]
    assert "dec" in sso_ephems["(1) Ceres"]


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
    assert "ra" in sso_ephems["(1) Ceres"]
    assert "dec" in sso_ephems["(1) Ceres"]
