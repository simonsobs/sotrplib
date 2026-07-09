import pytest


@pytest.fixture
def mock_jpl_ephem_db():
    """
    make temp parquet file with
    designation 	datetime_utc 	julian_day 	ra_deg 	dec_deg 	distance_au
    1 Ceres 	2019-04-10 00:00:00 	2.458584e+06 	253.366792 	-16.705278 	2.020291
        1 Ceres 	2019-04-10 02:00:00 	2.458584e+06 	253.366292 	-16.706944 	2.019416
        1 Ceres 	2019-04-10 04:00:00 	2.458584e+06 	253.365458 	-16.708611 	2.018541
        1 Ceres 	2019-04-10 06:00:00 	2.458584e+06 	253.364292 	-16.710222 	2.017671
        1 Ceres 	2019-04-10 08:00:00 	2.458584e+06 	253.362875 	-16.711778 	2.016811

    """
    import pandas as pd

    data = {
        "designation": ["1 Ceres", "1 Ceres", "1 Ceres", "1 Ceres", "1 Ceres"],
        "datetime_utc": [
            pd.to_datetime("2019-04-10 00:00:00"),
            pd.to_datetime("2019-04-10 02:00:00"),
            pd.to_datetime("2019-04-10 04:00:00"),
            pd.to_datetime("2019-04-10 06:00:00"),
            pd.to_datetime("2019-04-10 08:00:00"),
        ],
        "julian_day": [
            pd.to_datetime("2019-04-10 00:00:00").to_julian_date(),
            pd.to_datetime("2019-04-10 02:00:00").to_julian_date(),
            pd.to_datetime("2019-04-10 04:00:00").to_julian_date(),
            pd.to_datetime("2019-04-10 06:00:00").to_julian_date(),
            pd.to_datetime("2019-04-10 08:00:00").to_julian_date(),
        ],
        "ra_deg": [253.366792, 253.366292, 253.365458, 253.364292, 253.362875],
        "dec_deg": [-16.705278, -16.706944, -16.708611, -16.710222, -16.711778],
        "distance_au": [2.020291, 2.019416, 2.018541, 2.017671, 2.016811],
    }
    df = pd.DataFrame(data)
    yield df
