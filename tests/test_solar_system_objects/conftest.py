import pytest


@pytest.fixture
def orbital_params():
    # Sample orbital dataframe
    data = {
        "designation": ["(1) Ceres", "(2) Pallas", "(3) Juno", "(4) Vesta"],
        "semimajor_axis_au": [2.7656157, 2.7699258, 2.6708791, 2.3615413],
        "eccentricity": [0.0795763, 0.230643, 0.2558258, 0.0901676],
        "inclination_degrees": [10.58789, 34.92833, 12.98604, 7.14406],
        "argument_of_perihelion_degrees": [73.29974, 310.9334, 247.88367, 151.53712],
        "longitude_of_ascending_node_degrees": [
            80.24963,
            172.88859,
            169.81989,
            103.70232,
        ],
        "mean_anomaly_degrees": [231.53975, 211.52977, 217.59095, 26.80968],
        "epoch_packed": ["K25BL", "K25BL", "K25BL", "K25BL"],
    }

    yield data
