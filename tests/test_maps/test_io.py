"""
Tests the map I/O
"""

from sotrplib.map_io import load_maps

def test_get_depth1_mapset():

    map_list = load_maps.get_depth1_mapset(
        map_path="example_map/path"
    )

    assert len(map_list) == 5


def test_load_empty_map(empty_map):
    map_path = next(empty_map,None)
    assert map_path is not None
    ret = load_maps.load_depth1_mapset(
        map_path=map_path
    )

    # Ret is none becase the map is all zeros
    assert ret is None

def test_load_ones_map(ones_map):
    map_path = next(ones_map,None)
    ret = load_maps.load_depth1_mapset(
        map_path=map_path
    )

    # Ret is none becase the map is all zeros
    assert ret is not None


def test_load_depth1_mapset(ones_map_set):
    paths = ones_map_set
    assert paths is not None
    assert len(paths) == 5
    ret = load_maps.load_depth1_mapset(
        map_path=paths["map"],
        ivar_path=paths["ivar"],
        rho_path=paths["rho"],
        kappa_path=paths["kappa"],
        time_path=paths["time"]
    )

    # all maps are present
    assert ret is not None
    assert ret.intensity_map is not None
    assert ret.inverse_variance_map is not None
    assert ret.rho_map is not None
    assert ret.kappa_map is not None
    assert ret.time_map is not None
