"""
Tests the map I/O
"""

from sotrplib.map_io import load_maps

def test_get_depth1_mapset():
    map_list = load_maps.get_depth1_mapset(
        map_path="example_map/path"
    )

    # TODO: Improve these assertions?
    assert len(map_list) == 5


def test_load_map(empty_map):
    ret = load_maps.load_depth1_mapset(
        map_path=empty_map
    )

    # Ret is none becase the map is all zeros
    assert ret is None