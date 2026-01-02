from pixell import enmap


def pixell_map_union(map1, map2, op=lambda a, b: a + b):
    """Create a new pixell map that is the union of map1 and map2.
    The new map will have the shape and wcs that covers both input maps.
    The pixel values will be combined using the provided operation.

    Args:
        map1: First input pixell map.
        map2: Second input pixell map.
        op: Function to combine pixel values (default is addition).

    """
    oshape, owcs = enmap.union_geometry([map1.geometry, map2.geometry])
    omap = enmap.zeros(map1.shape[:-2] + oshape[-2:], owcs, map1.dtype)
    omap.insert(map1)
    omap.insert(map2, op=op)
    return omap
