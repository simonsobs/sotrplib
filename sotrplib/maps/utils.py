import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from pixell import enmap


def skycoord_box_to_enmap_box(sky_box: tuple[SkyCoord, SkyCoord]) -> list:
    """Convert a ``(lower_left, upper_right)`` SkyCoord box to a pixell enmap box.

    ``sky_box[0]`` must be ``SkyCoord(ra=ra_min, dec=dec_min)`` and ``sky_box[1]``
    must be ``SkyCoord(ra=ra_max, dec=dec_max)``.  Wrap is inferred when
    ``sky_box[0].ra > sky_box[1].ra``.
    """
    ra_min = sky_box[0].ra.to_value(u.rad)
    ra_max = sky_box[1].ra.to_value(u.rad)
    dec_min = sky_box[0].dec.to_value(u.rad)
    dec_max = sky_box[1].dec.to_value(u.rad)
    if ra_min > ra_max:
        return [[dec_min, ra_max], [dec_max, ra_min - 2 * np.pi]]
    return [[dec_min, ra_max], [dec_max, ra_min]]


def enmap_box_to_skycoord(raw_box) -> tuple[SkyCoord, SkyCoord]:
    """Convert a pixell enmap box to a ``(lower_left, upper_right)`` SkyCoord tuple.

    pixell boxes have the form ``[[dec_min, ra_max], [dec_max, ra_min]]`` where
    ``ra_min`` may be negative for wrapping regions.  The returned tuple follows
    the ``ProcessableMap.sky_box`` convention:
    ``(SkyCoord(ra=ra_min, dec=dec_min), SkyCoord(ra=ra_max, dec=dec_max))``.
    For a wrapping region ``raw_box[1][1]`` is negative and SkyCoord normalises
    it to near 2π, so ``sky_box[0].ra > sky_box[1].ra`` signals the wrap.
    """
    return (
        SkyCoord(raw_box[1][1] * u.rad, raw_box[0][0] * u.rad),  # (ra_min, dec_min)
        SkyCoord(raw_box[0][1] * u.rad, raw_box[1][0] * u.rad),  # (ra_max, dec_max)
    )


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
