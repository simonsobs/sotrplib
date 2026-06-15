from typing import Literal

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from pixell import enmap

type Box = np.ndarray[
    tuple[Literal[2], Literal[2]], np.dtype[np.float64]
]  # shape (2,2): [[dec_min, ra_max],[dec_max, ra_min]] in radians


def skycoord_box_to_enmap_box(sky_box: tuple[SkyCoord, SkyCoord]) -> Box:
    """Convert a ``(lower_left, upper_right)`` SkyCoord box to a pixell enmap box.

    Inputs are converted to ICRS internally.  ``sky_box[0]`` must be
    ``SkyCoord(ra=ra_min, dec=dec_min)`` and ``sky_box[1]`` must be
    ``SkyCoord(ra=ra_max, dec=dec_max)``.  Wrap is inferred when
    ``sky_box[0].ra > sky_box[1].ra``.

    Parameters
    ----------
    sky_box : tuple of SkyCoord
        ``(lower_left, upper_right)`` corners of the bounding box.

    Returns
    -------
    Box
        Pixell enmap box of the form
        ``[[dec_min, ra_max], [dec_max, ra_min]]`` in radians.
    """
    # make sure the input is in ICRS frame for internal use
    sky_box = (s.to_icrs() for s in sky_box)
    ra_min = sky_box[0].ra.to_value(u.rad)
    ra_max = sky_box[1].ra.to_value(u.rad)
    dec_min = sky_box[0].dec.to_value(u.rad)
    dec_max = sky_box[1].dec.to_value(u.rad)
    if ra_min > ra_max:
        return [[dec_min, ra_max], [dec_max, ra_min - 2 * np.pi]]
    return [[dec_min, ra_max], [dec_max, ra_min]]


def enmap_box_to_skycoord(raw_box: Box) -> tuple[SkyCoord, SkyCoord]:
    """Convert a pixell enmap box to a ``(lower_left, upper_right)`` SkyCoord tuple.

    Parameters
    ----------
    raw_box : Box
        Pixell box of the form ``[[dec_min, ra_max], [dec_max, ra_min]]`` in
        radians.  ``ra_min`` may be negative for wrap-around regions.

    Returns
    -------
    tuple of SkyCoord
        ``(SkyCoord(ra=ra_min, dec=dec_min), SkyCoord(ra=ra_max, dec=dec_max))``
        in the ICRS frame, following the ``ProcessableMap.sky_box`` convention.
        For a wrapping region ``raw_box[1][1]`` is negative and SkyCoord
        normalises it to near 2π, so ``sky_box[0].ra > sky_box[1].ra``
        signals the wrap.
    """
    return (
        SkyCoord(
            raw_box[1][1] * u.rad, raw_box[0][0] * u.rad, frame="icrs"
        ),  # (ra_min, dec_min)
        SkyCoord(
            raw_box[0][1] * u.rad, raw_box[1][0] * u.rad, frame="icrs"
        ),  # (ra_max, dec_max)
    )


def pixell_map_union(map1, map2, op=lambda a, b: a + b):
    """Create a new pixell map covering the union of two map geometries.

    Parameters
    ----------
    map1 : enmap.ndmap
        First input map.
    map2 : enmap.ndmap
        Second input map.
    op : callable, optional
        Function ``(a, b) -> result`` used to combine overlapping pixel values.
        Defaults to addition.

    Returns
    -------
    enmap.ndmap
        New map whose geometry covers both inputs, with pixel values combined
        by ``op``.
    """
    oshape, owcs = enmap.union_geometry([map1.geometry, map2.geometry])
    omap = enmap.zeros(map1.shape[:-2] + oshape[-2:], owcs, map1.dtype)
    omap.insert(map1)
    omap.insert(map2, op=op)
    return omap
