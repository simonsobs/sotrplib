from typing import Literal

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from pixell import enmap, wcsutils

type Box = np.ndarray[
    tuple[Literal[2], Literal[2]], np.dtype[np.float64]
]  # shape (2,2): [[dec_min, ra_max],[dec_max, ra_min]] in radians


def skycoord_box_to_enmap_box(sky_box: tuple[SkyCoord, SkyCoord]) -> Box:
    """Convert a ``(lower_left, upper_right)`` SkyCoord box to a pixell enmap box.
    Convert to ICRS frame to use internally.
    ``sky_box[0]`` must be ``SkyCoord(ra=ra_min, dec=dec_min)`` and ``sky_box[1]``
    must be ``SkyCoord(ra=ra_max, dec=dec_max)``.  Wrap is inferred when
    ``sky_box[0].ra > sky_box[1].ra``.
    returned pixell box has the form ``[[dec_min, ra_max], [dec_max, ra_min]]`` in radians.
    """
    # make sure the input is in ICRS frame for internal use
    sky_box = tuple(s.transform_to("icrs") for s in sky_box)
    ra_min = sky_box[0].ra.to_value(u.rad)
    ra_max = sky_box[1].ra.to_value(u.rad)
    dec_min = sky_box[0].dec.to_value(u.rad)
    dec_max = sky_box[1].dec.to_value(u.rad)
    if ra_min > ra_max:
        return [[dec_min, ra_max], [dec_max, ra_min - 2 * np.pi]]
    return [[dec_min, ra_max], [dec_max, ra_min]]


def enmap_box_to_skycoord(raw_box: Box) -> tuple[SkyCoord, SkyCoord]:
    """Convert a pixell enmap box to a ``(lower_left, upper_right)`` SkyCoord tuple in icrs frame.

    pixell boxes have the form ``[[dec_min, ra_max], [dec_max, ra_min]]`` where
    ``ra_min`` may be negative for wrapping regions.
    Expected input is in radians.

    The returned tuple follows
    the ``ProcessableMap.sky_box`` convention:
    ``(SkyCoord(ra=ra_min, dec=dec_min), SkyCoord(ra=ra_max, dec=dec_max))``.
    For a wrapping region ``raw_box[1][1]`` is negative and SkyCoord normalises
    it to near 2π, so ``sky_box[0].ra > sky_box[1].ra`` signals the wrap.
    """
    return (
        SkyCoord(
            raw_box[1][1] * u.rad, raw_box[0][0] * u.rad, frame="icrs"
        ),  # (ra_min, dec_min)
        SkyCoord(
            raw_box[0][1] * u.rad, raw_box[1][0] * u.rad, frame="icrs"
        ),  # (ra_max, dec_max)
    )


def ra_wrap_pixels(wcs) -> int | None:
    """Return the number of pixels in 360 deg of RA, or None if the wcs
    does not wrap in RA."""
    if wcsutils.is_plain(wcs):
        return None
    n = 360.0 / abs(wcs.wcs.cdelt[0])
    return int(round(n)) if abs(n - round(n)) < 1e-6 else None


def union_geometry(geometries):
    """
    Return the geometry that contains all the given (shape, wcs) geometries.

    Like enmap.union_geometry, but RA wraps. Each geometry is moved by whole
    wraps in RA to be nearest to the union of the geometries before it. If
    the union is wider than 360 deg, it is cut to 360 deg, with column 0 at
    RA +180 deg. enmap.insert wraps the inputs into it.
    """
    ref_wcs = geometries[0][1]
    nwrap = ra_wrap_pixels(ref_wcs)
    if nwrap is None:
        return enmap.union_geometry(geometries)

    boxes = [enmap.pixbox_of(ref_wcs, shape, wcs) for shape, wcs in geometries]
    y0 = min(b[0][0] for b in boxes)
    y1 = max(b[1][0] for b in boxes)
    x0, x1 = boxes[0][0][1], boxes[0][1][1]
    for box in boxes[1:]:
        bx0, bx1 = box[0][1], box[1][1]
        k = round((x0 + x1 - bx0 - bx1) / (2 * nwrap))
        k = min(
            (k - 1, k, k + 1),
            key=lambda j: max(x1, bx1 + j * nwrap) - min(x0, bx0 + j * nwrap),
        )
        x0, x1 = min(x0, bx0 + k * nwrap), max(x1, bx1 + k * nwrap)

    if x1 - x0 > nwrap:
        w = ref_wcs.wcs
        x180 = (w.crpix[0] - 1) + (180.0 - w.crval[0]) / w.cdelt[0]
        x0 = int(round(x180))
        x1 = x0 + nwrap

    owcs = ref_wcs.deepcopy()
    owcs.wcs.crpix -= np.array([x0, y0])
    return (int(y1 - y0), int(x1 - x0)), owcs


def pixell_map_union(map1, map2, op=lambda a, b: a + b):
    """Create a new pixell map that is the union of map1 and map2.
    The new map will have the shape and wcs that covers both input maps
    (see union_geometry; RA wraps). The pixel values will be combined
    using the provided operation.

    Args:
        map1: First input pixell map.
        map2: Second input pixell map.
        op: Function to combine pixel values (default is addition).

    """
    oshape, owcs = union_geometry([map1.geometry, map2.geometry])
    omap = enmap.zeros(map1.shape[:-2] + oshape[-2:], owcs, map1.dtype)
    omap.insert(map1)
    omap.insert(map2, op=op)
    return omap
