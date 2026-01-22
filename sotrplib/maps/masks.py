from datetime import datetime

import numpy as np
import structlog
from astropy import units as u
from pixell import enmap
from structlog.types import FilteringBoundLogger

from sotrplib.solar_system.solar_system import create_observer, get_sso_ephems_at_time


def mask_dustgal(imap: enmap.ndmap, galmask: enmap.ndmap):
    return enmap.extract(galmask, imap.shape, imap.wcs)


def mask_planets(
    input_map: enmap.ndmap,
    mask_time: datetime | list[datetime],
    array: str | None = None,
    frequency: str | None = None,
    mask_radius=10 * u.arcmin,
    planets=["Venus", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"],
    log: FilteringBoundLogger | None = None,
) -> enmap.ndmap:
    """
    Get's the planet ephemerides at start and end times of the map, and creates a mask around each planet position.
    Assumes that the planet positions do not change significantly during the observation.

    Args:
        input_map: enmap.ndmap to create planet mask for
        mask_time: time to get planet positions at, if None uses map start and end times
        array: array name, if None uses input_map.array
        frequency: frequency, if None uses input_map.frequency
        planets: list of planet names to mask
    """
    log = log or structlog.get_logger()
    log = log.bind(func="mask_planets")
    ephem_times = mask_time if isinstance(mask_time, list) else [mask_time]
    planet_ephems = get_sso_ephems_at_time(
        orbital_df=None,
        time=ephem_times,
        planets=planets,
        observer=create_observer(),
        log=log,
    )

    mask = enmap.ones(input_map.shape, input_map.wcs)
    for p in planet_ephems:
        sky_coord_pos = planet_ephems[p]["pos"]
        ras = sky_coord_pos.ra.rad
        decs = sky_coord_pos.dec.rad

        planet_locs = []
        for ra, dec in zip(ras, decs):
            planet_pix = input_map.sky2pix([dec, ra])
            planet_locs.append(planet_pix)

        if len(planet_locs) == 0:
            continue

        mask_radius_pix = 1.0
        planet_mask = make_src_mask(
            input_map,
            srcs_pix=planet_locs,
            fwhm=None,
            mask_radius=[mask_radius_pix] * len(planet_locs),
            arr=array,
            freq=frequency,
        )
        mask *= planet_mask

    ## some silliness required
    ## convert to bool, then grow mask, but grow mask grows the True
    ## so need to negate before, grow, then negate again
    mask = enmap.enmap(mask, mask.wcs, dtype=bool)
    mask = ~enmap.grow_mask(~mask, mask_radius.to_value(u.rad))
    log.info(
        "mask_planets.completed",
        input_map_wcs=input_map.wcs,
        times=mask_time,
        planets=planets,
    )
    return mask


def mask_edge(imap: enmap.ndmap, pix_num: int):
    """Get a mask that masking off edge pixels

    Args:
        imap:ndmap to create mask for,usually kappa map
        pix_num: pixels within this number from edge will be cutoff

    Returns:
        binary ndmap with 1 == unmasked, 0 == masked
    """
    from scipy.ndimage import morphology as morph

    from .maps import edge_map

    edge = edge_map(imap)
    dmap = enmap.enmap(morph.distance_transform_edt(edge), edge.wcs)
    del edge
    mask = enmap.zeros(imap.shape, imap.wcs)
    mask[np.where(dmap > pix_num)] = 1

    return mask


def get_masked_map(
    imap: enmap.ndmap,
    edgemask: enmap.ndmap,
    galmask: enmap.ndmap,
    planet_mask: enmap.ndmap,
):
    """returns map with edge, dustygalaxy, and planets masked

    Args:
        imap: input map
        edgemask: mask of map edges
        galmask: enmap of galaxy mask
        planet_mask: enmap of planet mask

    Returns:
        imap with masks applied
    """
    if galmask is None:
        return imap * edgemask * planet_mask
    else:
        galmask = mask_dustgal(imap, galmask)
        return imap * edgemask * galmask * planet_mask


def make_src_mask(
    imap: enmap.ndmap,
    srcs_pix: list = None,
    fwhm: float = None,
    mask_radius: list = [],
    arr: str = None,
    freq: str = None,
):
    """
    mask_radius in pixels, can be gotten using
    sotrplib.utils.utils.get_pix_from_peak_to_noise

    """
    from ..utils.utils import get_cut_radius

    mask = enmap.ones(imap.shape, wcs=imap.wcs, dtype=None)
    map_res = np.abs(imap.wcs.wcs.cdelt[0]) * u.deg
    if arr and freq and len(mask_radius) == 0:
        r_pix = get_cut_radius(map_res.to_value(u.arcmin), arr, freq, fwhm)
    else:
        r_pix = np.asarray(mask_radius)

    if len(srcs_pix) > 0:
        for i, cut_pix in enumerate(srcs_pix):
            r = round(r_pix[i]) + 1
            dec_pix = round(cut_pix[0])
            ra_pix = round(cut_pix[1])
            dec_min = max(dec_pix - r, 0)
            dec_max = min(dec_pix + r, imap.shape[0] - 1)
            dec, _ = imap.pix2sky([dec_pix, ra_pix])
            dec_factor = np.cos(dec)
            ra_min = max(round(ra_pix - r / dec_factor), 0)
            ra_max = min(round(ra_pix + r / dec_factor), imap.shape[1] - 1)
            mask[dec_min:dec_max, ra_min:ra_max] = 0

    return mask
