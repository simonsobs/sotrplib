from datetime import datetime

import numpy as np
import structlog
from astropy import units as u
from pixell import enmap
from structlog.types import FilteringBoundLogger

from sotrplib.solar_system.solar_system import create_observer, get_sso_ephems_at_time


def mask_dustgal(
    imap: enmap.ndmap, galmask: enmap.ndmap, log: FilteringBoundLogger | None = None
):
    """Project the galactic dust mask onto the geometry of a target map.

    Parameters
    ----------
    imap : enmap.ndmap
        Target map whose shape and WCS define the output geometry.
    galmask : enmap.ndmap
        Galactic dust mask to reproject.
    log : FilteringBoundLogger, optional
        Structured logger.

    Returns
    -------
    enmap.ndmap
        Reprojected mask with the same geometry as ``imap``.
    """
    log = log or structlog.get_logger()
    log = log.bind(func="mask_dustgal")
    masked_map = enmap.project(galmask.astype(float), imap.shape, imap.wcs)
    log.info("mask_dustgal.completed", input_map_wcs=imap.wcs)
    return masked_map


def mask_planets(
    input_map: enmap.ndmap,
    mask_time: datetime | list[datetime],
    array: str | None = None,
    frequency: str | None = None,
    mask_radius=10 * u.arcmin,
    planets=["Venus", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"],
    log: FilteringBoundLogger | None = None,
) -> enmap.ndmap:
    """Create a boolean mask blanking out solar-system planets.

    Fetches planet ephemerides at the provided times and grows a circular
    mask around each planet position.  Planet positions are assumed to be
    approximately constant over the observation.

    Parameters
    ----------
    input_map : enmap.ndmap
        Map that defines the output geometry.
    mask_time : datetime or list of datetime
        Time(s) at which to evaluate planet positions.
    array : str, optional
        Detector array name (used for beam lookup).
    frequency : str, optional
        Frequency band (used for beam lookup).
    mask_radius : Quantity, optional
        Radius of the circular mask around each planet.  Default is 10 arcmin.
    planets : list of str, optional
        Planet names to mask.  Defaults to the six outer planets plus Venus.
    log : FilteringBoundLogger, optional
        Structured logger.

    Returns
    -------
    enmap.ndmap
        Boolean mask: ``True`` where the map is unmasked, ``False`` near
        planets.
    """
    log = log or structlog.get_logger()
    log = log.bind(func="mask_planets")
    ephem_times = mask_time if isinstance(mask_time, list) else [mask_time]
    planet_ephems = get_sso_ephems_at_time(
        ephem_df=None,
        sample_times=ephem_times,
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
    """Create a mask that zeros a band of pixels near the map edge.

    Parameters
    ----------
    imap : enmap.ndmap
        Map to create the edge mask for (typically the kappa map).
    pix_num : int
        Width of the masked border in pixels.

    Returns
    -------
    enmap.ndmap
        Binary mask: 1 where unmasked, 0 within ``pix_num`` pixels of the edge.
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
    """Apply edge, galactic, and planet masks to a map.

    Parameters
    ----------
    imap : enmap.ndmap
        Input map to mask.
    edgemask : enmap.ndmap
        Binary edge mask (1 = keep, 0 = mask).
    galmask : enmap.ndmap
        Galactic dust mask in the native map projection.  If ``None``, the
        galactic mask step is skipped.
    planet_mask : enmap.ndmap
        Planet mask (1 = keep, 0 = mask).

    Returns
    -------
    enmap.ndmap
        ``imap`` multiplied by all applicable masks.
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
    """Create a mask blanking circular regions around a list of source positions.

    Parameters
    ----------
    imap : enmap.ndmap
        Map that defines the output geometry.
    srcs_pix : list of array-like, optional
        Source pixel positions ``[[dec_pix, ra_pix], ...]``.
    fwhm : float, optional
        Beam FWHM in arcminutes.  Used with ``arr`` and ``freq`` to compute
        the cut radius when ``mask_radius`` is not supplied.
    mask_radius : list of float, optional
        Per-source cut radius in pixels.  Computed from ``get_cut_radius`` if
        not provided.
    arr : str, optional
        Detector array name (used for cut-radius lookup).
    freq : str, optional
        Frequency band (used for cut-radius lookup).

    Returns
    -------
    enmap.ndmap
        Mask with value 1 everywhere except inside the source discs (value 0).
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
