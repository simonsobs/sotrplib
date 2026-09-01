import numpy as np
import pandas as pd
import structlog
from astropy import units as u
from astropy.time import Time
from pixell import enmap
from structlog.types import FilteringBoundLogger

from sotrplib.maps.core import ProcessableMap
from sotrplib.solar_system.solar_system import (
    create_observer,
    get_sso_ephem_in_map,
    get_sso_ephems_at_time,
)


def mask_dustgal(
    imap: enmap.ndmap, galmask: enmap.ndmap, log: FilteringBoundLogger | None = None
):
    """Get a mask that masks out the dusty galaxy"""
    log = log or structlog.get_logger()
    log = log.bind(func="mask_dustgal")
    masked_map = enmap.project(galmask.astype(float), imap.shape, imap.wcs)
    log.info("mask_dustgal.completed", input_map_wcs=imap.wcs)
    return masked_map


def mask_planets(
    input_map: enmap.ndmap,
    mask_time: Time | list[Time],
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


def _mask_from_positions(
    input_map: ProcessableMap,
    positions: list[tuple[str, u.Quantity, u.Quantity]],
    mask_radius: u.Quantity,
    log: FilteringBoundLogger,
    func_name: str,
) -> enmap.ndmap:
    """Shared "positions -> hole mask, grown to mask_radius" logic used by
    both mask_asteroids_socat and mask_asteroids (local ephemeris)."""
    asteroid_locs = []
    for _name, ra, dec in positions:
        # SkyCoord components from ephemeris interpolation can come back
        # array-shaped even for a single target -- flatten to a true scalar
        # before sky2pix, or it returns array-valued pixel coords that
        # make_src_mask can't round().
        dec_val = np.atleast_1d(dec.to_value(u.rad))[0]
        ra_val = np.atleast_1d(ra.to_value(u.rad))[0]
        asteroid_locs.append(input_map.time_mean.sky2pix([dec_val, ra_val]))

    mask = enmap.ones(input_map.time_mean.shape, input_map.time_mean.wcs)
    if len(asteroid_locs) > 0:
        mask_radius_pix = 1.0
        asteroid_mask = make_src_mask(
            input_map.time_mean,
            srcs_pix=asteroid_locs,
            fwhm=None,
            mask_radius=[mask_radius_pix] * len(asteroid_locs),
        )
        mask *= asteroid_mask

    ## same convert-grow-reconvert dance as mask_planets: grow_mask grows
    ## True regions, but we want to grow the masked (False) region.
    mask = enmap.enmap(mask, mask.wcs, dtype=bool)
    mask = ~enmap.grow_mask(~mask, mask_radius.to_value(u.rad))
    log.info(
        f"{func_name}.completed",
        input_map_wcs=input_map.time_mean.wcs,
        n_asteroids_masked=len(asteroid_locs),
        asteroid_names=[name for name, _, _ in positions],
    )
    return mask


def mask_asteroids_socat(
    input_map: ProcessableMap,
    socat,
    mask_radius: u.Quantity = 10 * u.arcmin,
    log: FilteringBoundLogger | None = None,
) -> enmap.ndmap:
    """
    Mask solar-system objects that SOCat reports within this map's
    footprint and observation window. Preferred over mask_asteroids
    (below) because SOCat iteratively refines each SSO's position using
    the map's own per-pixel observation time (see
    SOCat._sg_to_registered_with_refinement), rather than a single
    mean-crossing-time position.

    Args:
        input_map: map to create the asteroid mask for.
        socat: a constructed sotrplib.source_catalog.socat.SOCat instance.
        mask_radius: radius to mask around each detected asteroid position.
    """
    log = log or structlog.get_logger()
    log = log.bind(func="mask_asteroids_socat")

    sources = socat.get_sources_in_map(input_map, monitored=False)
    sso_sources = [s for s in sources if s.source_type == "sso"]

    positions = [(str(s.source_id), s.ra, s.dec) for s in sso_sources]
    return _mask_from_positions(
        input_map, positions, mask_radius, log, "mask_asteroids_socat"
    )


def mask_asteroids(
    input_map: ProcessableMap,
    ephem_df: pd.DataFrame,
    mask_radius: u.Quantity = 10 * u.arcmin,
    interp_time_range: u.Quantity = 0.5 * u.day,
    interp_to: u.Quantity | None = 10 * u.min,
    log: FilteringBoundLogger | None = None,
) -> enmap.ndmap:
    """
    Find asteroids that actually cross this map's footprint during its
    observation window (via get_sso_ephem_in_map, which accounts for their
    motion rather than treating them as fixed points like mask_planets
    does), and creates a mask around each one's interpolated position at
    its crossing time.

    Fallback for mask_asteroids_socat, used when SOCat is unavailable --
    uses a static, precomputed ephemeris file instead of a live catalog
    query, so positions are only as good as a single mean-crossing-time
    interpolation (no per-pixel-time refinement).

    Args:
        input_map: map to create the asteroid mask for. Needs time_mean,
            observation_start/end, and filter_sources -- i.e. a full
            ProcessableMap, not just a bare enmap.
        ephem_df: asteroid ephemeris database, as loaded by
            sotrplib.solar_system.solar_system.load_jpl_ephem_database or
            load_mpc_orbital_database.
        mask_radius: radius to mask around each detected asteroid position.
        interp_time_range: window (see interpolate_ephem) used both to find
            crossings and to interpolate position at the crossing time.
        interp_to: sample rate used to check whether an asteroid crosses the
            map footprint at all -- see get_sso_ephem_in_map.
    """
    log = log or structlog.get_logger()
    log = log.bind(func="mask_asteroids")

    sso_ephems = get_sso_ephem_in_map(
        input_map=input_map,
        ephem_df=ephem_df,
        interp_time_range=interp_time_range,
        interp_to=interp_to,
        planets=[],  # planets are handled separately, by mask_planets
        log=log,
    )

    positions = [
        (obj, sso_ephems[obj]["pos"].ra, sso_ephems[obj]["pos"].dec)
        for obj in sso_ephems
    ]
    return _mask_from_positions(
        input_map, positions, mask_radius, log, "mask_asteroids"
    )


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
