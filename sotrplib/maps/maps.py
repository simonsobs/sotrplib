import warnings
from pathlib import Path

import numpy as np
import structlog
from astropy import units as u

# TODO: move photutils utilities to a separate file.
from photutils.datasets import make_model_image
from photutils.psf import GaussianPSF
from pixell import enmap
from pixell.utils import arcmin, degree
from structlog.types import FilteringBoundLogger

from sotrplib.sources.sources import MeasuredSource, ProcessableMap

from ..sims.sim_utils import make_2d_gaussian_model_param_table


def get_map_info_from_filename(map_path):
    if not isinstance(map_path, Path):
        return

    suffix = str(map_path).split("/")[-1]
    wafer_name = suffix.split("_")[2]
    freq = str(map_path).split("/")[-1].split("_")[3]
    map_ctime = float(str(map_path).split("/")[-1].split("_")[1])
    start_ctime = get_observation_start_time(map_path)

    return {
        "wafer_name": wafer_name,
        "freq": freq,
        "map_ctime": map_ctime,
        "map_start_time": start_ctime,
    }


def init_empty_so_map(
    res: u.Quantity[u.arcmin] | None = None,
    intensity: bool = False,
    include_half_pixel_offset=False,
    box: u.Quantity[u.deg] | None = None,
):
    ## res in radian
    if not res:
        raise Exception("Need a resolution, no input res and self.res is None.")

    if box is None:
        empty_map = enmap.zeros(
            *widefield_geometry(
                res=res, include_half_pixel_offset=include_half_pixel_offset
            )
        )
    else:
        shape, wcs = enmap.geometry(box.to(u.rad).value, res=res.to(u.rad).value)
        empty_map = enmap.zeros(shape, wcs=wcs)

    return empty_map


def subtract_sources(
    input_map: ProcessableMap,
    sources: list[MeasuredSource],
    src_model: enmap.ndmap = None,
    verbose=False,
    cuts={},
    log: FilteringBoundLogger | None = None,
):
    """
    src_model is a simulated (model) map of the sources in the list.
    sources are fit using photutils, and are MeasuredSource
    objects with fwhm_ra, fwhm_dec, ra, dec, flux, and orientation
    """
    log = log if log else structlog.get_logger()
    log.bind(func_name="subtract_sources")
    if len(sources) == 0:
        log.warning("subtract_sources.no_sources", num_sources=0)
        return
    if src_model is None:
        from ..utils.utils import get_fwhm

        src_model = make_model_source_map(
            input_map.flux,
            sources,
            nominal_fwhm=get_fwhm(input_map.frequency),
            matched_filtered=True,
            verbose=verbose,
            cuts=cuts,
            log=log,
        )
    input_map.flux -= src_model / (input_map.flux_units.to(u.Jy))
    log.info("subtract_sources.source_flux_subtracted")
    ## TODO do we want to inject gaussian snr or is setting it to 0 kosher?
    input_map.snr[abs(src_model) > 1e-8] = 0.0
    log.info("subtract_sources.source_snr_masked")
    return src_model


## TODO: I think this bug has been fixed in pixell. need to confirm
def enmap_map_union(map1, map2):
    ## from enmap since enmap.zeros causes it to fail.
    oshape, owcs = enmap.union_geometry([map1.geometry, map2.geometry])
    omap = enmap.zeros(map1.shape[:-2] + oshape[-2:], owcs, map1.dtype)
    omap.insert(map1)
    omap.insert(map2, op=lambda a, b: a + b)
    return omap


def kappa_clean(
    kappa: np.ndarray,
    rho: np.ndarray,
    min_frac: float = 1e-3,
):
    kappa = np.maximum(kappa, np.nanmax(kappa) * min_frac)
    kappa[np.where(rho == 0.0)] = 0.0
    return kappa


def clean_map(
    imap: np.ndarray,
    inverse_variance: np.ndarray,
    fraction: float = 0.01,
    cut_on: str = "max",
):
    ## cut_on can be max or median, this sets the imap to zero for values of inverse variance
    ## which are below fraction*max or fraction*median of inverse variance map.
    if cut_on == "median" or cut_on == "med":
        imap[inverse_variance < (np.nanmedian(inverse_variance) * fraction)] = 0
    elif cut_on == "percentile" or cut_on == "pct":
        imap[inverse_variance < np.nanpercentile(inverse_variance, fraction)] = 0
    else:
        if cut_on != "max":
            print("%s cut_on not supported, defaulting to max cut" % cut_on)
        imap[inverse_variance < (np.nanmax(inverse_variance) * fraction)] = 0
    return imap


def get_snr(
    rho: np.ndarray,
    kappa: np.ndarray,
):
    # supress divide by zero warning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        snr = rho / kappa**0.5
    snr[np.where(kappa == 0.0)] = 0.0
    return snr


def get_flux(rho: np.ndarray, kappa: np.ndarray):
    # supress divide by zero warning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        flux = rho / kappa
    ## kind of want to do something else with this -AF
    flux[np.where(kappa == 0.0)] = 0.0
    return flux


def get_dflux(kappa: np.ndarray):
    # supress divide by zero warning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dflux = kappa**-0.5
    dflux[np.where(kappa == 0.0)] = 0.0
    return dflux


def get_maptype(map_path: Path):
    if "rho" in str(map_path):
        maptype = "rho"
    elif "kappa" in str(map_path):
        maptype = "kappa"
    elif "map" in str(map_path):
        maptype = "map"
    elif "ivar" in str(map_path):
        maptype = "ivar"
    elif "time" in str(map_path):
        maptype = "time"
    if isinstance(map_path, str):
        suffix = "." + map_path.split(".")[-1]
    else:
        suffix = map_path.suffix
    return maptype, suffix


def get_observation_start_time(map_path: Path):
    from os.path import exists

    maptype, _ = get_maptype(map_path)
    infofile = str(map_path).split(maptype)[0] + "info.hdf"
    if not exists(infofile):
        t0 = 0.0
    else:
        from pixell.bunch import read as bunch_read

        info = bunch_read(infofile)
        t0 = info.t
    return t0


def edge_map(imap: enmap.ndmap):
    """Finds the edges of a map

    Args:
        imap: ndmap to find edges of

    Returns:
        binary ndmap with 1 inside region, 0 outside
    """
    from scipy.ndimage import binary_fill_holes

    edge = enmap.enmap(imap, imap.wcs)  # Create map geometry
    edge[np.abs(edge) > 0] = 1  # Convert to binary
    edge = binary_fill_holes(edge)  # Fill holes

    return enmap.enmap(edge.astype("ubyte"), imap.wcs)


def get_submap(
    imap: enmap.ndmap,
    ra_deg: float,
    dec_deg: float,
    size_deg: float = 0.5,
) -> enmap:
    ## Does not reproject

    ra = ra_deg * degree
    dec = dec_deg * degree
    radius = size_deg * degree
    omap = imap.submap(
        [[dec - radius, ra - radius], [dec + radius, ra + radius]],
    )
    return omap


def get_thumbnail(
    imap: enmap.ndmap,
    ra_deg: float,
    dec_deg: float,
    size_deg: float = 0.5,
    proj: str = "tan",
) -> enmap:
    from pixell import reproject

    ra = ra_deg * degree
    dec = dec_deg * degree
    omap = reproject.thumbnails(
        imap,
        [dec, ra],
        size_deg * degree,
        proj=proj,
    )
    if np.all(np.isnan(omap)):
        omap = reproject.thumbnails(
            np.nan_to_num(imap),
            [dec, ra],
            size_deg * degree,
            proj=proj,
        )
    return omap


def get_time_safe(time_map: enmap.ndmap, poss: list, r: float = 5.0):
    ## pos [[dec,ra],[dec,ra],...]
    ## r radius in arcmin

    r *= arcmin
    poss = np.array(poss)
    vals = time_map.at(poss, order=0)
    bad = np.where(vals == 0)[0]
    if len(bad) > 0:
        pixboxes = enmap.neighborhood_pixboxes(
            time_map.shape, time_map.wcs, poss.T[bad], r=r
        )
        for i, pixbox in enumerate(pixboxes):
            thumb = time_map.extract_pixbox(pixbox)
            mask = thumb != 0
            vals[bad[i]] = np.sum(mask * thumb) / np.sum(mask)
    return vals


def widefield_geometry(
    res=None,
    shape=None,
    dims=(),
    proj="car",
    variant="fejer1",
    include_half_pixel_offset=False,
):
    """Build an enmap covering the full sky, with the outermost pixel centers
        at the poles and wrap-around points. Only the car projection is
        supported for now, but the variants CC and fejer1 can be selected using
        the variant keyword. This currently defaults to CC, but will likely
    change to fejer1 in the future.
    """
    from pixell import utils as pixell_utils
    from pixell import wcsutils

    if variant.lower() == "cc":
        yo = 1
    elif variant.lower() == "fejer1":
        yo = 0
    else:
        raise ValueError("Unrecognized CAR variant '%s'" % str(variant))

    # Set up the shape/resolution
    ra_width = 2 * np.pi
    dec_width = 80 * np.pi / 180.0  ## -60 to +20
    ra_cent = 0
    dec_cent = -20
    if shape is None:
        res = np.zeros(2) + res
        shape = pixell_utils.nint(([dec_width, ra_width] / res) + (yo, 0))
    else:
        res = np.array([dec_width, ra_width]) / (np.array(shape) - (yo, 0))
    ny, nx = shape
    assert abs(res[0] * (ny - yo) - dec_width) < 1e-8, (
        "Vertical resolution does not evenly divide the sky; this is required for SHTs."
    )
    assert abs(res[1] * nx - ra_width) < 1e-8, (
        "Horizontal resolution does not evenly divide the sky; this is required for SHTs."
    )
    wcs = wcsutils.WCS(naxis=2)
    # Note the reference point is shifted by half a pixel to keep
    # the grid in bounds, from ra=180+cdelt/2 to ra=-180+cdelt/2.
    #
    wcs.wcs.crval = [0, 0]
    if include_half_pixel_offset:
        wcs.wcs.crval = [res[1] / 2 / pixell_utils.degree, 0]
    wcs.wcs.cdelt = [
        -ra_width / pixell_utils.degree / nx,
        dec_width / pixell_utils.degree / (ny - yo),
    ]
    wcs.wcs.crpix = [nx // 2, (ny - dec_cent * pixell_utils.degree / res[0]) / 2]  #
    if include_half_pixel_offset:
        wcs.wcs.crpix = [nx // 2 + 0.5, (ny + 1) / 2]  #
    wcs.wcs.ctype = ["RA---CAR", "DEC--CAR"]
    return dims + (ny, nx), wcs


def flat_field_using_pixell(
    mapdata,
    tilegrid=1.0,
):
    """
    Use the tiles module to do map tiling using scan strategy.
    Not sure how much better (if at all) this is than using background2d
    """
    from .tiles import get_medrat, get_tmap_tiles

    try:
        med_ratio = get_medrat(
            mapdata.snr,
            get_tmap_tiles(
                np.nan_to_num(mapdata.time_map) - np.nanmin(mapdata.time_map),
                tilegrid,
                mapdata.snr,
                id=f"{mapdata.freq}",
            ),
        )

        # mapdata.flux*=med_ratio
        mapdata.snr *= med_ratio
        mapdata.flatfielded = True
    except Exception as e:
        print(
            e,
            " Cannot flatfield at this time... need to update medrat algorithm for coadds",
        )

    return


def flat_field_using_photutils(
    mapdata: ProcessableMap,
    tilegrid: u.Quantity[u.deg] = 1.0 * u.deg,
    mask: np.ndarray | None = None,
    sigmaclip: float = 5.0,
    log: FilteringBoundLogger | None = None,
):
    """
    use photutils.background.Background2D to calculate the rms in tiles.
    get the rms of the tiled snr map (i.e. the rms should be 1)
    calculate the median.
    divide the snr map by the tiled rms / median.

    tilegrid is the size of the tile to use for the background estimation.
    """
    from astropy.stats import SigmaClip
    from photutils.background import Background2D, StdBackgroundRMS

    log = log if log else structlog.get_logger()
    log = log.bind(func_name="flat_field_using_photutils")
    sigmaclip = SigmaClip(sigma=sigmaclip) if sigmaclip else None
    try:
        background = Background2D(
            mapdata.snr,
            int(tilegrid.to_value(u.deg) / mapdata.map_resolution.to_value(u.deg)),
            sigma_clip=sigmaclip,
            bkg_estimator=StdBackgroundRMS(sigmaclip),
            mask=mask,
        )
        relative_rms = background.background_rms / background.background_rms_median
        mapdata.snr /= relative_rms
        mapdata.flatfielded = True
    except Exception as e:
        log.error(f"flat_field_using_photutils.failed: {e}")

    return


def make_model_source_map(
    imap: enmap.ndmap,
    sources: list[MeasuredSource],
    nominal_fwhm: u.Quantity | None = None,
    matched_filtered=False,
    verbose=False,
    cuts={},
    log=None,
):
    """
    Use source list containing fwhm_a, fwhm_b, ra, dec, flux, and orientation
    to create a model map of the sources.
    This is used to subtract the sources from the map.

    Arguments:
        - imap: enmap object
        - sources: list of source candidates
        - nominal_fwhm_arcmin: nominal fwhm in arcmin
        - matched_filtered: if True, then the fwhm is matched filtered
        - verbose: if True, then print out the sources that are not included
        - cuts: dictionary of cuts to apply to the sources
    Returns:
        - model_map: enmap object of the model map
    """
    log = log.bind(func_name="make_model_source_map")
    if len(sources) == 0:
        log.warning("make_model_source_map.no_sources", num_sources=0)
        return imap

    if matched_filtered:
        nominal_fwhm *= np.sqrt(2)

    map_res = abs(imap.wcs.wcs.cdelt[0]) * u.deg
    log.bind(nominal_fwhm=nominal_fwhm, res=map_res)
    model_params = make_2d_gaussian_model_param_table(
        imap,
        sources,
        nominal_fwhm=nominal_fwhm,
        cuts=cuts,
        verbose=verbose,
        log=log,
    )

    shape = (
        int(5 * nominal_fwhm.to_value(u.rad) / map_res.to_value(u.rad)),
        int(5 * nominal_fwhm.to_value(u.rad) / map_res.to_value(u.rad)),
    )

    log.info("make_model_source_map.make_model_image.start")
    model_map = make_model_image(
        imap.shape,
        GaussianPSF(),
        model_params,
        model_shape=shape,
    )
    log.info("make_model_source_map.make_model_image.success")
    return model_map
