import os
import os.path as op
import numpy as np
from pixell import enmap, utils
from scipy import ndimage
from scipy.ndimage import morphology as morph
from scipy.stats import norm
import warnings
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
import pandas as pd

from sotdplib import filters, tiles, tools, inputs, cmb_minor_planets

from pydantic import BaseModel
from typing import List

# import enlib if needed
try:
    from enlib import planet9
except ModuleNotFoundError:
    print("enlib not found, will not be able to mask planets")
    pass


class Depth1:
    def __init__(
        self,
        intensity_map: enmap,
        inverse_variance_map: enmap,
        rho_map: enmap,
        kappa_map: enmap,
        time_map: enmap,
        wafer_name: str,
        freq: str,
        map_ctime: float,
    ):
        self.intensity_map = clean_map(intensity_map, inverse_variance_map)
        self.inverse_variance_map = inverse_variance_map
        self.rho_map = clean_map(rho_map, inverse_variance_map)
        self.kappa_map = clean_map(kappa_map, inverse_variance_map)
        self.time_map = clean_map(time_map, inverse_variance_map)
        self.wafer_name = wafer_name
        self.freq = freq
        self.map_ctime = map_ctime

    def snr(self):
        return get_snr(self.rho_map, self.kappa_map)

    def flux(self):
        return get_flux(self.rho_map, self.kappa_map)

    def dflux(self):
        return get_dflux(self.kappa_map)

    def mask_edge(self, edgecut):
        return mask_edge(self.kappa_map, edgecut)

    def mask_dustgal(self, galmask):
        return mask_dustgal(self.kappa_map, galmask)

    def mask_planets(self):
        return mask_planets(self.map_ctime, self.time_map)

    def masked_map(self, imap, edgecut, galmask):
        return get_masked_map(
            imap, self.mask_edge(edgecut), galmask, self.mask_planets()
        )

    def matched_filter_submap(self, ra, dec, sourcemask=None, size=0.5):
        return get_matched_filter_submap(
            self,
            ra,
            dec,
            sourcemask,
            size,
        )


def kappa_clean(kappa: enmap.ndmap, rho: enmap.ndmap):
    kappa = np.maximum(kappa, np.max(kappa) * 1e-3)
    kappa[np.where(rho == 0.0)] = 0.0
    return kappa


def clean_map(imap: enmap.ndmap, inverse_variance: enmap.ndmap):
    imap[inverse_variance < (np.max(inverse_variance) * 0.01)] = 0
    return imap


def get_snr(rho: enmap.ndmap, kappa: enmap.ndmap):
    # supress divide by zero warning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        snr = rho / kappa**0.5
    snr[np.where(kappa == 0.0)] = 0.0
    return snr


def get_flux(rho: enmap, kappa: enmap):
    # supress divide by zero warning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        flux = rho / kappa
    flux[np.where(kappa == 0.0)] = 0.0
    return flux


def get_dflux(kappa: enmap):
    # supress divide by zero warning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dflux = kappa**-0.5
    dflux[np.where(kappa == 0.0)] = 0.0
    return dflux


def get_tmap_tiles(tmap: enmap.ndmap, grid_deg: float, zeromap: enmap.ndmap, id=None):
    tile_map = tiles.tiles_t_quick(tmap, grid_deg, id=id)
    tile_map[np.where(zeromap == 0.0)] = 0.0
    return tile_map


def edge_map(imap: enmap.ndmap):
    """Finds the edges of a map

    Args:
        imap: ndmap to find edges of

    Returns:
        binary ndmap with 1 inside region, 0 outside
    """

    edge = enmap.enmap(imap, imap.wcs)  # Create map geometry
    edge[np.abs(edge) > 0] = 1  # Convert to binary
    edge = ndimage.binary_fill_holes(edge)  # Fill holes

    return enmap.enmap(edge.astype("ubyte"), imap.wcs)


def mask_edge(imap: enmap.ndmap, pix_num: int):
    """Get a mask that masking off edge pixels

    Args:
        imap:ndmap to create mask for,usually kappa map
        pix_num: pixels within this number from edge will be cutoff

    Returns:
        binary ndmap with 1 == unmasked, 0 == masked
    """
    edge = edge_map(imap)
    dmap = enmap.enmap(morph.distance_transform_edt(edge), edge.wcs)
    mask = enmap.zeros(imap.shape, imap.wcs)
    mask[np.where(dmap > pix_num)] = 1
    return mask


def mask_dustgal(imap: enmap.ndmap, galmask: enmap.ndmap):
    return enmap.extract(galmask, imap.shape, imap.wcs)


def mask_planets(
    ctime,
    tmap: enmap.ndmap,
    planets=["Venus", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"],
):
    mjd_min = utils.ctime2mjd(ctime)
    mjd_max = utils.ctime2mjd(ctime + np.amax(tmap))
    mjds = np.linspace(mjd_min, mjd_max, 10)
    try:
        mask = planet9.build_planet_mask(
            tmap.shape, tmap.wcs, planets, mjds, r=50 * utils.arcmin
        )
        mask_planet = mask.copy()
        mask_planet[np.where(mask == False)] = 1.0
        mask_planet[np.where(mask == True)] = 0.0
    except Exception:
        mask_planet = np.ones(tmap.shape)

    return mask_planet


def blazar_crossmatch(pos, tol=0.5):
    """
    Mask blazar 3C 454.3 using icrs position from SIMBAD

    Args:
        pos:  np.array of positions [[dec, ra]] in deg
        tol: tolerance in deg

    Returns:
        1 in blazar, 0 if not in blazar
    """
    blazar_pos = SkyCoord("22h53m57.7480438728s", "+16d08m53.561508864s", frame="icrs")
    can_pos = SkyCoord(pos[:, 1], pos[:, 0], frame="icrs", unit="deg")
    sep = blazar_pos.separation(can_pos).to(u.deg)
    match = []
    for s in sep:
        if s.value < tol:
            match.append(1)
        else:
            match.append(0)
    return np.array(match).astype(bool)


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


def get_matched_filter_submap(
    map_object: Depth1, ra: float, dec: float, sourcemask=None, size=0.5
):
    """
    calculate rho and kappa maps given a ra and dec

    Args:
        map_object: Depth1 object
        ra: right ascension in deg
        dec: declination in deg
        sourcemask: point sources to mask, default=None
        size: size of submap in deg, default=0.5

    Returns:
        rho and kappa maps
    """

    sub = filters.get_submap(map_object.intensity_map, ra, dec, size)
    sub_inverse_variance = filters.get_submap(
        map_object.inverse_variance_map, ra, dec, size
    )
    sub_tmap = filters.get_submap(map_object.time_map, ra, dec, size)
    rho, kappa = filters.matched_filter(
        sub,
        sub_inverse_variance,
        map_object.wafer_name,
        map_object.freq,
        ra,
        dec,
        source_cat=sourcemask,
    )

    return Depth1(
        sub,
        sub_inverse_variance,
        rho,
        kappa,
        sub_tmap,
        map_object.wafer_name,
        map_object.freq,
        map_object.map_ctime,
    )


def get_medrat(snr: enmap, tiledmap):
    """
    gets median ratio for map renormalization given tiles

    Args:
         snr: snr map
         tiledmap: tiles from tmap to get ratio from

    Returns:
        median ratio for each tile
    """
    t = tiledmap.astype(int)
    med0 = norm.ppf(0.75)
    snr2 = snr**2
    medians = ndimage.median(snr2, labels=t, index=np.arange(np.max(t + 1)))
    median_map = medians[t]
    # supress divide by zero warning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        med_ratio = med0 / median_map**0.5
    med_ratio[np.where(t == 0)] = 0.0
    return med_ratio


def source_detection(snr: enmap, flux: enmap, threshold=5, verbosity=0):
    """
    Detects sources and returns center of mass by flux

    Args:
         snr: signal-to-noise ratio map for detection
         flux: flux map for position
         threshold: signal detection threshold
         mapdata: other maps to include in output
         tags: data to include in every line, such as ctime or map name

    Returns:
        ra and dec of each source in deg

    """

    labels, nlabel = ndimage.label(snr > threshold)
    cand_pix = ndimage.center_of_mass(flux, labels, np.arange(nlabel) + 1)
    cand_pix = np.array(cand_pix)
    ra = []
    dec = []
    for pix in cand_pix:
        pos = flux.pix2sky(pix)
        d = pos[0] * 180.0 / np.pi
        if d < -90.0:
            d = 360.0 + d
        if d > 90.0:
            d = 360.0 - d
        dec.append(d)
        ra.append(pos[1] * 180.0 / np.pi)
    if len(ra) == 0:
        if verbosity > 0:
            print("did not find any candidates")

    return ra, dec


def asteroid_cut(df, ast_dir, ast_tol, ast_orbits, verbosity=0):
    # create subset with just candidates, so we don't need to waste time checking asteroid on sources
    ocat_can = df[~df["source"]]

    # Cut asteroids if there are remaining candidates
    if len(ocat_can) > 0:
        asteroids = cmb_minor_planets.check_loc_ast(
            np.deg2rad(ocat_can["ra"]).to_numpy(),
            np.deg2rad(ocat_can["dec"]).to_numpy(),
            ocat_can["ctime"].to_numpy(),
            ast_dir,
            tol=ast_tol,
            orbits=ast_orbits,
        )

        # if one dimension, convert to array
        if len(ocat_can) == 1:
            asteroids = np.array([asteroids])

        # remove asteroids
        ocat_can = ocat_can[~asteroids]

        # add back in sources
        df = pd.concat([ocat_can, df[df["source"]]], ignore_index=True)

        # count asteroid cut
        ast_cut = np.sum(asteroids)
    else:
        ast_cut = 0

    if verbosity > 0:
        print("number of candidates after asteroid cut: ", len(ocat_can))
        print(" ")

    return df, ast_cut


def recalc_source_detection(snr_sub, flux_sub, ra, dec, detection_threshold):
    """
    returns new ra/dec of source if source is detected in submap, otherwise returns original ra/dec
    Also returns bool of whether source is detected
    """

    ra_new, dec_new = source_detection(snr_sub, flux_sub, detection_threshold)

    if not ra_new:
        return ra, dec, False

    else:
        # calculate separation from original position. Reject sources not within 1 arcmin
        sep = tools.sky_sep(ra, dec, ra_new, dec_new) * 60.0  # in arcmin
        # continue if no sources are within 1 arcmin
        if np.all(sep > 1.0):
            return ra, dec, False
        # only keep source closest to original position
        else:
            ra_new = ra_new[np.argmin(sep)]
            dec_new = dec_new[np.argmin(sep)]

            return ra_new, dec_new, True


def perform_matched_filter(
    data: Depth1, ra_can, dec_can, source, sourcemask, detection_threshold
):
    mf = np.array(len(ra_can) * [False])
    renorm = np.array(len(ra_can) * [False])
    ra_arr = np.array(len(ra_can) * [np.nan])
    dec_arr = np.array(len(ra_can) * [np.nan])
    for i in range(len(ra_can)):
        if source[i]:
            ra_arr[i] = ra_can[i]
            dec_arr[i] = dec_can[i]

        else:
            ra = ra_can[i]
            dec = dec_can[i]
            data_sub = data.matched_filter_submap(ra, dec, sourcemask, size=0.5)

            # check if candidate still exists
            snr_sub = data_sub.snr()
            flux_sub = data_sub.flux()
            ra, dec, pass_matched_filter = recalc_source_detection(
                snr_sub, flux_sub, ra, dec, detection_threshold
            )
            mf[i] = pass_matched_filter

            if pass_matched_filter:
                # renormalization if matched filter is successful
                snr_sub_renorm = filters.renorm_ms(
                    snr_sub, data.wafer_name, data.freq, ra, dec, sourcemask
                )
                ra, dec, pass_renorm = recalc_source_detection(
                    snr_sub_renorm, flux_sub, ra, dec, detection_threshold
                )
                renorm[i] = pass_renorm

            ra_arr[i] = ra
            dec_arr[i] = dec

    return ra_arr, dec_arr, mf, renorm


def init_pipeline(f):
    path = f[:-8]
    imap = enmap.read_map(f)[0]
    inverse_variance = enmap.read_map(path + "inverse_variance.fits")
    rho = enmap.read_map(path + "rho.fits")[0]
    kappa = enmap.read_map(path + "kappa.fits")[0]
    time = enmap.read_map(path + "time.fits")
    arr = f.split("/")[-1].split("_")[2]
    freq = f.split("/")[-1].split("_")[3]
    ctime = float(f.split("/")[-1].split("_")[1])
    data = Depth1(imap, inverse_variance, rho, kappa, time, arr, freq, ctime)
    return data


def get_unprocessed_maps(map_path, odir, output="dirs", getall=False):
    # Bad maps
    badmaps = inputs.get_badmaps()

    dirs = os.listdir(map_path)
    # delete files that are not directories
    dirs = [d for d in dirs if op.isdir(op.join(map_path, d))]
    # sort dirs
    dirs.sort()

    maps_total = 0
    maps_processed = 0
    maps_unprocessed = 0
    subdir_unprocessed = []
    files_unprocessed = []
    for subdir in dirs:

        # list files with map extension in directory
        files = os.listdir(op.join(map_path, subdir))

        # only include files with map extension
        files = [f for f in files if f.endswith("map.fits")]

        # cut bad ctimes
        files = [f for f in files if f.split("_")[1] not in badmaps]

        # only files that have f220, f150, f090, f030, f040 in name
        files = [
            f
            for f in files
            if "f220" in f or "f150" in f or "f090" in f or "f040" in f or "f030" in f
        ]

        # Check if subdir is empty
        if len(files) == 0:
            continue

        for f in files:

            maps_total += 1

            # check if output file already exists
            if op.exists(op.join(odir, subdir, f[:-8] + "ocat.fits")):
                maps_processed += 1
                continue

            else:
                maps_unprocessed += 1
                subdir_unprocessed.append(subdir)
                if not getall:
                    files_unprocessed.append(op.join(map_path, subdir, f))
            if getall:
                files_unprocessed.append(op.join(map_path, subdir, f))

    if output == "dirs":
        return list(set(subdir_unprocessed))
    elif output == "files":
        return files_unprocessed


def radec_to_str_name(
    ra: float, dec: float, source_class="pointsource", observatory="SO"
):
    """
    ## stolen from spt3g_software -AF

    Convert RA & dec (in radians) to IAU-approved string name.

    Arguments
    ---------
    ra : float
        Source right ascension in radians
    dec : float
        Source declination in radians
    source_class : str
        The class of source to which this name is assigned.  Supported classes
        are ``pointsource``, ``cluster`` or ``transient``.  Shorthand class
        names for these sources are also allowed (``S``, ``CL``, or ``SV``,
        respectively).  Alternatively, the class can be ``None`` or ``short``,
        indicating a simple source identifier in the form ``"HHMM-DD"``.

    Returns
    -------
    name : str
        A unique identifier for the source with coordinates truncated to the
        appropriate number of significant figures for the source class.
    """
    from astropy.coordinates.angles import Angle

    source_class = str(source_class).lower()
    if source_class in ["sv", "t", "tr", "transient", "v", "var", "variable"]:
        source_class = "SV"
    elif source_class in ["c", "cl", "cluster"]:
        source_class = "CL"
    elif source_class in ["s", "p", "ps", "pointsource", "point_source"]:
        source_class = "S"
    elif source_class in ["none", "short"]:
        source_class = "short"
    else:
        core.log_fatal("Unrecognized source class {}".format(source_class))

    # ra in (0, 360)
    ra = np.mod(ra, 360)

    opts = dict(sep="", pad=True, precision=3)
    rastr = Angle(ra * u.deg).to_string(**opts, unit="hour")
    decstr = Angle(dec * u.deg).to_string(alwayssign=True, **opts)

    if source_class == "SV":
        rastr = rastr[:8]
        decstr = decstr.split(".")[0]
    elif source_class == "CL":
        rastr = rastr[:4]
        decstr = decstr[:5]
    elif source_class == "S":
        rastr = rastr.split(".")[0]
        decr = "{:.3f}".format(dec * 60).split(".")[1]
        decstr = decstr[:5] + ".{}".format(decr[:1])
    elif source_class == "short":
        rastr = rastr[:4]
        decstr = decstr[:3]
        return "{}{}".format(rastr, decstr)

    name = "{}-{} J{}{}".format(observatory, source_class, rastr, decstr)
    return name


class SourceCandidate(BaseModel):
    ra: float
    dec: float
    flux: float
    dflux: float
    snr: float
    ctime: float
    sourceID: str
    match_filtered: bool
    renormalized: bool
    catalog_crossmatch: bool



def depth1_pipeline(
    data: Depth1,
    grid: float,
    edgecut: int,
    galmask: enmap.ndmap,
    sourcemask=None,
    detection_threshold=5,
    sourcecut=5,
    ast_dir=None,
    ast_orbits=None,
    ast_tol=10.0 * utils.arcmin,
    verbosity=1,
):
    """
    Pipeline for transient detection

    Args:
        data: Depth1 object
        grid: gridsize in deg
        edgecut: number of pixels to cut from edge
        galmask: galaxy mask
        sourcemask: source catalog
        detection_threshold: snr threshold for initial source detection
        sourcecut: crossmatch radius for point source cut in arcminutes
        ast_dir: path to asteroid ephemeris directory
        ast_orbits: asteroid orbits
        ast_tol: tolerance for asteroid cut in arcmin
        verbosity: verbosity level

    Returns:
        astropy table of candidates
    """

    

    # check if map is all zeros
    if np.all(data.intensity_map == 0.0):
        if verbosity > 0:
            print("map is all zeros, skipping")

        return []

    # get galaxy, planet, edge masked flux, snr, tmap
    flux = data.masked_map(data.flux(), edgecut, galmask)
    snr = data.masked_map(data.snr(), edgecut, galmask)
    dflux = data.dflux()

    # renormalize maps using tile method
    med_ratio = get_medrat(
        snr,
        get_tmap_tiles(
            data.time_map,
            grid,
            snr,
            id=f"{data.map_ctime}_{data.wafer_name}_{data.freq}",
        ),
    )
    flux *= med_ratio
    snr *= med_ratio

    # initial point source detection
    ra_can, dec_can = source_detection(
        snr,
        flux,
        detection_threshold,
        verbosity=verbosity - 1,
    )
    if not ra_can:
        if verbosity > 0:
            print("did not find any candidates")
        return []
    if verbosity > 0:
        print("number of initial candidates: ", len(ra_can))

    # separate sources
    if sourcemask:
        sources = sourcemask[
            sourcemask["fluxJy"] > (inputs.get_sourceflux_threshold(data.freq) / 1000.0)
        ]
        catalog_match = tools.crossmatch_mask(
            np.array([dec_can, ra_can]).T,
            np.array([sources["decDeg"], sources["RADeg"]]).T,
            sourcecut,
        )
    else:
        catalog_match = np.array(len(ra_can) * [False])

    # mask blazar
    catalog_match = np.logical_or(
        catalog_match, blazar_crossmatch(np.array([dec_can, ra_can]).T)
    )

    if verbosity > 0:
        print(
            "number of candidates after point source/blazar cut: ",
            len(ra_can) - np.sum(catalog_match),
        )

    # matched filter and renormalization
    ra_new, dec_new, mf_sub, renorm_sub = perform_matched_filter(
        data, ra_can, dec_can, catalog_match, sourcemask, detection_threshold
    )
    if verbosity > 0:
        print("number of candidates after matched filter: ", np.sum(mf_sub))
        print("number of candidates after renormalization: ", np.sum(renorm_sub))

    coords = np.array(np.deg2rad([dec_new, ra_new]))
    source_string_names = [radec_to_str_name(ra_new[i], dec_new[i]) for i in range(len(ra_new))]
    print(source_string_names)
    
    flux_new = flux.at(coords, order=0, mask_nan=True)
    dflux_new = dflux.at(coords, order=0, mask_nan=True)
    snr_new = snr.at(coords, order=0, mask_nan=True)
    ctime_new = data.time_map.at(coords, order=0, mask_nan=True) + np.ones(len(ra_new)) * data.map_ctime
    source_candidates = []
    for i in range(len(ra_new)):
        source_candidates.append(
            SourceCandidate(ra=ra_new[i]%360,
                            dec=dec_new[i],
                            flux=flux_new[i],
                            dflux=dflux_new[i],
                            snr=snr_new[i],
                            ctime=ctime_new[i],
                            sourceID=source_string_names[i],
                            match_filtered=mf_sub[i],
                            renormalized=renorm_sub[i],
                            catalog_crossmatch=catalog_match[i],
                           )
        )
    
    # count sources
    source_count = len(source_candidates)

    '''
    # convert to pandas df
    ocat = pd.DataFrame.from_dict(source_candidates.dict())

    # asteroid cut
    if ast_dir:
        ocat, ast_cut = asteroid_cut(
            ocat, ast_dir, ast_tol, ast_orbits, verbosity=verbosity - 1
        )
    
    # convert to astropy table
    ocat = Table.from_pandas(ocat)

    # add units
    ocat["ra"].unit = u.deg
    ocat["dec"].unit = u.deg
    ocat["flux"].unit = u.mJy
    ocat["dflux"].unit = u.mJy
    ocat["ctime"].unit = u.s

    # add metadata
    ocat.meta["map_ctime"] = data.map_ctime
    ocat.meta["freq"] = data.freq
    ocat.meta["array"] = data.wafer_name
    ocat.meta["grid"] = grid
    ocat.meta["edgecut"] = edgecut
    ocat.meta["thresh"] = detection_threshold
    ocat.meta["ninit"] = len(ra_can)
    ocat.meta["mfcut"] = len(ra_can) - mfcount
    ocat.meta["nsource"] = source_count
    if ast_dir:
        ocat.meta["asttol"] = np.rad2deg(ast_tol) * 60
        ocat.meta["nast"] = ast_cut
    '''
    return source_candidates
    
