import os
import os.path as op
import math
from tqdm import tqdm
import numpy as np
from pixell import enmap, utils
from scipy import ndimage, stats
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import morphology as morph
from scipy.optimize import curve_fit
import glob as glob
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u



# todo: write a function that will generate thumbnail and/or lightcurve given time and position


def radec_to_str_name(ra: float, 
                      dec: float, 
                      source_class="pointsource", 
                      observatory="SO"
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
        print("Unrecognized source class {}".format(source_class))
        print("Defaulting to [ S ]")
        source_class = "S"

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

class SourceMap(enmap.ndmap):
    """Implements a ndmap with a list of sources"""

    def __new__(cls, arr, wcs, sources):
        """Wraps a numpy and a wcslib world coordinate system object into an ndmap."""
        obj = np.asarray(arr).view(cls)
        obj.wcs = wcs.deepcopy()
        obj.sources = np.asarray(sources).view(cls)
        return obj

    def edge_map(self):
        return edge_map(self)

    def edge_dist(self):
        return obj_dist(edge_map(self), self.sources)

    def mask_col(self, mask):
        return source_in_mask(self.sources, mask)

    def cross_mask(self, crosscat, r):
        return crossmatch_mask(self.sources, crosscat, r)


def crossmatch_mask(sources, crosscat, radius):
    """Determines if source matches with masked objects

    Args:
        sources: np.array of sources [[dec, ra]] in deg
        crosscat: catalog of masked objects [[dec, ra]] in deg
        radius: radius to search for matches in arcmin

    Returns:
        mask column for sources, 1 matches with at least one source, 0 no matches

    """

    radius = radius * utils.arcmin  # convert to radians

    crosspos_ra = crosscat[:, 1] * utils.degree
    crosspos_dec = crosscat[:, 0] * utils.degree
    crosspos = np.array([crosspos_ra, crosspos_dec]).T

    # if only one source
    if len(sources.shape) == 1:
        source_ra = sources[1] * utils.degree
        source_dec = sources[0] * utils.degree
        sourcepos = np.array([[source_ra, source_dec]])
        match = utils.crossmatch(sourcepos, crosspos, radius, mode="all")
        if len(match) > 0:
            return True
        else:
            return False

    mask = np.zeros(len(sources), dtype=bool)
    for i, m in enumerate(mask):
        source_ra = sources[i, 1] * utils.degree
        source_dec = sources[i, 0] * utils.degree
        sourcepos = np.array([[source_ra, source_dec]])
        match = utils.crossmatch(sourcepos, crosspos, radius, mode="all")
        if len(match) > 0:
            mask[i] = True

    return mask


def edge_map(imap):
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


def obj_dist(mask, sources):
    """returns catalog with distances from mask

    Args:
        mask: ndmap of mask
        sources: np.array of sources [[dec, ra]] in deg

    Returns:
        array of pixel distances from mask for each object
    """

    dec = sources[:, 0]
    ra = sources[:, 1]
    dmap = enmap.enmap(distance_transform_edt(mask), mask.wcs)
    coords_rad = np.deg2rad(np.array([dec, ra]))
    ypix, xpix = enmap.sky2pix(dmap.shape, mask.wcs, coords_rad)
    ps_dist = dmap[ypix.astype(int), xpix.astype(int)]  # lookup distance from mask

    return ps_dist


def source_in_mask(sources, maskmap):
    """Determines if source is inside mask

    Args:
        sources: np.array of sources [[dec, ra]] in deg
        maskmap: ndmap of mask (zero masked, one not masked)

    Returns:
        mask column for sources, 1 masked, 0 not masked
    """

    # Check if sources are in mask
    dec = sources[:, 0]
    ra = sources[:, 1]
    coords_rad = np.deg2rad(np.array([dec, ra]))
    ypix, xpix = enmap.sky2pix(maskmap.shape, maskmap.wcs, coords_rad)
    mask = maskmap[ypix.astype(int), xpix.astype(int)]  # lookup mask value

    # Convert to binary
    mask[np.abs(mask) > 0] = 1

    # switch 0 and 1
    mask = mask.astype("bool")
    mask = ~mask
    mask = mask.astype("int")

    return mask


def source_in_map(sources, imap):
    """Determines if source is inside map and unmasked

    Args:
        sources: np.array of sources [[dec, ra]] in deg
        imap: ndmap to find sources in

    Returns:
        mask column for sources, 1 inside map, 0 outside map
    """
    dec = sources[:, 0]
    ra = sources[:, 1]
    coords_rad = np.deg2rad(np.array([dec, ra]))
    # check if source is masked
    inmap = imap.contains(coords_rad)
    values = imap.at(coords_rad[:, inmap]).astype(bool)
    inmap[inmap] = values
    return inmap


def mask_map_edge(imap, pix_num):
    """Get a mask that masking off edge pixels

    Args:
        imap:ndmap to create mask for,usually kappa map
        pix_num: pixels within this number from edge will be cutoff

    Returns:
        binary ndmap with 1 inside region, 0 outside
    """
    edge = edge_map(imap)
    dmap = enmap.enmap(morph.distance_transform_edt(edge), edge.wcs)
    mask = enmap.zeros(imap.shape, imap.wcs)
    mask[np.where(dmap > pix_num)] = 1
    return mask


def snr_map(kappa, rho):
    snr = rho / kappa**0.5
    snr[np.where(kappa <= 1.0e-3)] = 0.0
    return snr


def find_kappa_w_rho(rho_file):
    kfile = rho_file[:-8] + "kappa.fits"
    return kfile


def find_time_w_rho(rho_file):
    tfile = rho_file[:-8] + "time.fits"
    return tfile


def find_rho(ctime, arr, freq):
    rho_file = glob.glob(
        "/scratch/gpfs/snaess/actpol/maps/depth1/release/*/depth1_%s_%s_%s_rho.fits"
        % (ctime, arr, freq)
    )[0]
    return rho_file


def init_mpi(arr, randomize=False):
    """
    init mpi and scatter array

    Args:
        arr:array to scatter
        randomize:whether to randomize order before scattering

    Returns:
        chunks of array for each process
    """

    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        if randomize:
            np.random.shuffle(arr)
        chunks = np.array_split(arr, size)
    else:
        chunks = None
    chunks = comm.scatter(chunks, root=0)

    return comm, rank, size, chunks


def consolidate_cols(dirs, colnames):
    """
    consolidate columns from different directories

    Args:
        dirs:directories to consolidate
        colnames:column names to consolidate

    Returns:
        consolidated column with cat name and column
    """

    # init array for each colname given unless not a list
    cols = {}
    cols["cat"] = []
    if type(colnames) == str:
        cols[colnames] = []
    else:
        for col in colnames:
            cols[col] = []

    for subdir in tqdm(dirs):
        # list files in directory
        files = os.listdir(subdir)

        # check if fits
        files = [f for f in files if f.endswith(".fits")]

        # Check if list is empty
        if len(files) == 0:
            continue

        for f in files:
            # Read table
            t = Table.read(op.join(subdir, f))

            # Check if table is empty
            if len(t) == 0:
                continue

            # check if colname is in table
            if type(colnames) == str:
                if colnames in t.colnames:
                    cols[colnames].append(t[colnames])
            else:
                for col in colnames:
                    if col in t.colnames:
                        cols[col].append(t[col])

            # add cat name to every row
            cols["cat"].append(np.repeat(op.join(subdir, f), len(t)))

    # convert to array
    for col in cols.keys():
        cols[col] = np.concatenate(cols[col])

    return cols


def count_ps(dirs, galcut=False, edgecut=None):
    """
    count number of point source matches in each catalog

    Args:
        dirs:directories to count point source matches
        galcut:whether to cut out galactic plane
        edgecut:number of pixels to cut from edge

    Returns:
        cat name and number of sources
    """

    # init array for each colname given unless not a list
    cols = {}
    cols["cat"] = []
    cols["source"] = []
    cols["all"] = []

    for subdir in tqdm(dirs):
        # list files in directory
        files = os.listdir(subdir)

        # check if fits
        files = [f for f in files if f.endswith(".fits")]

        # Check if list is empty
        if len(files) == 0:
            continue

        for f in files:
            # Read table
            t = Table.read(op.join(subdir, f))

            if galcut:
                # mask anything with gal==0
                mask = t["gal"] == 0
                t = t[~mask]

            if edgecut is not None:
                # mask number of pix from edge
                t = t[t["dist"] > edgecut]

            # check if colname is in table
            if "source" in t.colnames:
                cols["source"].append(len(t[t["source"]]))
            else:
                cols["source"].append(np.nan)

            # count all sources
            cols["all"].append(len(t))

            # add cat name
            cols["cat"].append(op.join(subdir, f))

    return cols


def get_ps_inmap(imap, sourcecat, fluxlim=None):
    """
    get point sources in map

    Args:
        imap:ndmap to get point sources in
        sourcecat:source catalog
        fluxlim:flux limit in mJy

    Returns:
        sourcecat with sources in map
    """
    if fluxlim:
        sourcecat = sourcecat[sourcecat["fluxJy"] > fluxlim / 1000.0]
    # convert ra to -180 to 180
    sourcecat["RADeg"] = np.where(
        sourcecat["RADeg"] > 180, sourcecat["RADeg"] - 360, sourcecat["RADeg"]
    )

    sourcecat_coords = enmap.sky2pix(
        imap.shape,
        imap.wcs,
        np.array([np.deg2rad(sourcecat["decDeg"]), np.deg2rad(sourcecat["RADeg"])]),
    )
    sourcecat = sourcecat[
        (sourcecat_coords[0] > 0)
        & (sourcecat_coords[0] < imap.shape[0])
        & (sourcecat_coords[1] > 0)
        & (sourcecat_coords[1] < imap.shape[1])
    ]

    return sourcecat


def get_depth1_time_w_rho(rho_file):
    kfile = rho_file[:-8] + "time.fits"
    return kfile


def ra_pos(ra):
    """converts ra to 0<ra<360

    Args:
        ra: arr to convert

    Returns:
        ra converted to 0<ra<360
    """

    ra[ra < 0] += 360.0

    return ra


def ra_neg(ra):
    """converts ra to -180<ra<180

    Args:
        ra: arr to convert ra of

    Returns:
        ra converted to -180<ra<180
    """

    ra[ra > 180] -= 360.0

    return ra


def sky_sep(ra1, dec1, ra2, dec2):
    """calculates angular separation between two points on the sky

    Args:
        ra1: ra of first point [deg]
        dec1: dec of first point [deg]
        ra2: ra of second point [deg]
        dec2: dec of second point [deg]

    Returns:
        angular separation between two points on the sky in degrees
    """

    # convert to SkyCoord
    c1 = SkyCoord(ra1, dec1, unit=(u.deg, u.deg), frame="icrs")
    c2 = SkyCoord(ra2, dec2, unit=(u.deg, u.deg), frame="icrs")
    sep = c1.separation(c2).to(u.deg)

    return sep.value


def get_mean_flux(ra_deg, dec_deg, freq, size):
    """calculates mean flux given position and band

    Args:
        ra_deg: ra of candidate [deg]
        dec_deg:dec of candidates [deg]
        freq: frequency band, i.e. f220,f150 etc
    Returns:
        flux [mJy], snr
    """
    from filters import matched_filter_1overf
    ra = ra_deg * np.pi / 180.0
    dec = dec_deg * np.pi / 180.0
    mean_map_file = (
        "/home/snaess/project/actpol/map_coadd/20211219/release/act_daynight_%s_map.fits"
        % freq
    )
    mean_ivar_file = (
        "/home/snaess/project/actpol/map_coadd/20211219/release/act_daynight_%s_ivar.fits"
        % freq
    )
    mean_map = enmap.read_map(
        mean_map_file, box=[[dec - size, ra - size], [dec + size, ra + size]]
    )[0, :, :]
    mean_ivar = enmap.read_map(
        mean_ivar_file, box=[[dec - size, ra - size], [dec + size, ra + size]]
    )
    wcs = mean_map.wcs
    shape = mean_map.shape
    rho, kappa = matched_filter_1overf(mean_map, mean_ivar, freq, size_deg=0.5)
    flux_map = rho / kappa
    snr_map = rho / kappa**0.5
    flux_map[np.where(kappa == 0.0)] = 0.0
    snr_map[np.where(kappa == 0.0)] = 0.0
    flux = flux_map.at([dec, ra])
    snr = snr_map.at([dec, ra])
    return flux, snr


def thumbnail(
    ra: float, dec: float, ctime: int, freq: str, arr: str, mapdir: str, radius: float
):
    # Parse position
    pos = np.deg2rad(np.array([dec, ra]))

    # Create pixbox around source
    subdir = str(ctime)[:5]
    imap = f"{mapdir}/{subdir}/depth1_{ctime}_{arr}_{freq}_map.fits"
    shape_full, wcs_full = enmap.read_map_geometry(imap)
    pixbox = enmap.neighborhood_pixboxes(
        shape_full, wcs_full, pos.T, radius * utils.degree
    )

    # Read in stamp maps
    stamp_rho = enmap.read_map(
        f"{mapdir}/{subdir}/depth1_{ctime}_{arr}_{freq}_rho.fits", pixbox=pixbox
    )[0]
    stamp_kappa = enmap.read_map(
        f"{mapdir}/{subdir}/depth1_{ctime}_{arr}_{freq}_kappa.fits", pixbox=pixbox
    )[0]
    stamp_kappa = sanitize_kappa(stamp_kappa)
    flux = stamp_rho / stamp_kappa

    return flux


def sanitize_kappa(kappa, tol=1e-4, inplace=False):
    return np.maximum(kappa, np.max(kappa) * tol)


def find_map_ctime(ctime, mapdir="/scratch/gpfs/snaess/actpol/maps/depth1/release"):
    """
    Find map that contains ctime

    Args:
        ctime:ctime to find map for

    Returns:
        ctime of map
    """

    # Find subdirectory clostest to ctime
    subdir = str(ctime)[:5]

    # list possible map ctimes
    maps = [f for f in os.listdir(op.join(mapdir, subdir)) if f.endswith("map.fits")]
    ctimes = np.unique([int(f.split("_")[1]) for f in maps])

    # find closest ctime that is less than ctime
    ctimes = ctimes[ctimes < ctime]

    ctime_map = ctimes[np.argmin(np.abs(ctimes - ctime))]

    return ctime_map


def ctime_to_pos(ctime, freq, arr, mapdir):
    # Find subdirectory clostest to ctime
    subdir = str(ctime)[:5]

    # list possible ctimes
    mapdir = mapdir
    maps = [f for f in os.listdir(op.join(mapdir, subdir)) if f.endswith("map.fits")]
    ctimes = np.unique([int(f.split("_")[1]) for f in maps])

    # find closest ctime that is less than tmid
    ctimes = ctimes[ctimes < ctime]
    if len(ctimes) == 0:
        print(f"No maps in {op.join(mapdir, subdir)}")
        return None, None, None
    ctime_map = ctimes[np.argmin(np.abs(ctimes - ctime))]
    print(f"Using ctime {ctime}")

    # get deltat a midpoint
    deltat = ctime - ctime_map

    tmap_path = op.join(mapdir, subdir, f"depth1_{ctime_map}_{arr}_{freq}_time.fits")

    if not op.exists(tmap_path):
        print(f"No time map for {tmap_path}")
        return None, None, None

    tmap = enmap.read_map(tmap_path)

    # Find if there is a time within deltat
    if deltat < np.min(tmap[tmap > 0]) or deltat > np.max(tmap):
        print(f"No time within {deltat} of {ctime} in {tmap_path}")
        return None, None, None

    # Make sure there exists a time within 1 second of deltat
    if np.min(np.abs(tmap - deltat)) > 1:
        print(f"No time within 1 second of {deltat} in {tmap_path}")
        return None, None, None

    # If deltat is in the map, find the closest time
    pos = np.unravel_index(np.argmin(np.abs(tmap - deltat)), tmap.shape)
    pos = tmap.pix2sky(pos)
    dec = pos[0] * 180 / np.pi
    ra = pos[1] * 180 / np.pi

    return ctime_map, ra, dec


def radial_profile(data, center_pix, mask, bin_width_arcsec):
    y, x = np.indices((data.shape))
    r = (
        np.sqrt((x - center_pix[0]) ** 2 + (y - center_pix[1]) ** 2)
        * 30.0
        / bin_width_arcsec
    )
    r = r.astype("int")
    data_masked = data * mask
    tbin = np.bincount(r.ravel(), data_masked.ravel())
    nr = np.bincount(r.ravel(), mask.ravel())
    radialprofile = tbin / nr
    return radialprofile


def merge_result(flux_data):  # merge pa4 and pa5 result
    result_merged = np.array(
        [[225.0 * 1e9, 0.0, 0.0], [150.0 * 1e9, 0.0, 0.0], [98.0 * 1e9, 0.0, 0.0]]
    )
    result_merged[0, 1] = flux_data["pa4_f220_flux"][0]
    result_merged[0, 2] = flux_data["pa4_f220_flux"][1]
    result_merged[2, 1] = flux_data["pa5_f090_flux"][0]
    result_merged[2, 2] = flux_data["pa5_f090_flux"][1]
    ivar_150 = 0.0
    for key in ["pa4_f150_flux", "pa5_f150_flux"]:
        if flux_data[key][1] != 0:
            result_merged[1, 1] += flux_data[key][0] / flux_data[key][1] ** 2.0
            ivar_150 += 1 / flux_data[key][1] ** 2.0
    if ivar_150 != 0.0:
        result_merged[1, 1] /= ivar_150
        result_merged[1, 2] = 1 / ivar_150**0.5
    return result_merged


def merge_result_all(flux_data):  # merge same band results from all array
    result_merged = np.array(
        [
            [225.0 * 1e9, 0.0, 0.0],
            [150.0 * 1e9, 0.0, 0.0],
            [98.0 * 1e9, 0.0, 0.0],
            [40 * 1e9, 0.0, 0.0],
            [30 * 1e9, 0.0, 0.0],
        ]
    )
    result_merged[0, 1] = flux_data["pa4_f220_flux"][0]
    result_merged[0, 2] = flux_data["pa4_f220_flux"][1]
    result_merged[3, 1] = flux_data["pa7_f040_flux"][0]
    result_merged[3, 2] = flux_data["pa7_f040_flux"][1]
    result_merged[4, 1] = flux_data["pa7_f030_flux"][0]
    result_merged[4, 2] = flux_data["pa7_f030_flux"][1]
    ivar_150 = 0.0
    for key in ["pa4_f150_flux", "pa5_f150_flux", "pa6_f150_flux"]:
        if flux_data[key][1] != 0:
            result_merged[1, 1] += flux_data[key][0] / flux_data[key][1] ** 2.0
            ivar_150 += 1 / flux_data[key][1] ** 2.0
    if ivar_150 != 0.0:
        result_merged[1, 1] /= ivar_150
        result_merged[1, 2] = 1 / ivar_150**0.5
    ivar_090 = 0.0
    for key in ["pa5_f090_flux", "pa6_f090_flux"]:
        if flux_data[key][1] != 0:
            result_merged[2, 1] += flux_data[key][0] / flux_data[key][1] ** 2.0
            ivar_090 += 1 / flux_data[key][1] ** 2.0
    if ivar_090 != 0.0:
        result_merged[2, 1] /= ivar_090
        result_merged[2, 2] = 1 / ivar_090**0.5
    return result_merged


def calculate_alpha(data):
    alpha = math.log(data[0, 1] / data[1, 1], data[0, 0] / data[1, 0])
    A = data[0, 1] / (data[0, 0] ** alpha)
    err = (
        (data[0, 2] / (data[0, 1] * np.log(data[0, 0] / data[1, 0]))) ** 2.0
        + (data[1, 2] / (data[1, 1] * np.log(data[0, 0] / data[1, 0]))) ** 2.0
    ) ** 0.5
    return A, alpha, err


def get_spectra_index(flux_data, ctime, data_type="pa4pa5_or_third"):
    def func(x, a, b):
        y = a * x**b
        return y

    if data_type == "pa4pa5_or_third":
        merged_data = merge_result(flux_data)
        result_merged_sel = merged_data[np.where(merged_data[:, 1] > 0.0)]
        if result_merged_sel.shape[0] == 3:
            pars, cov = curve_fit(
                f=func,
                xdata=result_merged_sel[:, 0],
                ydata=result_merged_sel[:, 1],
                sigma=result_merged_sel[:, 2],
                absolute_sigma=True,
                p0=[0, 1],
                maxfev=5000,
            )
            alpha = pars[1]
            err = np.sqrt(cov[1, 1])
        elif result_merged_sel.shape[0] == 2:
            amp, alpha, err = calculate_alpha(result_merged_sel)
        else:
            if ctime < 1580515200:
                pa6_data = np.array(
                    [
                        [
                            150 * 1e9,
                            flux_data["pa6_f150_flux"][0],
                            flux_data["pa6_f150_flux"][1],
                        ],
                        [
                            98 * 1e9,
                            flux_data["pa6_f090_flux"][0],
                            flux_data["pa6_f090_flux"][1],
                        ],
                    ]
                )
                pa6_data_sel = pa6_data[np.where(pa6_data[:, 1] > 0.0)]
                if pa6_data_sel.shape[0] == 2:
                    amp, alpha, err = calculate_alpha(pa6_data_sel)
                else:
                    alpha = 0
                    err = 0
            else:
                pa7_data = np.array(
                    [
                        [
                            40 * 1e9,
                            flux_data["pa7_f040_flux"][0],
                            flux_data["pa7_f040_flux"][1],
                        ],
                        [
                            30 * 1e9,
                            flux_data["pa7_f030_flux"][0],
                            flux_data["pa7_f030_flux"][1],
                        ],
                    ]
                )
                pa7_data_sel = pa7_data[np.where(pa7_data[:, 1] > 0.0)]
                if pa7_data_sel.shape[0] == 2:
                    amp, alpha, err = calculate_alpha(pa7_data_sel)
                else:
                    alpha = 0
                    err = 0
    if data_type == "all":
        merged_data = merge_result_all(flux_data)
        result_merged_sel = merged_data[np.where(merged_data[:, 1] > 0.0)]
        if result_merged_sel.shape[0] > 2:
            pars, cov = curve_fit(
                f=func,
                xdata=result_merged_sel[:, 0],
                ydata=result_merged_sel[:, 1],
                sigma=result_merged_sel[:, 2],
                absolute_sigma=True,
                p0=[0, 1],
                maxfev=5000,
            )
            alpha = pars[1]
            err = np.sqrt(cov[1, 1])
        elif result_merged_sel.shape[0] == 2:
            amp, alpha, err = calculate_alpha(result_merged_sel)
        else:
            alpha = 0
            err = 0
    return alpha, err


def get_spectra_index_general(flux_data):
    """
    calculate spectra index, flux_data[:,0] is freq in Hz, flux_data[:,1] is flux, flux_data[:,2] is dflux
    """

    def func(x, a, b):
        y = a * x**b
        return y

    if flux_data.shape[0] > 2:
        pars, cov = curve_fit(
            f=func,
            xdata=flux_data[:, 0],
            ydata=flux_data[:, 1],
            sigma=flux_data[:, 2],
            absolute_sigma=True,
            maxfev=5000,
        )
        alpha = pars[1]
        err = np.sqrt(cov[1, 1])
    elif flux_data.shape[0] == 2:
        amp, alpha, err = calculate_alpha(flux_data)
    else:
        alpha = 0.0
        err = 0.0
    return alpha, err


def rtsumofsquares(arr):
    """
    Calculate root sum of squares of array
    """
    return np.sqrt(np.sum(np.square(arr)))


def weightedmean(arr, err):
    """
    Calculate weighted mean of array
    """
    # If all errors are zero, return mean
    if np.all(err == 0.0):
        return np.mean(arr)
    else:
        # get rid of zero errors
        arr = arr[err != 0.0]
        err = err[err != 0.0]
        return np.sum(arr / err**2.0) / np.sum(1 / err**2.0)


def pos_variance(ra, dec, ra_data, dec_data):
    """
    Calculate variance of array
    """
    if len(ra_data) == 1:
        return np.nan
    count = 0
    var = 0
    for j in range(len(ra_data)):
        var += (ra_data[j] - ra) ** 2.0 + (dec_data[j] - dec) ** 2.0
        count += 1
    var /= count - 1
    return var**0.5


def bindata(x, y, yerr, binsize):
    # bin data
    bins = np.arange(np.min(x), np.max(x) + binsize, binsize)
    bin_idx = np.digitize(x, bins)

    # Get weighted mean, error, and bin centers
    bin_means = np.array(
        [weightedmean(y[bin_idx == i], yerr[bin_idx == i]) for i in range(1, len(bins))]
    )
    bin_err = np.array(
        [rtsumofsquares(yerr[bin_idx == i]) for i in range(1, len(bins))]
    )
    bin_centers = np.array([np.mean(x[bin_idx == i]) for i in range(1, len(bins))])

    # weightedmean_func = lambda y_data: weightedmean(y_data, yerr)
    # bin_means, bin_edges, binnumber = binned_statistic(x, y, statistic=weightedmean_func, bins=bin_edges)
    # bin_width = (bin_edges[1] - bin_edges[0])
    # bin_centers = bin_edges[1:] - bin_width / 2
    # bin_err = binned_statistic(x, yerr, statistic=rtsumofsquares, bins=nbins)[0]

    return np.array([bin_centers, bin_means, bin_err])


def mjy2lum(flux, freq, dist, fluxerr=None):
    """
    Calculate luminosity

    Args:
        flux: flux in mJy
        freq: frequency in GHz
        dist: distance in pc
        fluxerr: flux error in mJy

    Returns:
        luminosity in Watts
        luminosity error in Watts if fluxerr provided
    """
    luminosity = 4 * np.pi * (dist * 3.086e16) ** 2.0 * flux * 1e-26 * freq * 1e9
    if fluxerr is None:
        return luminosity
    else:
        lum_err = luminosity * (fluxerr / flux)
        return luminosity, lum_err


def ctime2datetime(ctime):
    """
    Convert ctime to datetime

    Args:
        ctime: ctime to convert

    Returns:
        datetime
    """
    from datetime import datetime

    return datetime.fromtimestamp(ctime)


def datetime2ctime(dt):
    """
    Convert datetime to ctime

    Args:
        dt: datetime to convert

    Returns:
        ctime
    """
    from datetime import datetime

    return datetime.timestamp(dt)


def q(p, mu=0, std=1):
    c0 = stats.norm.cdf(0, loc=mu, scale=std)
    return stats.norm.ppf(p * (1 - c0) + c0, loc=mu, scale=std)
