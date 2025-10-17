import glob as glob
import math
import os
import os.path as op
from typing import Union

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from pixell import enmap
from pixell import utils as pixell_utils
from scipy import stats
from scipy.ndimage import distance_transform_edt
from scipy.optimize import curve_fit
from tqdm import tqdm


def radec_to_str_name(
    ra: float, dec: float, source_class="pointsource", observatory="SO"
):
    """
    Convert RA & dec (in degrees) to IAU-approved string name.

    Arguments
    ---------
    ra : float
        Source right ascension in degrees
    dec : float
        Source declination in degrees
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


def ctime_to_local_time(ctimes, tz_loc="America/Santiago"):
    from datetime import datetime, timezone

    import pytz

    # Define the time zone for Atacama, Chile
    act_tz = pytz.timezone(tz_loc)
    local_time = [
        datetime.fromtimestamp(ct, tz=timezone.utc).astimezone(act_tz) for ct in ctimes
    ]
    return np.asarray(local_time)


def ctime_to_local_hour(ctimes, tz_loc="America/Santiago"):
    # Convert the Unix timestamp to a datetime object
    local_times = ctime_to_local_time(ctimes, tz_loc)
    local_hr = [lt.hour for lt in local_times]
    return np.asarray(local_hr)


def restrict_ctimes_to_local_night(
    ctimes: list,
    morning_hour: float = 11,
    night_hour: float = 22,
    tz_loc: str = "America/Santiago",
):
    """
    Given a list of ctimes, return the boolean mask for night times,
    where night times are defined as pre-morning_hour and post-night_hour.
    """
    local_hour = ctime_to_local_hour(ctimes, tz_loc=tz_loc)
    night_times = (local_hour > night_hour) & (local_hour < morning_hour)

    return night_times


def get_freq_array_splits(files: list):
    """
    given a list of map files with names depth1_[obsid]_[arr]_[freq]_[maptype].fits,
    return a dictionary with keys arr_freq for each combination of arr,freq as
    well as freq for coadds of all arrays.
    each element contains a boolean array indexing the files into those subsets.

    check if all of the files only contain one wafer, in which case remove the
    array-coadded key; i.e. if all files have pa5_f090, remove the f090.
    """
    arrs = np.asarray([f.split("_")[-3] for f in files])
    freqs = np.asarray([f.split("_")[-2] for f in files])

    splits = {}
    for freq in np.unique(freqs):
        splits[freq] = freqs == freq
        for arr in np.unique(arrs):
            splits[f"{arr}_{freq}"] = (freqs == freq) & (arrs == arr)

    ## check if any arr_freq splits are equal to the coadded freq split
    remove_list = []
    for freq in np.unique(freqs):
        coaddsplit = splits[freq]
        for arr in np.unique(arrs):
            if np.all(splits[f"{arr}_{freq}"] == coaddsplit):
                remove_list.append(freq)
    for freq in np.unique(remove_list):
        del splits[freq]
    del remove_list
    return splits


def get_map_groups(
    maps: list,
    coadd_days: float = 7,
    restrict_to_night: bool = False,
    morning_hour: float = 11,
    night_hour: float = 22,
):
    """
    given a list of map files with names depth1_[obsid]_[arr]_[freq]_[maptype].fits
    group into bins of length `coadd_days` days.
    Also supports restricting the maps to nighttime only.

    returns
     map_groups:dict
        dictionary keyed by array/freq combinations containing the relevant set of maps
     map_group_time_range:dict
        dictionary keyed by array/freq combinations containing the actual time range of maps in the bin.
     time_bins: array-like, None
        time bin centers in ctime seconds (with bin-widths of coadd_days) if coadd_days>0 else None.
    """
    from pathlib import Path

    map_group_time_range = {}
    map_groups = {}
    time_bins = None
    times = np.asarray([float(m.split("depth1_")[-1].split("_")[0]) for m in maps])
    maps = np.asarray(maps)
    if restrict_to_night:
        night_times = restrict_ctimes_to_local_night(
            times, morning_hour=morning_hour, night_hour=night_hour
        )
        maps = np.asarray(maps)[night_times]
        times = times[night_times]
        del night_times

    mintime = np.min(times)
    maxtime = np.max(times)
    full_time_range = (maxtime - mintime) / 86400.0
    if full_time_range > 1:
        print("Full time range: %.1f days" % (full_time_range))
    elif full_time_range > 1 / 24.0:
        print("Full time range: %.1f hours" % (full_time_range * 24))
    else:
        print("Full time range: %.1f minutes" % (full_time_range * 24 * 60))
    if coadd_days != 0:
        if mintime == maxtime:
            time_bins = [mintime]
        elif full_time_range < coadd_days:
            time_bins = [np.mean([mintime, maxtime])]
        else:
            time_bins = np.arange(mintime, maxtime, coadd_days * 86400)
        time_idx = np.digitize(times, time_bins) - 1

    freq_arr_splits = get_freq_array_splits(maps)

    for key in freq_arr_splits:
        freq_arr_inds = freq_arr_splits[key]
        inmaps = maps[freq_arr_inds]
        map_groups[key] = [[m] for m in inmaps]
        map_group_time_range[key] = [[] for _ in inmaps]
        if coadd_days > 0:
            map_groups[key] = [[] for _ in range(len(time_bins))]
            freq_arr_time_inds = time_idx[freq_arr_inds]
            for i in range(len(inmaps)):
                map_groups[key][freq_arr_time_inds[i]].append(Path(inmaps[i]))
            mapgrouptimes = [
                float(str(m).split("depth1_")[-1].split("_")[0]) for m in inmaps
            ]
            if mapgrouptimes:
                map_group_time_range[key].append(
                    [min(mapgrouptimes), max(mapgrouptimes)]
                )
            else:
                map_group_time_range[key].append([])

    return map_groups, map_group_time_range, time_bins


def get_frequency(freq: str, arr: str = None):
    # Array not used yet, but could have per array/freq band centers
    ## from lat white paper https://arxiv.org/pdf/2503.00636
    frequency = {
        "f030": 30 * u.GHz,
        "f040": 40 * u.GHz,
        "f090": 90 * u.GHz,
        "f150": 150 * u.GHz,
        "f220": 220 * u.GHz,
        "f280": 280 * u.GHz,
    }

    return frequency[freq]


def get_fwhm(freq: str, arr: str = None):
    # Array not used yet, but could have per array/freq fwhm
    ## from lat white paper https://arxiv.org/pdf/2503.00636
    fwhm = {
        "f030": 7.4 * u.arcmin,
        "f040": 5.1 * u.arcmin,
        "f090": 2.2 * u.arcmin,
        "f150": 1.4 * u.arcmin,
        "f220": 1.0 * u.arcmin,
        "f280": 0.9 * u.arcmin,
    }

    return fwhm[freq]


def get_pix_from_peak_to_noise(
    peak: Union[float, list],
    noise: Union[float, list],
    fwhm_pix: Union[float, list],
    minpix: Union[int, float] = 1,
):
    """
    Given a gaussian with amplitude, `peak`, and fwhm in units of pixels, `fwhm_pix`,
    calculate the distance from the peak to the noise level given by `noise`.

    Assumes that peak, noise, fwhm_pix are either the same length or floats
    (i.e. the same noise value or same fwhm for each source)

    Units are in pixels.

    minpix provides the minimum radius.

    Returns array of pixels corresponding to mask radius for each source.
    """
    from astropy.modeling.models import Gaussian1D

    mask_pix = []
    std = np.atleast_1d(fwhm_pix) / (2 * np.sqrt(2 * np.log(2)))
    peak = np.atleast_1d(peak)
    noise = np.atleast_1d(noise)
    for i in range(len(peak)):
        ## quick check that they're
        if len(std) == 1:
            stdi = std[0]
        else:
            stdi = std[i]
        if len(noise) == 1:
            noisei = noise[0]
        else:
            noisei = noise[i]

        f = Gaussian1D(peak[i], stddev=stdi)
        pix = 0
        while f(pix) > noisei:
            pix += 1
        if pix < minpix:
            pix = minpix
        mask_pix.append(pix)
    return np.atleast_1d(mask_pix)


def get_cut_radius(
    map_res_arcmin: float,
    arr: str,
    freq: str,
    fwhm_arcmin: float = None,
    matched_filtered: bool = False,
    source_amplitudes: list = [],
    map_noise: list = [],
    min_radius_arcmin: float = 0.5,
    max_radius_arcmin: float = 60.0,
):
    """
    get desired radius that we want to cut on point sources or signal at the center

    Uses the noise in the map which can be a single value or localized for each source.

    if source_amplitudes provided, will perform a scaling of the cut radius up to max_radius
    by taking the number of pixels required for a gaussian with peak amplitude to fall below the map noise.
    Args:
        map_res_arcmin: resolution of map, in arcmin
        arr: array name
        freq: frequency
        fwhm_arcmin: float fwhm (arcmin) or None
        match_filtered: bool , increase radius by 2sqrt(2) if so
        source_amplitudes: list of amplitudes, used to set mask radius
        map_noise: noise in the map, for scaling mask radius.
        max_radius_arcmin: maximum mask radius, in arcmin
    Returns:
        radius in pixel
    """

    if not fwhm_arcmin:
        fwhm_arcmin = get_fwhm(freq, arr)

    ## I guess this is needed for filtering wings?
    mf_factor = 2**0.5 if matched_filtered else 1.0

    ## sqrt amplitude scaling after some amplitude pivot
    if len(source_amplitudes) == 0:
        radius_pix = np.rint(2 * mf_factor * fwhm_arcmin / map_res_arcmin)
    else:
        if len(map_noise) == 0:
            raise ValueError("Map noise must be given")
        radius_pix = get_pix_from_peak_to_noise(
            source_amplitudes,
            map_noise,
            fwhm_arcmin / map_res_arcmin,
            minpix=min_radius_arcmin / map_res_arcmin,
        )

    max_radius_pix = max_radius_arcmin / map_res_arcmin
    radius_pix[radius_pix > max_radius_pix] = max_radius_pix
    return radius_pix


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
    if type(colnames) is str:
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
            if type(colnames) is str:
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


def angular_separation(ra1, dec1, ra2, dec2):
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

    return sep


def fit_poss(rho, kappa, poss, rmax=8 * pixell_utils.arcmin, tol=1e-4, snmin=3):
    """Given a set of fiducial src positions [{dec,ra},nsrc],
    return a new set of positions measured from the local center-of-mass
    Assumes scalar rho and kappa"""
    from scipy import ndimage

    ref = np.max(kappa)
    if ref == 0:
        ref = 1
    snmap2 = rho**2 / np.maximum(kappa, ref * tol)
    # label regions that are strong enough and close enough to the
    # fiducial positions
    mask = snmap2 > snmin**2
    mask &= snmap2.distance_from(poss, rmax=rmax) < rmax
    labels = enmap.samewcs(ndimage.label(mask)[0], rho)
    del mask
    # Figure out which labels correspond to which objects
    label_inds = labels.at(poss, order=0)
    good = label_inds > 0
    # Compute the center-of mass position for the good labels
    # For the bad ones, just return the original values
    oposs = poss.copy()
    if np.sum(good) > 0:
        oposs[:, good] = snmap2.pix2sky(
            np.array(ndimage.center_of_mass(snmap2, labels, label_inds[good])).T
        )
    del labels
    osns = snmap2.at(oposs, order=1) ** 0.5
    # for i in range(nsrc):
    # dpos = pixell_utils.rewind(oposs[:,i]-poss[:,i])
    # print("%3d %6.2f %8.3f %8.3f %8.3f %8.3f" % (i, osns[i], poss[1,i]/pixell_utils.degree, poss[0,i]/pixell_utils.degree, dpos[1]/pixell_utils.arcmin, dpos[0]/pixell_utils.arcmin))
    return oposs, osns


def extract_flux_thumbnail_from_file(
    ra: float,
    dec: float,
    ctime: int,
    freq: str,
    arr: str = None,
    mapdir: str = "/scratch/gpfs/SIMONSOBS/so/maps/actpol/depth1",
    radius: float = 0.5,
):
    from ..maps.maps import kappa_clean

    # Parse position
    pos = np.deg2rad(np.array([dec, ra]))

    # Create pixbox around source
    subdir = str(ctime)[:5]
    imap = f"{mapdir}/{subdir}/depth1_{ctime}_{arr}_{freq}_map.fits"
    shape_full, wcs_full = enmap.read_map_geometry(imap)
    pixbox = enmap.neighborhood_pixboxes(
        shape_full, wcs_full, pos.T, radius * pixell_utils.degree
    )

    # Read in stamp maps
    stamp_rho = enmap.read_map(
        f"{mapdir}/{subdir}/depth1_{ctime}_{arr}_{freq}_rho.fits", pixbox=pixbox
    )[0]
    stamp_kappa = enmap.read_map(
        f"{mapdir}/{subdir}/depth1_{ctime}_{arr}_{freq}_kappa.fits", pixbox=pixbox
    )[0]
    stamp_kappa = kappa_clean(
        stamp_kappa,
        stamp_rho,
    )
    flux = stamp_rho / stamp_kappa

    return flux


def find_map_ctime(ctime, mapdir="/scratch/gpfs/SIMONSOBS/so/maps/actpol/depth1"):
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


def ctime_to_pos(
    ctime: float,
    freq: str,
    arr: str,
    mapdir: str = "/scratch/gpfs/SIMONSOBS/so/maps/actpol/depth1",
):
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


def calculate_alpha(data):
    alpha = math.log(data[0, 1] / data[1, 1], data[0, 0] / data[1, 0])
    A = data[0, 1] / (data[0, 0] ** alpha)
    err = (
        (data[0, 2] / (data[0, 1] * np.log(data[0, 0] / data[1, 0]))) ** 2.0
        + (data[1, 2] / (data[1, 1] * np.log(data[0, 0] / data[1, 0]))) ** 2.0
    ) ** 0.5
    return A, alpha, err


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


def euclidean_variance(ra, dec, ra_data, dec_data):
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
    ## quantile?
    c0 = stats.norm.cdf(0, loc=mu, scale=std)
    return stats.norm.ppf(p * (1 - c0) + c0, loc=mu, scale=std)
