from typing import Union

import numpy as np
from pixell.enmap import ndmap

"""
Source finding routines from spt3g_software.
Photutils version implemented by Melanie Archipley

"""


def get_source_sky_positions(extracted_sources, skymap):
    """
    from output of extract_sources, use xpeak and ypeak to
    convert to sky coordinates.
    """
    from pixell.utils import degree

    for f in extracted_sources:
        x, y = extracted_sources[f]["xpeak"], extracted_sources[f]["ypeak"]

        dec, ra = skymap.pix2sky(np.asarray([[y], [x]]))
        extracted_sources[f]["ra"] = ra[0] % (360 * degree)
        extracted_sources[f]["dec"] = dec[0]

        if "err_xpeak" in extracted_sources[f] and "err_ypeak" in extracted_sources[f]:
            dx, dy = (
                extracted_sources[f]["err_xpeak"],
                extracted_sources[f]["err_ypeak"],
            )
            ## assume flat sky
            mapres = abs(skymap.wcs.wcs.cdelt[0]) * degree
            extracted_sources[f]["err_ra"] = dx * mapres / np.cos(dec[0])
            extracted_sources[f]["err_dec"] = dy * mapres

    return extracted_sources


def get_source_observation_time(extracted_sources, timemap: np.ndarray = None):
    """
    from output of extract_sources, use xpeak and ypeak to
    get the observed time given the map `timemap`.
    """

    for f in extracted_sources:
        if not isinstance(timemap, type(None)):
            x, y = (
                int(extracted_sources[f]["xpeak"]),
                int(extracted_sources[f]["ypeak"]),
            )
            extracted_sources[f]["time"] = timemap[y, x]
        else:
            extracted_sources[f]["time"] = np.nan

    return extracted_sources


def extract_sources(
    inmap: ndmap,
    timemap: ndmap = None,
    maprms: Union[float, ndmap] = None,
    nsigma: float = 5.0,
    minrad: list = [0.5],
    sigma_thresh_for_minrad: list = [0.0],
    res: float = None,
    pixel_mask: np.ndarray = None,
    log=None,
):
    """
    Quick 'n' dirty source finding

    Arguments
    ---------
    inmap : 2d array or enmap.ndmap
        2d-array representing an unweighted flux map.
        Must be an enmap.ndmap to get sky coordinates.
    timemap : enmap.ndmap (or 2D array)
        time at each pixel, used to get observed time if provided
    maprms : enmap.ndmap or float
        The 1-sigma noise level in the map which can be postition-dependent. If None, will be calculated as rms.
    nsigma :
        Required signal-to-noise to detect a source.
    minrad : array-like
        The required separation between detected sources in arcmin. If given as
        a list, provides different radii for different-sigma sources.
    sigma_thresh_for_minrad : array-like
        The source detection strengths corresponding to different exclusion
        radii. Only used if more than one element in ``minrad``.
    res :
        Resolution of map, in arcmin.  Required if ``inmap`` is an array.
    pixel_mask : 2d array or enmap.ndmap
        Optional mask applied to map before source finding.

    Returns
    -------
    output_struct: dict
        Contains a dictionary for each source, labelled by sequential integers,
        with keys 'xpeak', 'ypeak','peakval','peaksig'.
        Also the various photutils output describing the 2D gaussian fit:
            "ellipticity", "elongation", "fwhm", semimajor_sigma, semiminor_sigma, orientation
        Also kron aperture things:
          "kron_aperture","kron_flux", "kron_fluxerr", "kron_radius"

    Notes
    -----
    Jan 2019: DPD ported from spt_analysis/sources/find_sources_quick.pro
    Jan 2025: AF porting to sotrplib
    """
    from pixell.utils import arcmin, degree

    log.bind(func_name="extract_sources")

    if res is None:
        try:
            res = np.abs(inmap.wcs.wcs.cdelt[0]) * degree / arcmin
        except AttributeError:
            log.error("extract_sources.invalid_res", res=res)
            raise ValueError("Argument `res` required if inmap is an array")

    minrad = np.atleast_1d(minrad)
    sigma_thresh_for_minrad = np.atleast_1d(sigma_thresh_for_minrad)

    if len(sigma_thresh_for_minrad) != len(minrad):
        log.error(
            "extract_sources.invalid_minrad",
            minrad=minrad,
            sigma_thresh_for_minrad=sigma_thresh_for_minrad,
        )
        raise ValueError(
            "If you are specifying multiple avoidance radii,"
            + "please supply a threshold level for each one."
        )

    if pixel_mask is not None:
        inmap *= pixel_mask

    # get rms in map if not supplied
    if isinstance(maprms, type(None)):
        whn0 = np.where(np.abs(inmap) > 1.0e-8)
        if len(whn0[0]) == 0:
            maprms = np.nanstd(inmap)
        else:
            maprms = np.nanstd(np.asarray(inmap)[whn0])
    log.bind(map_rms=maprms)
    peaks = find_using_photutils(
        np.asarray(inmap),
        maprms,
        nsigma=nsigma,
        minnum=2,
        log=log,
    )

    npeaks = peaks["n_detected"]
    log.info("extract_sources.n_detected", npeaks=npeaks)
    if npeaks == 0:
        return {}

    # gather detected peaks into output structure, ignoring repeat
    # detections of same object
    xpeaks = peaks["xcen"]
    ypeaks = peaks["ycen"]
    peakvals = peaks["maxvals"]
    peaksigs = peaks["sigvals"]
    ellipticities = peaks["ellipticity"]
    elongations = peaks["elongation"]
    fwhms = peaks["fwhm"]
    sigma_major = peaks["semimajor_sigma"]
    sigma_minor = peaks["semiminor_sigma"]
    orientation = peaks["orientation"]
    covar_sigx2 = peaks["covar_sigx2"]
    covar_sigy2 = peaks["covar_sigy2"]
    covar_sigxy = peaks["covar_sigxy"]
    kron_apertures = peaks["kron_aperture"]
    kron_fluxes = peaks["kron_flux"]
    kron_fluxerrs = peaks["kron_fluxerr"]
    kron_radii = peaks["kron_radius"]
    peak_assoc = np.zeros(npeaks)

    output_struct = dict()
    for i in np.arange(npeaks):
        output_struct[i] = {
            "xpeak": xpeaks[0],
            "ypeak": ypeaks[0],
            "peakval": peakvals[0],
            "peaksig": peaksigs[0],
            "ellipticity": ellipticities[0],
            "elongation": elongations[0],
            "fwhm": fwhms[0].value * res,
            "semimajor_sigma": sigma_major[0].value * res,
            "semiminor_sigma": sigma_minor[0].value * res,
            "covar_sigx": np.sqrt(covar_sigx2[0].value) * res,
            "covar_sigy": np.sqrt(covar_sigy2[0].value) * res,
            "covar_sigxy": np.sqrt(abs(covar_sigxy[0].value)) * res,
            "orientation": orientation[0].value * degree,
            "kron_aperture": kron_apertures[0],
            "kron_flux": kron_fluxes[0],
            "kron_fluxerr": kron_fluxerrs[0],
            "kron_radius": kron_radii[0].value * res,
        }

    minrad_pix = minrad / res
    log.info("extract_sources.output_dict_initialized", minrad_pix=minrad_pix)

    ksource = 1
    # different accounting if exclusion radius is specified as a function
    # of significance
    if len(minrad) > 1:
        minrad_pix_all = np.zeros(npeaks)
        sthresh = np.argsort(sigma_thresh_for_minrad)
        for j in np.arange(len(minrad)):
            i = sthresh[j]
            whgthresh = np.where(peaksigs >= sigma_thresh_for_minrad[i])[0]
            if len(whgthresh) > 0:
                minrad_pix_all[whgthresh] = minrad_pix[i]
        minrad_pix_os = np.zeros(npeaks)
        minrad_pix_os[0] = minrad_pix_all[0]
        for j in np.arange(npeaks):
            prev_x = np.array(
                [output_struct[n]["xpeak"] for n in np.arange(0, ksource)]
            )
            prev_y = np.array(
                [output_struct[n]["ypeak"] for n in np.arange(0, ksource)]
            )
            distpix = np.sqrt((prev_x - xpeaks[j]) ** 2 + (prev_y - ypeaks[j]) ** 2)
            whclose = np.where(distpix <= minrad_pix_os[0:ksource])[0]
            if len(whclose) == 0:
                output_struct[ksource] = {
                    "xpeak": xpeaks[j],
                    "ypeak": ypeaks[j],
                    "peakval": peakvals[j],
                    "peaksig": peaksigs[j],
                    "ellipticity": ellipticities[j],
                    "elongation": elongations[j],
                    "fwhm": fwhms[j].value * res,
                    "semimajor_sigma": sigma_major[j].value * res,
                    "semiminor_sigma": sigma_minor[j].value * res,
                    "covar_sigx": np.sqrt(covar_sigx2[j].value) * res,
                    "covar_sigy": np.sqrt(covar_sigy2[j].value) * res,
                    "covar_sigxy": np.sqrt(abs(covar_sigxy[j].value)) * res,
                    "orientation": orientation[j].value * degree,
                    "kron_aperture": kron_apertures[j],
                    "kron_flux": kron_fluxes[j],
                    "kron_fluxerr": kron_fluxerrs[j],
                    "kron_radius": kron_radii[j].value * res,
                }

                peak_assoc[j] = ksource
                minrad_pix_os[ksource] = minrad_pix_all[j]
                ksource += 1
            else:
                mindist = min(distpix)
                peak_assoc[j] = distpix.argmin()
    else:
        for j in range(npeaks):
            prev_x = np.array(
                [output_struct[n]["xpeak"] for n in np.arange(0, ksource)]
            )
            prev_y = np.array(
                [output_struct[n]["ypeak"] for n in np.arange(0, ksource)]
            )
            distpix = np.sqrt((prev_x - xpeaks[j]) ** 2 + (prev_y - ypeaks[j]) ** 2)
            mindist = min(distpix)
            if mindist > minrad_pix[0]:
                output_struct[ksource] = {
                    "xpeak": xpeaks[j],
                    "ypeak": ypeaks[j],
                    "peakval": peakvals[j],
                    "peaksig": peaksigs[j],
                    "ellipticity": ellipticities[j],
                    "elongation": elongations[j],
                    "fwhm": fwhms[j].value * res,
                    "semimajor_sigma": sigma_major[j].value * res,
                    "semiminor_sigma": sigma_minor[j].value * res,
                    "covar_sigx": np.sqrt(covar_sigx2[j].value) * res,
                    "covar_sigy": np.sqrt(covar_sigy2[j].value) * res,
                    "covar_sigxy": np.sqrt(abs(covar_sigxy[j].value)) * res,
                    "orientation": orientation[j].value * degree,
                    "kron_aperture": kron_apertures[j],
                    "kron_flux": kron_fluxes[j],
                    "kron_fluxerr": kron_fluxerrs[j],
                    "kron_radius": kron_radii[j].value * res,
                }
                peak_assoc[j] = ksource
                ksource += 1
            else:
                peak_assoc[j] = distpix.argmin()
    for src in list(output_struct.keys()):
        if src >= ksource:
            del output_struct[src]
    log.info(
        "extract_sources.output_dict_prepared",
        n_sources=len(list(output_struct.keys())),
    )
    output_struct = get_source_sky_positions(output_struct, inmap)
    output_struct = get_source_observation_time(output_struct, timemap)
    log.info("extract_sources.output_dict_finalized")
    return output_struct


def find_using_photutils(
    Tmap: ndmap,
    signoise: Union[float, ndmap] = None,
    minnum: int = 2,
    nsigma: float = 5.0,
    log=None,
):
    """
    Written to take same inputs and return outputs in the same format as the
    function ``find_groups``.  Utilizing astropy.photutils, one can deblend
    sources close to each other.  Exactly how the deblending is done uses the
    default arguments of the package, influenced by SExtractor.

    From ``find_groups``: given a 2d array (a map), will find groups of elements
    (pixels) that are spatially associated (sources).

    Arguments
    ---------
    Tmap: enamp.ndmap
        Intensity map.
    signoise : float or enmap.ndmap
        Noise in the map, can be float or enmap.ndmap of same shape as Tmap.
    offset : float
        Zero point of map.
    minnum : int
        Minimum number of pixels needed to form a group.
    nsigma : float
        Required detection threshold for a group.

    Returns
    -------
    Dictionary with following keys:
        * maxvals - array of heights (in map units) of found objects.
        * sigvals - array of heights (in significance units) of found objects.
        * xcen - array of x-location of group centers. From the documentation,
          the "centroid is computed as the center of mass of the unmasked pixels
          within the source segment."
        * ycen - array of Y-location of group centers.
        * semimajor_sigma, semiminor_sigma - sigma in major and minor axes of 2D gauss fit.
        * orientation - The angle between the x axis and the semimajor axis.
                        The angle increases in the counter-clockwise direction.
        * n_detected - Number of detected sources.

    The keys below are outputs of the astropy source finding functionality,
    being used to test if any of them indicate extendedness of a source.
        * ellipticity
        * elongation
        * fwhm - units of pixels. From the documentation, "circularized FWHM of
          2D Gaussian function with same second order moments as the source."
        * kron_aperture
        * kron_flux - Parameter requires that Tmap already be in units of mJy.
        * kron_fluxerr - Parameter requires that Tmap already be in units of mJy.
        * kron_radius - units of pixels.

    Function written by Melanie Archipley, adapted by AF Jan 2025
    """
    # this import is here instead of at the beginning because it has strange
    # dependencies that I (Melanie) do not want to cause problems for other people
    from photutils import segmentation as pseg

    log.bind(func_name="find_using_photutils")
    default_keys = {
        "maxvals": "max_value",
        "xcen": "xcentroid",
        "ycen": "ycentroid",
        "sigvals": None,
        "n_detected": None,
    }

    extra_keys = [
        "ellipticity",
        "elongation",
        "fwhm",
        "semimajor_sigma",
        "semiminor_sigma",
        "orientation",
        "covar_sigx2",
        "covar_sigy2",
        "covar_sigxy",
        "kron_aperture",
        "kron_flux",
        "kron_fluxerr",
        "kron_radius",
    ]

    groups = {k: 0 for k in list(default_keys) + extra_keys}
    log.info("find_using_photutils.start")
    ## convert ndmap to 2D numpy array for photutils
    Tmap = np.asarray(Tmap)
    assert len(Tmap.shape) == 2
    if isinstance(signoise, ndmap):
        signoise = np.asarray(signoise)
        assert signoise.shape == Tmap.shape

    img = pseg.detect_sources(Tmap, threshold=nsigma * signoise, npixels=minnum)
    if img is None:
        return groups

    img = pseg.deblend_sources(Tmap, img, npixels=minnum)
    if img is None:
        return groups

    cat = pseg.SourceCatalog(
        Tmap,
        img,
        error=signoise,
        apermask_method="correct",
        kron_params=(2.5, 1.0),
    )

    # convert catalog to table
    columns = [v for _, v in default_keys.items() if v] + extra_keys
    tbl = cat.to_table(columns=columns)

    # rename keys to match find_groups output
    for k, v in default_keys.items():
        if v is not None:
            tbl.rename_column(v, k)
    tbl.sort("maxvals", reverse=True)

    # store signal at each source location
    ix, iy = [np.floor(tbl[k] + 0.5).astype(int) for k in ("xcen", "ycen")]
    tbl["sigvals"] = Tmap[iy, ix] / signoise[iy, ix]

    # populate output dictionary
    groups["n_detected"] = len(tbl)
    for k in groups:
        if k in tbl.columns:
            groups[k] = tbl[k]
    log.info("find_using_photutils.end")
    return groups


def mask_sources_outside_map(sourcecat, maskmap: ndmap):
    """
    Determines if source is inside mask or observed region

    Args:
        sourcecat: source catalog, need keys RADeg, decDeg
        maskmap: ndmap of mask (zero masked, one not masked)
                 or weights map which will be zero or nan elsewhere.

    Returns:
        mask column for sources, 1 observed, 0 not observed or outside map
    """
    from pixell.enmap import sky2pix

    # Check if sources are in mask
    coords_rad = np.deg2rad(np.array([sourcecat["decDeg"], sourcecat["RADeg"]]))
    ypix, xpix = sky2pix(maskmap.shape, maskmap.wcs, coords_rad)
    mask = np.zeros(len(ypix))
    for i, (x, y) in enumerate(zip(xpix.astype(int), ypix.astype(int))):
        if x < 0 or y < 0:
            m = 0
        else:
            try:
                m = np.nan_to_num(maskmap[y, x])  # lookup mask value
            except IndexError:
                m = 0
        mask[i] = m
    # Convert to binary
    ## if non-observed regions are nan, ignore those as well
    mask[np.abs(mask) > 1e-8] = 1
    # switch 0 and 1
    mask = mask.astype("bool")

    return mask


def get_ps_inmap(imap: np.ndarray, sourcecat, fluxlim: bool = None):
    """
    get point sources in map

    Args:
        imap:ndmap to get point sources in
        sourcecat:source catalog
        fluxlim:flux limit in mJy

    Returns:
        sourcecat with sources in map
    """
    from pixell.enmap import sky2pix

    if fluxlim:
        sourcecat = sourcecat[sourcecat["fluxJy"] > fluxlim / 1000.0]
    # convert ra to -180 to 180
    sourcecat["RADeg"] = np.where(
        sourcecat["RADeg"] > 180, sourcecat["RADeg"] - 360, sourcecat["RADeg"]
    )

    sourcecat_coords = sky2pix(
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
