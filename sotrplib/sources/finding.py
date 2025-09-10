from typing import Union

import numpy as np
from astropy import units as u
from astropydantic import AstroPydanticQuantity
from numpydantic import NDArray
from pixell.enmap import ndmap
from structlog import get_logger
from structlog.types import FilteringBoundLogger

from sotrplib.maps.core import ProcessableMap
from sotrplib.sources.sources import MeasuredSource
from sotrplib.utils.utils import get_frequency

"""
Source finding routines from spt3g_software.
Photutils version implemented by Melanie Archipley

"""


def get_source_sky_positions(extracted_sources, skymap):
    """
    from output of extract_sources, use xpeak and ypeak to
    convert to sky coordinates.
    """

    for f in extracted_sources:
        x, y = extracted_sources[f]["xpeak"], extracted_sources[f]["ypeak"]

        dec, ra = skymap.pix2sky(np.asarray([[y], [x]]))
        ra = u.Quantity(ra[0], "rad")
        dec = u.Quantity(dec[0], "rad")
        extracted_sources[f]["ra"] = ra % (360 * u.deg)
        extracted_sources[f]["dec"] = dec

        if "err_xpeak" in extracted_sources[f] and "err_ypeak" in extracted_sources[f]:
            dx, dy = (
                extracted_sources[f]["err_xpeak"],
                extracted_sources[f]["err_ypeak"],
            )
            ## assume flat sky
            mapres = u.Quantity(abs(skymap.wcs.wcs.cdelt[0]), "deg")
            extracted_sources[f]["err_ra"] = dx * mapres / np.cos(dec)
            extracted_sources[f]["err_dec"] = dy * mapres

    return extracted_sources


def get_source_observation_time(
    extracted_sources, timemap: ProcessableMap | NDArray = None
):
    """
    from output of extract_sources, use xpeak and ypeak to
    get the observed time given the map `timemap`.
    """

    for f in extracted_sources:
        if timemap is None:
            extracted_sources[f]["time"] = np.nan
            continue

        x, y = (
            int(extracted_sources[f]["xpeak"]),
            int(extracted_sources[f]["ypeak"]),
        )
        if isinstance(timemap, ProcessableMap):
            t_start = timemap.time_first[y, x]
            t_mean = timemap.time_mean[y, x]
            t_end = timemap.time_end[y, x]
        else:
            t_mean = timemap[y, x]
            t_start = np.nan
            t_end = np.nan
        # Set the extracted source times, backwards compatibility
        extracted_sources[f]["time"] = t_mean
        extracted_sources[f]["time_start"] = t_start
        extracted_sources[f]["time_end"] = t_end

    return extracted_sources


def extract_sources(
    inmap: ProcessableMap,
    nsigma: float = 5.0,
    maprms: NDArray | float = None,
    minrad: list[AstroPydanticQuantity[u.arcmin]] = [u.Quantity(0.5, "arcmin")],
    sigma_thresh_for_minrad: list[float] = [0.0],
    pixel_mask: NDArray = None,
    log: FilteringBoundLogger | None = None,
) -> list[MeasuredSource]:
    log = log or get_logger()
    log.bind(func_name="extract_sources")
    if not inmap.finalized:
        log.error("extract_sources.map_not_finalized")
        return []

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
        inmap.flux *= pixel_mask
    else:
        pixel_mask = np.ones(inmap.flux.shape)
        pixel_mask[np.isnan(inmap.flux)] = 0.0
        inmap.flux *= pixel_mask
        inmap.snr *= pixel_mask

    # get rms in map if not supplied
    if maprms is None:
        maprms = np.asarray(inmap.flux / inmap.snr)
        maprms *= pixel_mask
        log.bind(map_rms="flux/snr map")
    else:
        maprms = maprms
        log.bind(map_rms=maprms)
    peaks = find_using_photutils(
        np.asarray(inmap.flux),
        maprms,
        nsigma=nsigma,
        minnum=2,
        log=log,
    )

    npeaks = peaks["n_detected"]
    log.info("extract_sources.n_detected", npeaks=npeaks)
    if npeaks == 0:
        return []

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
    peak_assoc = np.zeros(npeaks)

    output_struct = dict()
    for i in np.arange(npeaks):
        output_struct[i] = {
            "xpeak": xpeaks[0],
            "ypeak": ypeaks[0],
            "peakval": peakvals[0] * inmap.flux_units,
            "peaksig": peaksigs[0],
            "ellipticity": ellipticities[0],
            "elongation": elongations[0],
            "fwhm": fwhms[0].value * inmap.map_resolution,
            "semimajor_sigma": sigma_major[0].value * inmap.map_resolution,
            "semiminor_sigma": sigma_minor[0].value * inmap.map_resolution,
            "orientation": orientation[0].value * u.deg,
        }

    minrad_pix = [
        mr.to(u.rad).value / inmap.map_resolution.to(u.rad).value for mr in minrad
    ]
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
                    "peakval": peakvals[j] * inmap.flux_units,
                    "peaksig": peaksigs[j],
                    "ellipticity": ellipticities[j],
                    "elongation": elongations[j],
                    "fwhm": fwhms[j].value * inmap.map_resolution,
                    "semimajor_sigma": sigma_major[j].value * inmap.map_resolution,
                    "semiminor_sigma": sigma_minor[j].value * inmap.map_resolution,
                    "orientation": orientation[j].value * u.deg,
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
                    "peakval": peakvals[j] * inmap.flux_units,
                    "peaksig": peaksigs[j],
                    "ellipticity": ellipticities[j],
                    "elongation": elongations[j],
                    "fwhm": fwhms[j].value * inmap.map_resolution,
                    "semimajor_sigma": sigma_major[j].value * inmap.map_resolution,
                    "semiminor_sigma": sigma_minor[j].value * inmap.map_resolution,
                    "orientation": orientation[j].value * u.deg,
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
    output_struct = get_source_sky_positions(output_struct, inmap.flux)
    output_struct = get_source_observation_time(output_struct, inmap)
    output_data = convert_outstruct_to_measured_source_objects(output_struct, inmap)
    log.info("extract_sources.output_dict_finalized")
    return output_data


def convert_outstruct_to_measured_source_objects(
    output_struct: dict,
    inmap: ProcessableMap | None = None,
) -> list[MeasuredSource]:
    """
    Convert the output dictionary from extract_sources to MeasuredSource objects.
    """
    outlist = []
    for struct in output_struct.values():
        ms = MeasuredSource(**struct)
        ms.measurement_type = "blind"
        ms.frequency = get_frequency(inmap.frequency) if inmap else None
        ms.flux = struct["peakval"]
        ms.err_flux = struct["peakval"] / struct["peaksig"]
        ms.snr = struct["peaksig"]

        sigma_maj = struct["semimajor_sigma"]
        sigma_min = struct["semiminor_sigma"]
        theta = struct["orientation"].to(u.rad).value
        sigma_ra = np.sqrt(
            sigma_maj**2 * np.cos(theta) ** 2 + sigma_min**2 * np.sin(theta) ** 2
        )
        sigma_dec = np.sqrt(
            sigma_maj**2 * np.sin(theta) ** 2 + sigma_min**2 * np.cos(theta) ** 2
        )
        ms.err_ra = sigma_ra
        ms.err_dec = sigma_dec
        # convert to FWHM
        k = 2.0 * np.sqrt(2.0 * np.log(2.0))  # ≈ 2.354820045
        ms.fwhm_ra = k * sigma_ra
        ms.fwhm_dec = k * sigma_dec
        outlist.append(ms)
    return outlist


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

    ## photutils seems to return nan sometimes
    mask = ~np.isnan(tbl["xcen"]) & ~np.isnan(tbl["ycen"])
    # only compute indices for valid entries
    tbl = tbl[mask]
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
