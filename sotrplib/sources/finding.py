import numpy as np
import uuid7
from astropy import units as u
from astropydantic import AstroPydanticQuantity
from numpydantic import NDArray
from photutils import segmentation as pseg
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
    """Convert pixel peak positions in the extraction dict to sky coordinates.

    Parameters
    ----------
    extracted_sources : dict
        Output dict from source extraction; entries must have ``xpeak``/``ypeak``.
    skymap : enmap.ndmap
        Map used to convert pixel to sky coordinates via ``pix2sky``.

    Returns
    -------
    dict
        Same dict updated with ``ra``, ``dec``, and (if available)
        ``err_ra`` / ``err_dec`` entries.
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
    """Look up pixel observation times for each detected source.

    Parameters
    ----------
    extracted_sources : dict
        Output dict from source extraction; entries must have ``xpeak``/``ypeak``.
    timemap : ProcessableMap or NDArray, optional
        Time map used to look up per-pixel observation times.  If ``None``,
        ``extracted_sources`` is returned unchanged.

    Returns
    -------
    dict
        Same dict updated with ``observation_mean_time``, ``observation_start_time``,
        and ``observation_end_time`` entries.
    """
    if timemap is None:
        return extracted_sources

    for f in extracted_sources:
        x, y = (
            int(extracted_sources[f]["xpeak"]),
            int(extracted_sources[f]["ypeak"]),
        )
        if isinstance(timemap, ProcessableMap):
            t_start, t_mean, t_end = timemap.get_pixel_times((y, x))
        else:
            t_mean = timemap[y, x]
            t_start = np.nan
            t_end = np.nan
        # Set the extracted source times, backwards compatibility
        extracted_sources[f]["observation_mean_time"] = t_mean
        extracted_sources[f]["observation_start_time"] = t_start
        extracted_sources[f]["observation_end_time"] = t_end

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
    """Find sources above a significance threshold using photutils.

    Parameters
    ----------
    inmap : ProcessableMap
        Finalized flux map to search.
    nsigma : float, optional
        Detection threshold in map noise units (default 5.0).
    maprms : NDArray or float, optional
        Map noise; if ``None``, computed from ``flux / snr``.
    minrad : list of Quantity[arcmin], optional
        Minimum angular separation(s) between detected sources.
        Paired 1-to-1 with ``sigma_thresh_for_minrad``.
    sigma_thresh_for_minrad : list of float, optional
        SNR thresholds that activate each entry in ``minrad``.
    pixel_mask : NDArray, optional
        Pixel validity mask (0 = bad, 1 = good).
    log : FilteringBoundLogger, optional
        Structured logger.

    Returns
    -------
    list of MeasuredSource
    """
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
            " please supply a threshold level for each one."
        )

    # Apply mask
    if pixel_mask is not None:
        inmap.flux *= pixel_mask
    else:
        pixel_mask = np.ones(inmap.flux.shape)
        pixel_mask[np.isnan(inmap.flux)] = 0.0
        inmap.flux *= pixel_mask
        inmap.snr *= pixel_mask

    # Get rms in map if not supplied
    if maprms is None:
        maprms = np.asarray(inmap.flux / inmap.snr)
        maprms *= pixel_mask
        log.bind(map_rms="flux/snr map")
    else:
        log.bind(map_rms=maprms)

    # Find peaks
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

    # Convert minrad to pixel units
    minrad_pix = [
        mr.to(u.rad).value / inmap.map_resolution.to(u.rad).value for mr in minrad
    ]
    log.info("extract_sources.output_dict_initialized", minrad_pix=minrad_pix)

    # Assign exclusion radii per peak
    # If multiple radii are given, assign based on significance thresholds
    # (highest significance gets largest radius, start from lowest threshold)
    if len(minrad) > 1:
        minrad_pix_all = np.zeros(npeaks)
        for t, r in sorted(zip(sigma_thresh_for_minrad, minrad_pix)):
            minrad_pix_all[peaks["sigvals"] >= t] = r
    else:
        minrad_pix_all = np.full(npeaks, minrad_pix[0])

    output_struct = {}
    for j in range(npeaks):
        if output_struct:
            prev_x = np.array([src["xpeak"] for src in output_struct.values()])
            prev_y = np.array([src["ypeak"] for src in output_struct.values()])
            distpix = np.sqrt(
                (prev_x - peaks["xcen"][j]) ** 2 + (prev_y - peaks["ycen"][j]) ** 2
            )
            close_idx = np.where(distpix <= minrad_pix_all[j])[0]
            if len(close_idx) > 0:
                # npeaks sorted by snr.
                # if it's already associated with a source, skip it
                continue

        k = len(output_struct)
        output_struct[k] = {
            "xpeak": peaks["xcen"][j],
            "ypeak": peaks["ycen"][j],
            "peakval": peaks["maxvals"][j] * inmap.flux_units,
            "peaksig": peaks["sigvals"][j],
            "ellipticity": peaks["ellipticity"][j],
            "elongation": peaks["elongation"][j],
            "fwhm": peaks["fwhm"][j].value * inmap.map_resolution,
            "semimajor_sigma": peaks["semimajor_sigma"][j].value * inmap.map_resolution,
            "semiminor_sigma": peaks["semiminor_sigma"][j].value * inmap.map_resolution,
            "orientation": peaks["orientation"][j].value * u.deg,
            "measurement_id": str(uuid7.create()),
        }

    log.info(
        "extract_sources.output_dict_prepared",
        n_sources=len(list(output_struct.keys())),
    )

    # Post-processing
    output_struct = get_source_sky_positions(output_struct, inmap.flux)
    output_struct = get_source_observation_time(output_struct, inmap)
    output_data = convert_outstruct_to_measured_source_objects(output_struct, inmap)
    log.info("extract_sources.output_dict_finalized")

    return output_data


def convert_outstruct_to_measured_source_objects(
    output_struct: dict,
    inmap: ProcessableMap | None = None,
) -> list[MeasuredSource]:
    """Convert the internal extraction dict to ``MeasuredSource`` objects.

    Parameters
    ----------
    output_struct : dict
        Per-source dict produced by ``extract_sources``.
    inmap : ProcessableMap, optional
        Map used to populate instrument/array/frequency metadata.

    Returns
    -------
    list of MeasuredSource
    """
    outlist = []
    for struct in output_struct.values():
        ms = MeasuredSource(**struct)
        ms.measurement_type = "blind"
        ms.frequency = get_frequency(inmap.frequency) if inmap else None
        ms.instrument = inmap.instrument if inmap else None
        ms.array = inmap.array if inmap else None
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
    signoise: float | ndmap = None,
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

    Parameters
    ----------
    Tmap : enmap.ndmap
        Intensity map.
    signoise : float or enmap.ndmap, optional
        Noise in the map; scalar or same-shape ndmap.
    minnum : int, optional
        Minimum number of pixels needed to form a group (default 2).
    nsigma : float, optional
        Detection threshold in noise units (default 5.0).
    log : optional
        Structured logger.

    Returns
    -------
    dict
        Keys: ``maxvals``, ``sigvals``, ``xcen``, ``ycen``,
        ``semimajor_sigma``, ``semiminor_sigma``, ``orientation``,
        ``n_detected``, ``ellipticity``, ``elongation``, ``fwhm``,
        ``kron_aperture``, ``kron_flux``, ``kron_fluxerr``, ``kron_radius``.
        All positional keys use pixel coordinates.
        ``n_detected`` is 0 if no sources are found.

    Notes
    -----
    Written by Melanie Archipley, adapted by AF Jan 2025.
    """
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
    """Return a boolean mask indicating which catalog sources lie inside the map.

    Parameters
    ----------
    sourcecat : dict-like
        Source catalog with keys ``RADeg`` and ``decDeg``.
    maskmap : enmap.ndmap
        Mask or weights map (zero / NaN = unobserved; non-zero = observed).

    Returns
    -------
    np.ndarray of bool
        ``True`` for sources inside the observed region, ``False`` otherwise.
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
    """Return the subset of a source catalog whose positions fall inside a map.

    Parameters
    ----------
    imap : enmap.ndmap
        Map that defines the sky footprint.
    sourcecat : table-like
        Source catalog with ``RADeg`` and ``decDeg`` columns.
    fluxlim : float, optional
        Minimum flux in mJy; sources below this value are excluded first.

    Returns
    -------
    table-like
        Filtered source catalog containing only sources inside the map.
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
