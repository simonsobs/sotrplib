import warnings
from typing import Union

import numpy as np
from astropy import units as u
from astropydantic import AstroPydanticQuantity, AstroPydanticUnit
from numpy.typing import ArrayLike
from pixell import enmap, reproject
from pydantic import BaseModel
from scipy.optimize import OptimizeWarning, curve_fit
from structlog import get_logger
from structlog.types import FilteringBoundLogger
from tqdm import tqdm

from sotrplib.maps.core import ProcessableMap
from sotrplib.sources.sources import ForcedPhotometrySource, RegisteredSource


class GaussianFitParameters(BaseModel):
    amplitude: AstroPydanticQuantity[u.mJy] | None = None
    amplitude_err: AstroPydanticQuantity[u.mJy] | None = None
    ra_offset: AstroPydanticQuantity[u.deg] | None = None
    ra_err: AstroPydanticQuantity[u.deg] | None = None
    dec_offset: AstroPydanticQuantity[u.deg] | None = None
    dec_err: AstroPydanticQuantity[u.deg] | None = None
    fwhm_ra: AstroPydanticQuantity[u.deg] | None = None
    fwhm_ra_err: AstroPydanticQuantity[u.deg] | None = None
    fwhm_dec: AstroPydanticQuantity[u.deg] | None = None
    fwhm_dec_err: AstroPydanticQuantity[u.deg] | None = None
    theta: AstroPydanticQuantity[u.deg] | None = None
    theta_err: AstroPydanticQuantity[u.deg] | None = None
    forced_center: bool | None = None
    failure_reasons: list[str] | None = None


def gaussian_2d(
    xy: tuple[ArrayLike, ArrayLike],
    amplitude: float,
    x0: float,
    y0: float,
    sigma_x: float,
    sigma_y: float,
    theta: float,
    offset: float,
):
    """
    2D Gaussian function.
    """
    x, y = xy
    x0 = float(x0)
    y0 = float(y0)
    a = (np.cos(theta) ** 2) / (2 * sigma_x**2) + (np.sin(theta) ** 2) / (
        2 * sigma_y**2
    )
    b = -(np.sin(2 * theta)) / (4 * sigma_x**2) + (np.sin(2 * theta)) / (4 * sigma_y**2)
    c = (np.sin(theta) ** 2) / (2 * sigma_x**2) + (np.cos(theta) ** 2) / (
        2 * sigma_y**2
    )
    return offset + amplitude * np.exp(
        -(a * (x - x0) ** 2 + 2 * b * (x - x0) * (y - y0) + c * (y - y0) ** 2)
    )


def curve_fit_2d_gaussian(
    flux_thumb: enmap.ndmap,
    map_resolution: AstroPydanticQuantity[u.arcmin],
    flux_map_units: AstroPydanticUnit,
    snr_thumb: enmap.ndmap | None = None,
    fwhm_guess: AstroPydanticQuantity[u.arcmin] = u.Quantity(2.2, "arcmin"),
    thumbnail_center: tuple[
        AstroPydanticQuantity[u.deg], AstroPydanticQuantity[u.deg]
    ] = (u.Quantity(0.0, "deg"), u.Quantity(0.0, "deg")),
    force_center: bool = False,
    log: FilteringBoundLogger | None = None,
):
    """
    Fits a 2D Gaussian to the given data array and calculates uncertainties.

    input_map: ProcessableMap
        Must be finalized; i.e. have a flux and snr.
        Should just be a thumbnail map, though not necessarily centered on the source as
        you can set thumbnail center coordinates.

    thumbnail_center: tuple
        The ra,dec center of the thumbnail in decimal degrees. A reprojected map has center 0,0 so
        the fits give the ra,dec offsets. If using a simple cut (flux_map[-x:x,-y:y]) the center is
        at thumbnail_center so we will subtract that off to get the ra,dec offsets.

    force_center: bool
        if True, center of gaussian fit is centered at the expected location and not allowed to vary.

    """
    log = log or get_logger()
    log.info("curve_fit_2d_gaussian.start_fit")
    warnings.simplefilter("ignore", category=OptimizeWarning)

    ny, nx = flux_thumb.shape
    x = np.arange(nx)
    y = np.arange(ny)
    X, Y = np.meshgrid(x, y)
    xy = (X.ravel(), Y.ravel())

    # Initial guess for parameters
    amplitude_guess = flux_thumb.max()
    x0_guess, y0_guess = nx / 2, ny / 2  # Assume center of the image
    sigma_guess = fwhm_guess / map_resolution / 2.355  # Rough width estimate
    theta_guess = 0
    offset_guess = 0
    initial_guess = [
        amplitude_guess,
        x0_guess,
        y0_guess,
        sigma_guess,
        sigma_guess,
        theta_guess,
        offset_guess,
    ]

    ## if force center, fix the x0,y0 guesses by not varying them
    if force_center:

        def gaussian_2d_model(xy, amplitude, sigma_x, sigma_y, theta, offset):
            return gaussian_2d(
                xy, amplitude, x0_guess, y0_guess, sigma_x, sigma_y, theta, offset
            )

        initial_guess = [
            amplitude_guess,
            sigma_guess,
            sigma_guess,
            theta_guess,
            offset_guess,
        ]
        log.debug("curve_fit_2d_gaussian.force_center", initial_guess=initial_guess)
    else:
        log.debug("curve_fit_2d_gaussian.free_center", initial_guess=initial_guess)
        gaussian_2d_model = gaussian_2d

    # Fit the data
    try:
        popt, pcov = curve_fit(
            gaussian_2d_model,
            xy,
            flux_thumb.ravel(),
            # sigma=1./snr_thumb.ravel(), ## can be used? never done it before.
            p0=initial_guess,
            # absolute_sigma=False
        )
    except RuntimeError:
        log.error("curve_fit_2d_gaussian.curve_fit.failed")
        return GaussianFitParameters()

    log.debug("curve_fit_2d_gaussian.curve_fit.success", popt=popt)
    perr = np.sqrt(np.diag(pcov))  # Parameter uncertainties
    # Extract parameters and uncertainties
    if force_center:
        amplitude, sigma_x, sigma_y, theta, offset = popt
        amplitude_err, sigma_x_err, sigma_y_err, theta_err, offset_err = perr
        ra_fit, dec_fit = None, None
        ra_offset, dec_offset = None, None
        ra_err, dec_err = None, None
    else:
        amplitude, x0, y0, sigma_x, sigma_y, theta, offset = popt
        (
            amplitude_err,
            x0_err,
            y0_err,
            sigma_x_err,
            sigma_y_err,
            theta_err,
            offset_err,
        ) = perr
        dec_fit, ra_fit = flux_thumb.pix2sky([y0, x0]) * u.rad
        dec_offset = dec_fit - thumbnail_center[1]
        ra_offset = ra_fit - thumbnail_center[0]
        if ra_offset > 180.0 * u.deg:
            ra_offset -= 360.0 * u.deg
        elif ra_offset < -180.0 * u.deg:
            ra_offset += 360.0 * u.deg
        ra_err = x0_err * map_resolution
        dec_err = y0_err * map_resolution

    fwhm_x = 2.355 * sigma_x * map_resolution
    fwhm_y = 2.355 * sigma_y * map_resolution
    fwhm_x_err = 2.355 * sigma_x_err * map_resolution
    fwhm_y_err = 2.355 * sigma_y_err * map_resolution
    ## effectively set offset to zero
    amplitude = AstroPydanticQuantity(amplitude + offset, flux_map_units)
    amplitude_err = AstroPydanticQuantity(
        np.sqrt(amplitude_err**2 + offset_err**2), flux_map_units
    )
    log.debug(
        "curve_fit_2d_gaussian.unitized_parameters",
        amplitude=amplitude,
        amplitude_err=amplitude_err,
        fwhm_x=fwhm_x,
        fwhm_x_err=fwhm_x_err,
        fwhm_y=fwhm_y,
        fwhm_y_err=fwhm_y_err,
        ra_offset=ra_offset,
        ra_err=ra_err,
        dec_offset=dec_offset,
        dec_err=dec_err,
        theta=theta,
        theta_err=theta_err,
    )

    gauss_fit = GaussianFitParameters(
        amplitude=amplitude,
        amplitude_err=amplitude_err,
        ra_offset=ra_offset,
        ra_err=ra_err,
        dec_offset=dec_offset,
        dec_err=dec_err,
        fwhm_ra=fwhm_x,
        fwhm_dec=fwhm_y,
        fwhm_ra_err=fwhm_x_err,
        fwhm_dec_err=fwhm_y_err,
        theta=theta * u.rad,
        theta_err=theta_err * u.rad,
    )

    return gauss_fit


def scipy_2d_gaussian_fit(
    input_map: ProcessableMap,
    source_catalog: list[RegisteredSource],
    flux_lim_fit_centroid: AstroPydanticQuantity[u.Jy] = u.Quantity(0.3, "Jy"),
    thumbnail_half_width: AstroPydanticQuantity[u.deg] = u.Quantity(0.25, "deg"),
    fwhm: AstroPydanticQuantity[u.arcmin] = u.Quantity(2.2, "arcmin"),
    reproject_thumb: bool = False,
    log: FilteringBoundLogger | None = None,
) -> list[ForcedPhotometrySource]:
    """ """

    log = log.bind(func_name="scipy_2d_gaussian_fit")
    preamble = "sources.fitting.scipy_2d_gaussian_fit."
    fit_sources = []
    for i in tqdm(
        range(len(source_catalog)),
        desc="Cutting thumbnails and fitting sources w 2D Gaussian",
    ):
        source = source_catalog[i]
        source_name = source.source_id
        map_res = input_map.flux.map_resolution
        pix = input_map.flux.sky2pix(
            [source.dec.to(u.rad).value, source.ra.to(u.rad).value]
        )
        size_pix = thumbnail_half_width.to(u.deg).value / map_res.to(u.deg).value
        fit = GaussianFitParameters()

        if (
            pix[0] + size_pix > input_map.flux.shape[0]
            or pix[1] + size_pix > input_map.flux.shape[1]
            or pix[0] - size_pix < 0
            or pix[1] - size_pix < 0
        ):
            fit.failure_reasons = ["source_near_map_edge"]

            log.warning(f"{preamble}source_near_map_edge", source=source_name)

            ## add source information and fit here
            ## somethign like this
            fit_sources.append(
                ForcedPhotometrySource(
                    ra=source.ra,
                    dec=source.dec,
                    fit_method="2d_gaussian",
                    fit_params=fit.to_dict(),
                )
            )

            continue

        if reproject_thumb:
            thumbnail_center = (0 * u.deg, 0 * u.deg)
            try:
                flux_thumb = reproject.thumbnails(
                    input_map.flux,
                    [source.dec.to(u.rad).value, source.ra.to(u.rad).value],
                    r=thumbnail_half_width.to(u.rad).value,
                    res=map_res.to(u.rad).value,
                )
            except Exception as e:
                log.warning(f"{preamble}reproject_failed", source=source_name, error=e)
                fit.failure_reasons = ["reproject_failed"]
                fit_sources.append(
                    ForcedPhotometrySource(
                        ra=source.ra,
                        dec=source.dec,
                        fit_method="2d_gaussian",
                        fit_params=fit.to_dict(),
                    )
                )
                continue
        else:
            thumbnail_center = (source.ra, source.dec)
            ra = source.ra.to(u.rad).value + map_res.to(u.rad).value / 2
            dec = source.dec.to(u.rad).value + map_res.to(u.rad).value / 2
            radius = thumbnail_half_width.to(u.rad).value
            flux_thumb = input_map.flux.submap(
                [[dec - radius, ra - radius], [dec + radius, ra + radius]],
            )
            flux_thumb = flux_thumb.upgrade(factor=2)

        if np.any(np.isnan(flux_thumb)):
            log.warning(f"{preamble}flux_thumb_has_nan", source=source_name)
            fit.failure_reasons = ["flux_thumb_has_nan"]
            fit_sources.append(
                ForcedPhotometrySource(
                    ra=source.ra,
                    dec=source.dec,
                    fit_method="2d_gaussian",
                    fit_params=fit.to_dict(),
                )
            )
            continue

        fit = curve_fit_2d_gaussian(
            flux_thumb,
            map_resolution=abs(flux_thumb.wcs.wcs.cdelt[0]) * u.deg,
            flux_map_units=input_map.flux.map_units,
            snr_thumb=None,
            fwhm_guess=fwhm,
            thumbnail_center=thumbnail_center,
            force_center=True if source.flux < flux_lim_fit_centroid else False,
            log=log,
        )
        log.debug(f"{preamble}gauss_fit", source=source_name, fit=fit)
        fit_sources.append(fit)
    n_successful = 0
    for s in fit_sources:
        if s.failure_reasons is None:
            n_successful += 1
    log.info(f"{preamble}fits_complete", n_sources=len(fit_sources), n_fit=n_successful)
    return fit_sources


def fit_2d_gaussian(
    flux_map: enmap.ndmap,
    noise_map: enmap.ndmap,
    resolution_arcmin: float,
    fwhm_guess_arcmin: float = 2.2,
    thumbnail_center: tuple = (0, 0),
    force_center: bool = False,
    PLOT: bool = False,
):
    """
    Fits a 2D Gaussian to the given data array and calculates uncertainties.

    noise_map : enmap
        can be map of pixel noise,but matched filtered maps seem to have correlated noise proportional to signal.
        So this isn't used right now.

    thumbnail_center: tuple
        The ra,dec center of the thumbnail in decimal degrees. A reprojected map has center 0,0 so
        the fits give the ra,dec offsets. If using a simple cut (flux_map[-x:x,-y:y]) the center is
        at thumbnail_center so we will subtract that off to get the ra,dec offsets.

    force_center: bool
        if True, center of gaussian fit is centered at the expected location and not allowed to vary.

    """
    import warnings

    from pixell.utils import arcmin, degree
    from scipy.optimize import OptimizeWarning, curve_fit

    warnings.simplefilter("ignore", category=OptimizeWarning)

    ny, nx = flux_map.shape
    x = np.arange(nx)
    y = np.arange(ny)
    X, Y = np.meshgrid(x, y)
    xy = (X.ravel(), Y.ravel())

    # Initial guess for parameters
    amplitude_guess = flux_map.max()
    x0_guess, y0_guess = nx / 2, ny / 2  # Assume center of the image
    sigma_guess = fwhm_guess_arcmin / resolution_arcmin / 2.355  # Rough width estimate
    theta_guess = 0
    offset_guess = 0
    initial_guess = [
        amplitude_guess,
        x0_guess,
        y0_guess,
        sigma_guess,
        sigma_guess,
        theta_guess,
        offset_guess,
    ]

    ## if force center, fix the x0,y0 guesses by not varying them
    if force_center:

        def gaussian_2d_model(xy, amplitude, sigma_x, sigma_y, theta, offset):
            return gaussian_2d(
                xy, amplitude, x0_guess, y0_guess, sigma_x, sigma_y, theta, offset
            )

        initial_guess = [
            amplitude_guess,
            sigma_guess,
            sigma_guess,
            theta_guess,
            offset_guess,
        ]
    else:
        gaussian_2d_model = gaussian_2d

    # Fit the data with uncertainty weighting
    try:
        popt, pcov = curve_fit(
            gaussian_2d_model,
            xy,
            flux_map.ravel(),
            # sigma=noise_map.ravel(),
            p0=initial_guess,
            # absolute_sigma=True
        )
    except RuntimeError:
        return {}

    perr = np.sqrt(np.diag(pcov))  # Parameter uncertainties

    # Extract parameters and uncertainties
    if force_center:
        amplitude, sigma_x, sigma_y, theta, offset = popt
        amplitude_err, sigma_x_err, sigma_y_err, theta_err, offset_err = perr
        ra_fit, dec_fit = None, None
        ra_offset_arcmin, dec_offset_arcmin = None, None
        ra_err_arcmin, dec_err_arcmin = None, None
    else:
        amplitude, x0, y0, sigma_x, sigma_y, theta, offset = popt
        (
            amplitude_err,
            x0_err,
            y0_err,
            sigma_x_err,
            sigma_y_err,
            theta_err,
            offset_err,
        ) = perr
        dec_fit, ra_fit = flux_map.pix2sky([y0, x0]) / degree
        dec_offset_arcmin = (dec_fit - thumbnail_center[1]) * degree / arcmin
        ra_offset = ra_fit - thumbnail_center[0]
        if ra_offset > 180.0:
            ra_offset -= 360.0
        elif ra_offset < -180.0:
            ra_offset += 360.0
        ra_offset_arcmin = ra_offset * degree / arcmin
        ra_err_arcmin = x0_err * resolution_arcmin
        dec_err_arcmin = y0_err * resolution_arcmin

    fwhm_x_arcmin = 2.355 * sigma_x * resolution_arcmin
    fwhm_y_arcmin = 2.355 * sigma_y * resolution_arcmin

    fwhm_x_err_arcmin = 2.355 * sigma_x_err * resolution_arcmin
    fwhm_y_err_arcmin = 2.355 * sigma_y_err * resolution_arcmin
    theta_err_deg = theta_err / degree

    ## effectively set offset to zero
    amplitude = amplitude + offset
    amplitude_err = np.sqrt(amplitude_err**2 + offset_err**2)

    gauss_fit = {
        "amplitude": (amplitude, amplitude_err),
        "ra_offset_arcmin": (ra_offset_arcmin, ra_err_arcmin),
        "dec_offset_arcmin": (dec_offset_arcmin, dec_err_arcmin),
        "fwhm_x_arcmin": (fwhm_x_arcmin, fwhm_x_err_arcmin),
        "fwhm_y_arcmin": (fwhm_y_arcmin, fwhm_y_err_arcmin),
        "theta": (np.degrees(theta), theta_err_deg),
        "forced": force_center,
    }

    if PLOT:
        from matplotlib import pylab as plt

        # Generate fitted Gaussian data
        fitted_data = gaussian_2d_model((X, Y), *popt).reshape(ny, nx)

        # Compute contour levels using best-fit sigma_x, sigma_y, and theta
        sigma_levels = np.array([1, 2, 3])
        levels = offset + amplitude * np.exp(-0.5 * (sigma_levels**2))
        if not np.all(np.diff(levels) > 0):
            levels = levels[::-1]
        # Plot the data with contours
        plt.figure(figsize=(6, 6))
        plt.imshow(
            flux_map,
            origin="lower",
            cmap="viridis",
            extent=[
                -nx / 2 * resolution_arcmin,
                nx / 2 * resolution_arcmin,
                -ny / 2 * resolution_arcmin,
                ny / 2 * resolution_arcmin,
            ],
        )
        plt.colorbar(label="Intensity")

        if np.all(np.diff(levels) > 0):
            plt.contour(
                (X - nx / 2 + 0.5) * resolution_arcmin,
                (Y - ny / 2 + 0.5) * resolution_arcmin,
                fitted_data,
                levels=levels,
                colors="white",
                linewidths=1.5,
            )

        plt.xlabel(r"$\Delta$RA (arcmin)")
        plt.ylabel(r"$\Delta$Dec (arcmin)")
        plt.title("2D Gaussian Fit with Contours")
        plt.show()
        print(gauss_fit)

    return gauss_fit


def convert_catalog_to_source_objects(
    catalog_sources: dict,
    freq: str = "none",
    arr: str = "none",
    ctime: Union[float, enmap.ndmap] = None,
    map_id: str = "",
    source_type: str = "",
    log=None,
):
    """
    take each source in the catalog and convert to a list
    of Source objects.

    """
    from pixell.utils import arcmin, degree

    from ..sources.sources import SourceCandidate

    log = log.bind(func_name="convert_catalog_to_source_objects")

    if isinstance(catalog_sources, list):
        log.info(
            "convert_catalog_to_source_objects.catalog_is_list",
            num_sources=len(catalog_sources),
        )
        return catalog_sources
    if not catalog_sources:
        log.warning("convert_catalog_to_source_objects.catalog_is_empty", num_sources=0)
        return []

    known_sources = []
    for i in range(len(catalog_sources["name"])):
        if "gauss_fit_flag" in catalog_sources:
            bad_fit = catalog_sources["gauss_fit_flag"][i]
        else:
            bad_fit = True

        fit_type = None
        if "forced" in catalog_sources:
            if catalog_sources["forced"][i]:
                fit_type = "forced"
            else:
                fit_type = "pointing"
        else:
            fit_type = "none"

        source_ra = catalog_sources["RADeg"][i] % 360
        source_dec = catalog_sources["decDeg"][i]
        if isinstance(ctime, enmap.ndmap):
            x, y = ctime.sky2pix([source_dec * degree, source_ra * degree])
            source_ctime = ctime[int(x), int(y)]
        elif isinstance(ctime, float):
            source_ctime = ctime
        else:
            source_ctime = np.nan
        ra_uncert = np.nan
        if bad_fit:
            cs = SourceCandidate(
                ra=source_ra,
                dec=source_dec,
                err_ra=np.nan,
                err_dec=np.nan,
                flux=catalog_sources["fluxJy"][i],
                err_flux=catalog_sources["err_fluxJy"][i],
                snr=np.nan,
                freq=freq,
                arr=arr,
                ctime=source_ctime,
                map_id=map_id,
                sourceID=catalog_sources["name"][i],
                crossmatch_names=[catalog_sources["name"][i]],
                crossmatch_probabilities=[1.0],
                catalog_crossmatch=True,
                fit_type=fit_type,
                source_type=source_type,
            )
        else:
            if catalog_sources["ra_offset_arcmin"][i]:
                source_ra += catalog_sources["ra_offset_arcmin"][i] * arcmin / degree
                ra_uncert = catalog_sources["err_ra_offset_arcmin"][i] * arcmin / degree

            dec_uncert = np.nan
            if catalog_sources["dec_offset_arcmin"][i]:
                source_dec += catalog_sources["dec_offset_arcmin"][i] * arcmin / degree
                dec_uncert = (
                    catalog_sources["err_dec_offset_arcmin"][i] * arcmin / degree
                )
            cs = SourceCandidate(
                ra=source_ra,
                dec=source_dec,
                err_ra=ra_uncert,
                err_dec=dec_uncert,
                flux=catalog_sources["fluxJy"][i],
                err_flux=catalog_sources["err_fluxJy"][i],
                snr=np.nan,
                freq=freq,
                arr=arr,
                ctime=source_ctime,
                map_id=map_id,
                sourceID=catalog_sources["name"][i],
                crossmatch_names=[catalog_sources["name"][i]],
                crossmatch_probabilities=[1.0],
                catalog_crossmatch=True,
                fwhm_a=catalog_sources["fwhm_x_arcmin"][i],
                fwhm_b=catalog_sources["fwhm_y_arcmin"][i],
                err_fwhm_a=catalog_sources["err_fwhm_x_arcmin"][i],
                err_fwhm_b=catalog_sources["err_fwhm_y_arcmin"][i],
                orientation=catalog_sources["theta"][i],
                fit_type=fit_type,
                source_type=source_type,
            )

            if cs.err_flux > 0.0:
                cs.snr = cs.flux / cs.err_flux
        known_sources.append(cs)
    log.info(
        "convert_catalog_to_source_objects.catalog_converted",
        num_sources=len(known_sources),
    )
    return known_sources


def return_initial_catalog_entry(
    source,
    add_dict: dict = {},
    keyconv={},
):
    """
    source is a SourceCandidate object
    """
    outdict = {}
    for item in vars(source):
        outdict[item] = getattr(source, item)
    outdict.update(add_dict)
    for key in keyconv:
        if key in outdict:
            outdict[keyconv[key]] = outdict.pop(key)

    return outdict


def photutils_2D_gauss_fit(
    flux_map,
    snr_map,
    source_catalog: list,
    rakey: str = "RADeg",
    deckey: str = "decDeg",
    fluxkey: str = "fluxJy",
    dfluxkey: str = "err_fluxJy",
    flux_lim_fit_centroid: float = 0.3,
    size_deg=0.1,
    fwhm_arcmin=2.2,
    PLOT: bool = False,
    reproject_thumb: bool = False,
    return_thumbnails: bool = False,
    debug=False,
    log=None,
):
    """ """
    from pixell import reproject
    from pixell.utils import arcmin, degree
    from tqdm import tqdm

    from ..source_catalog.source_catalog import convert_gauss_fit_to_source_cat

    log = log.bind(func_name="photutils_2D_gauss_fit")
    preamble = "sources.fitting.photutils2DGauss."
    fits = []
    thumbnails = []
    ## these are the keys output by the fitting function
    ## but including the failed fit flag
    failed_fit_dict = {
        "ra_offset_arcmin": (None, None),
        "dec_offset_arcmin": (None, None),
        "fwhm_x_arcmin": (None, None),
        "fwhm_y_arcmin": (None, None),
        "theta": (None, None),
        "pix": (None, None),
        "forced": None,
        "gauss_fit_flag": 1,
    }

    for i in tqdm(range(len(source_catalog)), desc="Fitting sources w 2D Gaussian"):
        sc = source_catalog[i]
        source_name = sc.sourceID
        map_res = abs(flux_map.wcs.wcs.cdelt[0] * degree)
        pix = flux_map.sky2pix([sc.dec * degree, sc.ra * degree])
        failed_fit_dict["pix"] = pix
        thumbsize = size_deg * degree / map_res
        ## set default to failed dictionary
        fit_dict = return_initial_catalog_entry(
            sc,
            add_dict=failed_fit_dict,
            keyconv={
                "ra": rakey,
                "dec": deckey,
                "flux": fluxkey,
                "err_flux": dfluxkey,
                "name": "SourceID",
            },
        )
        if (
            pix[0] + thumbsize > flux_map.shape[0]
            or pix[1] + thumbsize > flux_map.shape[1]
            or pix[0] - thumbsize < 0
            or pix[1] - thumbsize < 0
        ):
            fits.append(fit_dict)

            log.warning(f"{preamble}source_near_map_edge", source=source_name)

            if return_thumbnails:
                thumbnails.append([])

            continue
        if reproject_thumb:
            thumbnail_center = (0, 0)
            try:
                flux_thumb = reproject.thumbnails(
                    flux_map,
                    [sc.dec * degree, sc.ra * degree],
                    r=size_deg * degree,
                    res=map_res,
                )
                snr_thumb = reproject.thumbnails(
                    snr_map,
                    [sc.dec * degree, sc.ra * degree],
                    r=size_deg * degree,
                    res=map_res,
                )
                with np.errstate(divide="ignore"):
                    noise_thumb = flux_thumb / snr_thumb
            except Exception as e:
                log.warning(f"{preamble}reproject_failed", source=source_name, error=e)
                fits.append(fit_dict)
                if return_thumbnails:
                    thumbnails.append([])
                continue
        else:
            thumbnail_center = (sc.ra, sc.dec)
            ra = sc.ra * degree + map_res / 2
            dec = sc.dec * degree + map_res / 2
            radius = size_deg * degree
            flux_thumb = flux_map.submap(
                [[dec - radius, ra - radius], [dec + radius, ra + radius]],
            )
            snr_thumb = snr_map.submap(
                [[dec - radius, ra - radius], [dec + radius, ra + radius]],
            )
            noise_thumb = flux_thumb / snr_thumb
            flux_thumb = flux_thumb.upgrade(factor=2)
            noise_thumb = noise_thumb.upgrade(factor=2)

        if np.any(np.isnan(flux_thumb)):
            log.warning(f"{preamble}flux_thumb_has_nan", source=source_name)
            fits.append(fit_dict)
            if return_thumbnails:
                thumbnails.append([flux_thumb, noise_thumb])
            continue

        map_res = abs(flux_thumb.wcs.wcs.cdelt[0] * degree)
        gauss_fit = fit_2d_gaussian(
            flux_thumb,
            noise_thumb,
            map_res / arcmin,
            fwhm_guess_arcmin=fwhm_arcmin,
            thumbnail_center=thumbnail_center,
            force_center=True if sc.flux < flux_lim_fit_centroid else False,
            PLOT=PLOT,
        )
        if debug:
            log.debug(f"{preamble}gauss_fit", source=source_name, fit=gauss_fit)
        if gauss_fit:
            gauss_fit[rakey] = sc.ra
            gauss_fit[deckey] = sc.dec
            gauss_fit["pix"] = pix
            gauss_fit[fluxkey], gauss_fit["err_%s" % fluxkey] = gauss_fit.pop(
                "amplitude"
            )
            for key in vars(sc):
                if key not in gauss_fit:
                    gauss_fit[key] = getattr(sc, key)
            gauss_fit["gauss_fit_flag"] = 0
            fits.append(gauss_fit)
        else:
            log.warning(f"{preamble}gauss_fit_failed", source=source_name)
            fits.append(fit_dict)
        if return_thumbnails:
            thumbnails.append([flux_thumb, noise_thumb])
    fits = convert_gauss_fit_to_source_cat(fits, log=log)
    log.info(f"{preamble}fit_complete", source=source_name)
    return fits, thumbnails


def fit_xy_slices(
    image: enmap.ndmap, noise: enmap.ndmap, res_arcmin: float, PLOT: bool = False
):
    """
    Calculate source centroid (or use center if centroid wonky)
    Slice an image and noise map through centroid/center in x and y .
    Fit 1D gaussian to the slices to extract:
        x,y offset and uncertainty
        x,y fwhm and uncertainty

    uses astropy modeling and levmarlsq fitter.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from astropy.modeling import fitting, models
    from photutils.centroids import centroid_2dg

    model_gauss = models.Gaussian1D()
    fitter_gauss = fitting.LevMarLSQFitter()

    # Get the middle indices
    ny, nx = image.shape
    centroid_x, centroid_y = centroid_2dg(image, error=noise)
    mid_y, mid_x = ny // 2, nx // 2
    if centroid_x > nx or centroid_x < 0 or centroid_y > ny or centroid_y < 0:
        print("Centroiding failed, defaulting to map center")
        centroid_x = mid_x
        centroid_y = mid_y
    # Define extent so center of image is (0,0)
    extent = np.asarray([-mid_x, mid_x, -mid_y, mid_y]) * res_arcmin
    x_arcmin = np.linspace(-mid_x, mid_x, nx) * res_arcmin
    y_arcmin = np.linspace(-mid_y, mid_y, ny) * res_arcmin

    best_fit_gaussx = fitter_gauss(
        model_gauss,
        x_arcmin,
        image[int(centroid_y), :],
        weights=1 / noise[int(centroid_y), :],
    )

    cov_diagx = np.diag(fitter_gauss.fit_info["param_cov"])

    x_off = best_fit_gaussx.mean.value
    x_off_err = np.copy(np.sqrt(cov_diagx[1]))
    x_fwhm = 2.355 * np.copy(best_fit_gaussx.stddev.value)
    x_fwhm_err = 2.355 * np.sqrt(cov_diagx[2])

    best_fit_gaussy = fitter_gauss(
        model_gauss,
        y_arcmin,
        image[:, int(centroid_x)],
        weights=1 / noise[:, int(centroid_x)],
    )

    cov_diagy = np.diag(fitter_gauss.fit_info["param_cov"])

    y_off = best_fit_gaussy.mean.value
    y_off_err = np.copy(np.sqrt(cov_diagy[1]))
    y_fwhm = 2.355 * np.copy(best_fit_gaussy.stddev.value)
    y_fwhm_err = 2.355 * np.sqrt(cov_diagy[2])

    if PLOT:
        # Create a figure with a gridspec layout
        fig = plt.figure(figsize=(12, 12))
        gs = fig.add_gridspec(
            2, 2, width_ratios=[3, 1], height_ratios=[1, 3], hspace=0, wspace=0
        )

        # Plot the image
        ax_img = fig.add_subplot(gs[1, 0])
        im = ax_img.imshow(image, origin="lower", cmap="viridis", extent=extent)
        ax_img.grid(color="white", ls="--", alpha=0.5)
        ax_img.set_xlabel("X Offset (arcmin)")
        ax_img.set_ylabel("Y Offset (arcmin)")
        ax_img.axvline(x_off, c="w", ls="--")
        ax_img.axhline(y_off, c="w", ls="--")
        # Plot the x-cut below
        ax_xcut = fig.add_subplot(gs[0, 0], sharex=ax_img)

        ax_xcut.errorbar(
            x_arcmin,
            image[mid_y, :],
            noise[mid_y, :],
            marker="o",
            ls="",
            color="k",
            label="data",
        )

        ax_xcut.plot(
            x_arcmin,
            best_fit_gaussx(x_arcmin),
            color="cyan",
            ls="--",
            label="best-fit:\n"
            r"$\hat{x}$=%.2f$\pm$%.2f"
            "\n"
            r"FWHM=%.2f$\pm$%.2f" % (x_off, x_off_err, x_fwhm, x_fwhm_err),
        )

        ax_xcut.set_ylabel("Flux")
        ax_xcut.tick_params(axis="x", labelbottom=False)
        ax_xcut.grid(True, linestyle="--", alpha=0.5)
        ax_xcut.legend(loc="upper right", fontsize=10)
        # Extend grid lines into the side plot
        ax_xcut_twin = ax_xcut.twiny()
        ax_xcut_twin.set_xlim(ax_img.get_xlim())

        # Plot the y-cut to the right (flipped)
        ax_ycut = fig.add_subplot(gs[1, 1], sharey=ax_img)
        ax_ycut.errorbar(
            image[:, mid_x],
            y_arcmin,
            xerr=noise[:, mid_x],
            marker="o",
            ls="",
            color="k",
            label="data",
        )

        ax_ycut.plot(
            best_fit_gaussy(y_arcmin),
            y_arcmin,
            color="cyan",
            label="best-fit:\n"
            r"$\hat{y}$=%.2f$\pm$%.2f"
            "\n"
            r"FWHM=%.2f$\pm$%.2f" % (y_off, y_off_err, y_fwhm, y_fwhm_err),
        )

        ax_ycut.set_xlabel("Flux")
        ax_ycut.tick_params(axis="y", labelleft=False)
        ax_ycut.grid(True, linestyle="--", alpha=0.5)
        ax_ycut.legend(loc="lower right", fontsize=10)

        # Extend grid lines into the side plot
        ax_ycut_twin = ax_ycut.twinx()
        ax_ycut_twin.set_ylim(ax_img.get_ylim())

        # Invert the y-axis of the cut plot to match the image
        ax_ycut.invert_yaxis()
        plt.show()

    return x_off, x_off_err, x_fwhm, x_fwhm_err, y_off, y_off_err, y_fwhm, y_fwhm_err


def photutils_gauss_slice_fit(
    flux_map,
    snr_map,
    source_catalog: dict,
    rakey: str = "RADeg",
    deckey: str = "decDeg",
    fluxkey: str = "fluxJy",
    size_deg=0.1,
    PLOT=False,
):
    """ """
    from pixell import reproject
    from pixell.utils import arcmin, degree

    ras = source_catalog[rakey]
    decs = source_catalog[deckey]
    fluxes = source_catalog[fluxkey]
    x_offsets = []
    y_offsets = []
    x_uncertainties = []
    y_uncertainties = []
    x_fwhms = []
    y_fwhms = []
    x_fwhm_uncertainties = []
    y_fwhm_uncertainties = []

    for i in range(len(ras)):
        map_res = abs(flux_map.wcs.wcs.cdelt[0] * degree)
        print(ras[i], decs[i], fluxes[i])
        pix = flux_map.sky2pix([decs[i] * degree, ras[i] * degree])
        thumbsize = size_deg * degree / map_res
        # print(flux_map.shape,pix,thumbsize)
        if (
            pix[0] + thumbsize > flux_map.shape[0]
            or pix[1] + thumbsize > flux_map.shape[1]
            or pix[0] - thumbsize < 0
            or pix[1] - thumbsize < 0
        ):
            continue
        try:
            flux_thumb = reproject.thumbnails(
                flux_map,
                [decs[i] * degree, ras[i] * degree],
                r=size_deg * degree,
                res=0.5 * map_res,
            )
            snr_thumb = reproject.thumbnails(
                snr_map,
                [decs[i] * degree, ras[i] * degree],
                r=size_deg * degree,
                res=0.5 * map_res,
            )
            noise_thumb = flux_thumb / snr_thumb
        except Exception as e:
            print(e)
            continue
        if np.all(np.isnan(flux_thumb)):
            print("map all nan")
            continue

        map_res = abs(flux_thumb.wcs.wcs.cdelt[0] * degree)

        x_off, x_off_err, x_fwhm, x_fwhm_err, y_off, y_off_err, y_fwhm, y_fwhm_err = (
            fit_xy_slices(flux_thumb, noise_thumb, map_res / arcmin, PLOT=PLOT)
        )

        x_offsets.append(x_off)
        y_offsets.append(y_off)
        x_uncertainties.append(x_off_err)
        y_uncertainties.append(y_off_err)
        x_fwhms.append(x_fwhm)
        y_fwhms.append(y_fwhm)
        x_fwhm_uncertainties.append(x_fwhm_err)
        y_fwhm_uncertainties.append(y_fwhm_err)

    return (
        np.asarray(x_offsets),
        np.asarray(y_offsets),
        np.asarray(x_uncertainties),
        np.asarray(y_uncertainties),
        np.asarray(x_fwhms),
        np.asarray(y_fwhms),
        np.asarray(x_fwhm_uncertainties),
        np.asarray(y_fwhm_uncertainties),
    )
