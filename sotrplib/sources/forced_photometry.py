import math

import numpy as np
import structlog
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropydantic import AstroPydanticQuantity
from numpy.typing import ArrayLike
from pixell import enmap, reproject
from pydantic import BaseModel
from scipy.optimize import curve_fit
from structlog import get_logger
from structlog.types import FilteringBoundLogger
from tqdm import tqdm

from sotrplib.maps.core import ProcessableMap
from sotrplib.maps.pointing import MapPointingOffset
from sotrplib.sources.sources import (
    CrossMatch,
    MeasuredSource,
    RegisteredSource,
)
from sotrplib.utils.utils import get_frequency, get_fwhm


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
    failed: bool = False
    failure_reason: str | None = None


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


class Gaussian2DFitter:
    """
    provide class for fitting a 2D gaussian to a source thumbnail.

    Parameters
    ----------
    source : MeasuredSource
        The source object containing the thumbnail to fit.
    fwhm_guess : AstroPydanticQuantity[u.arcmin], optional
        Initial guess for the FWHM of the Gaussian, by default u.Quantity(2.2, "arcmin")
    force_center : bool, optional
        Whether to force the Gaussian center to be at the thumbnail center, by default False
    reprojected : bool, optional
        Whether the thumbnail was reprojected (affects RA offset sign), by default False
    allowable_center_offset : AstroPydanticQuantity[u.arcmin], optional
        Maximum allowable offset from the thumbnail center for the fit, by default u.Quantity(1.0, "arcmin")
    """

    def __init__(
        self,
        source: MeasuredSource,
        fwhm_guess: AstroPydanticQuantity[u.arcmin] = u.Quantity(2.2, "arcmin"),
        force_center: bool = False,
        reprojected: bool = False,
        allowable_center_offset: AstroPydanticQuantity[u.arcmin] = u.Quantity(
            1.0, "arcmin"
        ),
        debug: bool = False,
        log: FilteringBoundLogger | None = None,
    ):
        self.log = log or get_logger()
        self.log = self.log.bind(func_name="Gaussian2DFitter")
        self.source = source
        self.fwhm_guess = fwhm_guess
        self.force_center = force_center
        self.reprojected = reprojected
        self.debug = debug
        self.allowable_center_offset = allowable_center_offset

    def initialize_model(self):
        """
        Docstring for initialize_model

        Initialize the 2D Gaussian model and initial parameter guesses
        used by scipy curve_fit.

        Forcing center removes the x0,y0 from the fit parameters.
        """

        ny, nx = self.source.thumbnail.shape

        # Define coordinate system centered at the image middle (i.e. the thumbnail center)
        x = np.arange(nx) - (nx - 1) / 2
        y = np.arange(ny) - (ny - 1) / 2
        X, Y = np.meshgrid(x, y)

        xy = (X.ravel(), Y.ravel())

        self.xy = xy
        # Initial guess for parameters
        amplitude_guess = self.source.thumbnail.max()
        y0_guess, x0_guess = np.unravel_index(
            np.argmax(self.source.thumbnail), self.source.thumbnail.shape
        )
        x0_guess -= (nx - 1) / 2
        y0_guess -= (ny - 1) / 2

        sigma_guess = (
            self.fwhm_guess.to(u.arcmin).value
            / abs(self.source.thumbnail_res.to(u.arcmin).value)
            / 2.355
        )  # Rough width estimate, in pixels
        theta_guess = 0
        offset_guess = 0

        initial_guess = [
            amplitude_guess,
            x0_guess,
            y0_guess,
            sigma_guess[0],
            sigma_guess[1],
            theta_guess,
            offset_guess,
        ]

        ## if force center, fix the x0,y0 guesses by not varying them
        if self.force_center:

            def gaussian_2d_model(xy, amplitude, sigma_x, sigma_y, theta, offset):
                return gaussian_2d(
                    xy, amplitude, x0_guess, y0_guess, sigma_x, sigma_y, theta, offset
                )

            initial_guess = [
                amplitude_guess,
                sigma_guess[0],
                sigma_guess[1],
                theta_guess,
                offset_guess,
            ]
            if self.debug:
                self.log.debug(
                    "curve_fit_2d_gaussian.force_center", initial_guess=initial_guess
                )
        else:
            if self.debug:
                self.log.debug(
                    "curve_fit_2d_gaussian.free_center", initial_guess=initial_guess
                )
            gaussian_2d_model = gaussian_2d

        self.model = gaussian_2d_model
        self.initial_guess = initial_guess

    def fit(self) -> GaussianFitParameters:
        """
        Set up and perform the 2D Gaussian fit using scipy curve_fit.

        sets the MeasuredSource fit_params attribute to the fit results.

        """

        self.fit_params = GaussianFitParameters()
        allowable_center_offset_pixels = self.allowable_center_offset.to(
            u.arcmin
        ).value / abs(self.source.thumbnail_res.to(u.arcmin).value)
        ## set bounds if not forcing center
        try:
            bounds = (-np.inf, np.inf)
            if not self.force_center:
                bounds = (
                    [
                        -np.inf,
                        self.initial_guess[1] - allowable_center_offset_pixels[0],
                        self.initial_guess[2] - allowable_center_offset_pixels[1],
                        -np.inf,
                        -np.inf,
                        -np.inf,
                        -np.inf,
                    ],
                    [
                        np.inf,
                        self.initial_guess[1] + allowable_center_offset_pixels[0],
                        self.initial_guess[2] + allowable_center_offset_pixels[1],
                        np.inf,
                        np.inf,
                        np.inf,
                        np.inf,
                    ],
                )
            popt, pcov = curve_fit(
                self.model,
                self.xy,
                self.source.thumbnail.ravel(),
                p0=self.initial_guess,
                bounds=bounds,
            )

        except RuntimeError:
            self.log.error("curve_fit_2d_gaussian.curve_fit.failed")
            self.fit_params.failed = True
            self.fit_params.failure_reason = "curve_fit_runtime_error"
            return self.fit_params

        perr = np.sqrt(np.diag(pcov))  # Parameter uncertainties
        if self.debug:
            self.log.debug(
                "curve_fit_2d_gaussian.curve_fit.success", popt=popt, perr=perr
            )
        # Extract parameters and uncertainties
        if self.force_center:
            amplitude, sigma_x, sigma_y, theta, offset = popt
            amplitude_err, sigma_x_err, sigma_y_err, theta_err, offset_err = perr
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
            sign_flip = -1
            dec_offset = sign_flip * y0 * self.source.thumbnail_res[1]
            ## if the thumbnail is reprojected or just cut out of the map
            ## determines the sign of the RA offset and the declination
            ## correction.
            ra_offset = sign_flip * x0 * self.source.thumbnail_res[0]
            if not self.reprojected:
                ra_offset *= math.cos(self.source.dec.to(u.rad).value)
            if ra_offset > 180.0 * u.deg:
                ra_offset -= 360.0 * u.deg
            elif ra_offset < -180.0 * u.deg:
                ra_offset += 360.0 * u.deg

            ra_err = x0_err * abs(self.source.thumbnail_res[0])
            dec_err = y0_err * abs(self.source.thumbnail_res[1])

        fwhm_x = 2.355 * sigma_x * abs(self.source.thumbnail_res[0])
        fwhm_y = 2.355 * sigma_y * abs(self.source.thumbnail_res[1])
        fwhm_x_err = 2.355 * sigma_x_err * abs(self.source.thumbnail_res[0])
        fwhm_y_err = 2.355 * sigma_y_err * abs(self.source.thumbnail_res[1])
        ## effectively set offset to zero
        amplitude = AstroPydanticQuantity(
            amplitude + offset, self.source.thumbnail_unit
        )
        amplitude_err = AstroPydanticQuantity(
            np.sqrt(amplitude_err**2 + offset_err**2), self.source.thumbnail_unit
        )

        self.fit_params = GaussianFitParameters(
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
            forced_center=self.force_center,
        )
        return self.fit_params


def scipy_2d_gaussian_fit(
    input_map: ProcessableMap,
    source_list: list[RegisteredSource],
    flux_lim_fit_centroid: u.Quantity = u.Quantity(0.3, "Jy"),
    thumbnail_half_width: u.Quantity = u.Quantity(0.25, "deg"),
    fwhm: u.Quantity | None = None,
    reproject_thumb: bool = False,
    pointing_residuals: MapPointingOffset | None = None,
    allowable_center_offset: u.Quantity = u.Quantity(1.0, "arcmin"),
    flags: dict = {},
    log: FilteringBoundLogger | None = None,
    debug: bool = False,
) -> list[MeasuredSource]:
    """
    Fit 2D Gaussian models to a list of registered sources on a map using SciPy.
    Parameters
    ----------
    input_map : ProcessableMap
        Map object containing the image data on which the sources are detected
        and to which the 2D Gaussian models are fitted.
    source_list : list[RegisteredSource]
        List of registered source objects whose positions are used as initial
        guesses for the Gaussian fits.
    flux_lim_fit_centroid : astropy.units.Quantity, optional
        Minimum peak flux density used to decide whether to fit for the source
        centroid. Sources with peak flux below this limit may have their
        centroid fixed. The default is 0.3 Jy.
    thumbnail_half_width : astropy.units.Quantity, optional
        Half-width of the thumbnail cut-out around each source position, in
        angular units. The default is 0.25 deg.
    fwhm : astropy.units.Quantity or None, optional
        Full-width at half maximum (FWHM) of the instrumental beam. If given,
        it is used to set the initial size (and/or bounds) of the Gaussian.
        If None, the FWHM is inferred from the map or other defaults.
     reproject_thumb : bool, optional
        If True, thumbnails are reprojected (e.g., to a local tangent-plane
        projection) before fitting. If False, thumbnails are extracted in the
        native projection. Defaults to False.
    pointing_residuals : MapPointingOffset or None, optional
        Optional pointing residuals to apply to the nominal source positions
        before cutting thumbnails and fitting.
    allowable_center_offset : astropy.units.Quantity, optional
        Maximum allowed offset between the fitted source position and the
        nominal catalog position. Fits exceeding this offset may be rejected.
        The default is 1.0 arcmin.
    flags : dict, optional
        Dictionary mapping flag names (str) to boolean lists of length
        ``len(source_list)``. For each source, all flag names whose list entry
        is True are attached to the corresponding measured source.
    log : structlog.types.FilteringBoundLogger or None, optional
        Logger instance used for structured logging. If None, a default logger
        from :func:`structlog.get_logger` is created and used.
    debug : bool, optional
        If True, enable additional debug output and/or diagnostic behaviour
        during fitting. Defaults to False.

    Returns
    -------
    list[MeasuredSource]
        List of measured source objects containing the results of the 2D
        Gaussian fits, including fit parameters and status flags.
    """
    log = log or get_logger()
    log = log.bind(func_name="scipy_2d_gaussian_fit")
    preamble = "sources.fitting.scipy_2d_gaussian_fit."
    fit_method = "2d_gaussian"
    fit_sources = []
    for i in tqdm(
        range(len(source_list)),
        desc="Cutting thumbnails and fitting sources w 2D Gaussian",
    ):
        source = source_list[i]

        source_flags = []
        for flag in flags:
            if flags[flag][i]:
                source_flags.append(flag)
        ## apply pointing residuals to source position
        source_pos = SkyCoord(ra=source.ra, dec=source.dec)
        source_pos = (
            pointing_residuals.apply_offset_at_position(source_pos)
            if pointing_residuals
            else source_pos
        )

        source.ra = source_pos.ra
        source.dec = source_pos.dec

        source_name = source.source_id
        map_res = input_map.map_resolution
        size_pix = thumbnail_half_width / map_res
        pix = input_map.flux.sky2pix(
            [source.dec.to(u.rad).value, source.ra.to(u.rad).value]
        )

        ## set default fit to empty parameters and fit_failed.
        ## once the fit is successfully completed, this will be updated.
        fit = GaussianFitParameters()
        fit.failed = True

        ## if outside the map; dont attempt to fit it.
        if (
            pix[0] + size_pix > input_map.flux.shape[0]
            or pix[1] + size_pix > input_map.flux.shape[1]
            or pix[0] - size_pix < 0
            or pix[1] - size_pix < 0
        ):
            log.warning(
                f"{preamble}source_outside_map_bounds_after_pointing_correction",
                source=source_name,
            )
            forced_source = MeasuredSource(
                ra=source.ra,
                dec=source.dec,
                flux=source.flux,
                source_id=source.source_id,
                source_type=source.source_type,
                instrument=input_map.instrument,
                array=input_map.array,
                frequency=get_frequency(input_map.frequency),
                flags=source_flags,
                crossmatches=[
                    CrossMatch(
                        ra=source.ra,
                        dec=source.dec,
                        observation_time=None,
                        source_id=source.source_id,
                        source_type=source.source_type,
                        probability=1.0,  ## todo: set properly
                        catalog_name=source.catalog_name,
                        flux=source.flux,
                        angular_separation=None,
                    )
                ],
            )
            forced_source.fit_method = fit_method
            forced_source.fit_failed = True
            fit.failure_reason = "source_outside_map_bounds"
            forced_source.fit_failure_reason = "source_outside_map_bounds"
            log.warning(
                f"{preamble}source_outside_map_bounds",
                source=source_name,
                thumb_size_pix=size_pix.value,
            )
            forced_source.fit_params = fit.model_dump()
            fit_sources.append(forced_source)
            continue

        t_start, t_mean, t_end = input_map.get_pixel_times(pix)
        ## setup default MeasuredSource
        forced_source = MeasuredSource(
            ra=source.ra,
            dec=source.dec,
            flux=source.flux,
            source_id=source.source_id,
            source_type=source.source_type,
            observation_start_time=t_start,
            observation_mean_time=t_mean,
            observation_end_time=t_end,
            instrument=input_map.instrument,
            array=input_map.array,
            frequency=get_frequency(input_map.frequency),
            flags=source_flags,
            crossmatches=[
                CrossMatch(
                    ra=source.ra,
                    dec=source.dec,
                    observation_time=t_mean,
                    source_id=source.source_id,
                    source_type=source.source_type,
                    probability=1.0,
                    flux=source.flux,
                    catalog_name=source.catalog_name,
                    angular_separation=None,
                )
            ],
        )

        try:
            forced_source.extract_thumbnail(
                input_map,
                thumb_width=thumbnail_half_width,
                reproject_thumb=reproject_thumb,
            )
        except Exception as e:
            log.error(
                f"{preamble}extract_thumbnail_failed", source=source_name, error=e
            )
            forced_source.fit_failure_reason = "extract_thumbnail_failed"
            fit_sources.append(forced_source)
            continue

        if np.any(np.isnan(forced_source.thumbnail)):
            log.warning(f"{preamble}flux_thumb_has_nan", source=source_name)
            forced_source.fit_failure_reason = "flux_thumb_has_nan"
            fit_sources.append(forced_source)
            continue

        fitter = Gaussian2DFitter(
            forced_source,
            reprojected=reproject_thumb,
            fwhm_guess=fwhm
            if fwhm is not None
            else get_fwhm(freq=input_map.frequency, arr=input_map.array),
            force_center=forced_source.flux < flux_lim_fit_centroid
            if forced_source.flux is not None
            else False,
            allowable_center_offset=allowable_center_offset,
            debug=debug,
            log=log,
        )

        fitter.initialize_model()
        fit = fitter.fit()
        if debug:
            log.debug(f"{preamble}gauss_fit", source=source_name, fit=fit)

        forced_source.fit_failed = fit.failed
        forced_source.fit_failure_reason = fit.failure_reason
        forced_source.flux = fit.amplitude
        forced_source.err_flux = fit.amplitude_err
        forced_source.snr = (
            (fit.amplitude / fit.amplitude_err).value
            if (fit.amplitude_err is not None) and (fit.amplitude is not None)
            else None
        )

        forced_source.offset_ra = fit.ra_offset
        forced_source.offset_dec = fit.dec_offset
        forced_source.fwhm_ra = fit.fwhm_ra
        forced_source.fwhm_dec = fit.fwhm_dec
        forced_source.err_fwhm_ra = fit.fwhm_ra_err
        forced_source.err_fwhm_dec = fit.fwhm_dec_err
        forced_source.fit_params = fit.model_dump()
        fit_sources.append(forced_source)

    n_successful = 0
    for s in fit_sources:
        if not s.fit_failed:
            n_successful += 1
    log.info(f"{preamble}fits_complete", n_sources=len(fit_sources), n_fit=n_successful)
    return fit_sources


def convert_catalog_to_registered_source_objects(
    catalog_sources: dict,
    source_type: str | None = None,
    log: FilteringBoundLogger | None = None,
):
    """
    take each source in the catalog and convert to a list
    of RegisteredSource objects.

    """
    log = log or structlog.get_logger()
    log = log.bind(func_name="convert_catalog_to_registered_source_objects")

    if isinstance(catalog_sources, list):
        log.info(
            "convert_catalog_to_registered_source_objects.catalog_is_list",
            num_sources=len(catalog_sources),
        )
        return catalog_sources
    if not catalog_sources:
        log.warning(
            "convert_catalog_to_registered_source_objects.catalog_is_empty",
            num_sources=0,
        )
        return []

    known_sources = []
    for i in range(len(catalog_sources["name"])):
        source_ra = catalog_sources["RADeg"][i] % 360
        source_dec = catalog_sources["decDeg"][i]
        cs = RegisteredSource(
            ra=source_ra * u.deg,
            dec=source_dec * u.deg,
            flux=catalog_sources["fluxJy"][i] * u.Jy,
            source_id=catalog_sources["name"][i],
            source_type=source_type,
        )
        if "ra_offset_arcmin" in catalog_sources:
            if catalog_sources["ra_offset_arcmin"][i]:
                cs.err_ra = catalog_sources["err_ra_offset_arcmin"][i] * u.arcmin
                cs.err_dec = catalog_sources["err_dec_offset_arcmin"][i] * u.arcmin

        cs.add_crossmatch(
            CrossMatch(
                source_id=catalog_sources["name"][i],
                probability=1.0,
                catalog_idx=i,
            )
        )
        known_sources.append(cs)
    log.info(
        "convert_catalog_to_registered_source_objects.catalog_converted",
        num_sources=len(known_sources),
    )
    return known_sources


def return_initial_catalog_entry(
    source,
    add_dict: dict = {},
    keyconv={},
):
    """
    source is a MeasuredSource object
    """
    outdict = {}
    for item in vars(source):
        outdict[item] = getattr(source, item)
    outdict.update(add_dict)
    for key in keyconv:
        if key in outdict:
            outdict[keyconv[key]] = outdict.pop(key)

    return outdict


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
