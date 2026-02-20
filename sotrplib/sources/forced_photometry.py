import math

import lmfit
import numpy as np
import structlog
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropydantic import AstroPydanticQuantity
from numpy.typing import ArrayLike
from pydantic import BaseModel
from scipy.optimize import curve_fit
from structlog import get_logger
from structlog.types import FilteringBoundLogger
from tqdm import tqdm

from sotrplib.maps.core import ProcessableMap
from sotrplib.maps.pointing import EmptyPointingOffset, MapPointingOffset, PointingData
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
    chisquared: float | None = None
    reduced_chisquared: float | None = None
    goodness_of_fit: float | None = None


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


def gaussian_2d_lmfit(
    x: ArrayLike,
    y: ArrayLike,
    amplitude: float,
    x0: float,
    y0: float,
    sigma_x: float,
    sigma_y: float,
    theta: float,
    offset: float,
):
    """
    wrapper for the 2d gaussian to have the
    required signature for lmfit Model to work with independent variables x and y.

    """
    return gaussian_2d((x, y), amplitude, x0, y0, sigma_x, sigma_y, theta, offset)


class Gaussian2DFitter:
    """
    provide class for fitting a 2D gaussian to a source thumbnail.

    Parameters
    ----------
    source : MeasuredSource
        The source object containing the thumbnail to fit.
    fit_method: str
        The method to use for fitting the Gaussian. Options are "scipy" for
        scipy.optimize.curve_fit or "lmfit" for the lmfit library. The default is "lmfit".
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
        fit_method: str = "lmfit",
        fwhm_guess: AstroPydanticQuantity[u.arcmin] = u.Quantity(2.2, "arcmin"),
        force_center: bool = False,
        reprojected: bool = False,
        allowable_center_offset: AstroPydanticQuantity[u.arcmin] = u.Quantity(
            1.0, "arcmin"
        ),
        log: FilteringBoundLogger | None = None,
    ):
        self.log = log or get_logger()
        self.log = self.log.bind(func_name="Gaussian2DFitter")
        self.source = source
        self.fwhm_guess = fwhm_guess
        self.force_center = force_center
        self.reprojected = reprojected
        self.allowable_center_offset = allowable_center_offset
        self.fit_method = fit_method
        self.model = None

    def initialize_model(self):
        """
        Docstring for initialize_model

        Initialize the 2D Gaussian model and initial parameter guess

        Forcing center removes the x0,y0 from the fit parameters.
        """

        if self.fit_method == "lmfit":
            self.model = lmfit.Model(gaussian_2d_lmfit, independent_vars=["x", "y"])
        else:
            self.model = gaussian_2d

        ny, nx = self.source.thumbnail.shape

        # Define coordinate system centered at the image middle (i.e. the thumbnail center)
        x = np.arange(nx) - (nx - 1) / 2
        y = np.arange(ny) - (ny - 1) / 2
        X, Y = np.meshgrid(x, y)
        self.X = X
        self.Y = Y
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

        allowable_center_offset_pixels = self.allowable_center_offset.to(
            u.arcmin
        ).value / abs(self.source.thumbnail_res.to(u.arcmin).value)

        if self.fit_method == "scipy":
            ## if force center, fix the x0,y0 guesses by not varying them
            if self.force_center:

                def gaussian_2d_model(xy, amplitude, sigma_x, sigma_y, theta, offset):
                    return gaussian_2d(
                        xy,
                        amplitude,
                        x0_guess,
                        y0_guess,
                        sigma_x,
                        sigma_y,
                        theta,
                        offset,
                    )

                initial_guess = [
                    amplitude_guess,
                    sigma_guess[0],
                    sigma_guess[1],
                    theta_guess,
                    offset_guess,
                ]
            else:
                gaussian_2d_model = gaussian_2d
            self.model = gaussian_2d_model
            self.initial_guess = initial_guess

        else:
            self.initial_guess = self.model.make_params(
                amplitude=amplitude_guess,
                x0=x0_guess,
                y0=y0_guess,
                sigma_x=sigma_guess[0],
                sigma_y=sigma_guess[1],
                theta=theta_guess,
                offset=offset_guess,
            )

            if self.force_center:
                self.initial_guess["x0"].set(vary=False)
                self.initial_guess["y0"].set(vary=False)
            else:
                self.initial_guess["x0"].set(
                    min=x0_guess - allowable_center_offset_pixels[0],
                    max=x0_guess + allowable_center_offset_pixels[0],
                )
                self.initial_guess["y0"].set(
                    min=y0_guess - allowable_center_offset_pixels[1],
                    max=y0_guess + allowable_center_offset_pixels[1],
                )

            self.initial_guess["sigma_x"].set(min=0.0, max=nx)
            self.initial_guess["sigma_y"].set(min=0.0, max=ny)
            self.initial_guess["theta"].set(min=0, max=np.pi)
        self.log.info(
            "Gaussian2DFitter.model_initialized",
            fit_method=self.fit_method,
            model=self.model,
            initial_guess=self.initial_guess,
        )

    def fit(self) -> GaussianFitParameters:
        if self.fit_method == "scipy":
            return self.fit_scipy()
        elif self.fit_method == "lmfit":
            return self.fit_lmfit()
        else:
            raise ValueError(f"Unsupported fit method: {self.fit_method}")

    def fit_scipy(self) -> GaussianFitParameters:
        """
        Set up and perform the 2D Gaussian fit with Scipy's curve_fit

        sets the MeasuredSource fit_params attribute to the fit results.

        """

        self.fit_params = GaussianFitParameters()

        if self.model is None:
            self.log.error(f"{self.fit_model}_gaussian.model_not_available")
            self.fit_params.failed = True
            self.fit_params.failure_reason = "model_not_available"
            return self.fit_params

        allowable_center_offset_pixels = self.allowable_center_offset.to(
            u.arcmin
        ).value / abs(self.source.thumbnail_res.to(u.arcmin).value)
        ## set bounds if not forcing center
        bounds = (-np.inf, np.inf)
        try:
            if not self.force_center:
                bounds = (
                    [
                        -np.inf,
                        self.initial_guess[1] - allowable_center_offset_pixels[0],
                        self.initial_guess[2] - allowable_center_offset_pixels[1],
                        0,
                        0,
                        0,
                        -np.inf,
                    ],
                    [
                        np.inf,
                        self.initial_guess[1] + allowable_center_offset_pixels[0],
                        self.initial_guess[2] + allowable_center_offset_pixels[1],
                        np.inf,
                        np.inf,
                        np.pi,
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
        self.log.debug("curve_fit_2d_gaussian.curve_fit.success", popt=popt, perr=perr)
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
        y_fit = self.model(self.xy, *popt)
        ss_res = np.sum((self.source.thumbnail.ravel() - y_fit) ** 2)
        ss_tot = np.sum(
            (self.source.thumbnail.ravel() - np.mean(self.source.thumbnail.ravel()))
            ** 2
        )
        rsquared = 1 - (ss_res / ss_tot)

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
            goodness_of_fit=rsquared,
        )
        return self.fit_params

    def fit_lmfit(self) -> GaussianFitParameters:
        """
        Set up and perform the 2D Gaussian fit using lmfit.
        """
        self.fit_params = GaussianFitParameters()

        if self.model is None:
            self.log.error("lmfit_2d_gaussian.lmfit_not_available")
            self.fit_params.failed = True
            self.fit_params.failure_reason = "lmfit_not_available"
            return self.fit_params

        try:
            result = self.model.fit(
                self.source.thumbnail, self.initial_guess, x=self.X, y=self.Y
            )
        except Exception as exc:  # noqa: BLE001
            self.log.error("lmfit_2d_gaussian.fit_failed", error=exc)
            self.fit_params.failed = True
            self.fit_params.failure_reason = "lmfit_fit_error"
            return self.fit_params

        if not result.success:
            self.log.error("lmfit_2d_gaussian.fit_failed", error=result.message)
            self.fit_params.failed = True
            self.fit_params.failure_reason = "lmfit_fit_failed"
            return self.fit_params

        params = result.params
        amplitude = params["amplitude"].value
        amplitude_err = params["amplitude"].stderr
        sigma_x = params["sigma_x"].value
        sigma_y = params["sigma_y"].value
        sigma_x_err = params["sigma_x"].stderr
        sigma_y_err = params["sigma_y"].stderr
        theta = params["theta"].value
        theta_err = params["theta"].stderr
        offset = params["offset"].value
        offset_err = params["offset"].stderr
        chisquared = result.chisqr
        reduced_chisquared = result.redchi
        goodness_of_fit = result.rsquared

        if self.force_center:
            ra_offset, dec_offset = None, None
            ra_err, dec_err = None, None
        else:
            x0 = params["x0"].value
            y0 = params["y0"].value
            x0_err = params["x0"].stderr
            y0_err = params["y0"].stderr

            sign_flip = -1
            dec_offset = sign_flip * y0 * self.source.thumbnail_res[1]
            ra_offset = sign_flip * x0 * self.source.thumbnail_res[0]
            if not self.reprojected:
                ra_offset *= math.cos(self.source.dec.to(u.rad).value)
            if ra_offset > 180.0 * u.deg:
                ra_offset -= 360.0 * u.deg
            elif ra_offset < -180.0 * u.deg:
                ra_offset += 360.0 * u.deg

            ra_err = (
                x0_err * abs(self.source.thumbnail_res[0])
                if x0_err is not None
                else None
            )
            dec_err = (
                y0_err * abs(self.source.thumbnail_res[1])
                if y0_err is not None
                else None
            )

        fwhm_x = 2.355 * sigma_x * abs(self.source.thumbnail_res[0])
        fwhm_y = 2.355 * sigma_y * abs(self.source.thumbnail_res[1])
        fwhm_x_err = (
            2.355 * sigma_x_err * abs(self.source.thumbnail_res[0])
            if sigma_x_err is not None
            else None
        )
        fwhm_y_err = (
            2.355 * sigma_y_err * abs(self.source.thumbnail_res[1])
            if sigma_y_err is not None
            else None
        )

        amplitude = AstroPydanticQuantity(
            amplitude + offset, self.source.thumbnail_unit
        )
        amplitude_err = (
            AstroPydanticQuantity(
                np.sqrt(amplitude_err**2 + offset_err**2), self.source.thumbnail_unit
            )
            if (amplitude_err is not None and offset_err is not None)
            else None
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
            theta_err=theta_err * u.rad if theta_err is not None else None,
            forced_center=self.force_center,
            chisquared=chisquared,
            reduced_chisquared=reduced_chisquared,
            goodness_of_fit=goodness_of_fit,
        )
        return self.fit_params


def gaussian_fit(
    input_map: ProcessableMap,
    source_list: list[RegisteredSource],
    fit_method: str = "lmfit",
    flux_lim_fit_centroid: u.Quantity = u.Quantity(0.3, "Jy"),
    thumbnail_half_width: u.Quantity = u.Quantity(0.25, "deg"),
    fwhm: u.Quantity | None = None,
    reproject_thumb: bool = False,
    pointing_residuals: MapPointingOffset | None = EmptyPointingOffset(),
    pointing_offset_data: PointingData | None = None,
    allowable_center_offset: u.Quantity = u.Quantity(1.0, "arcmin"),
    goodness_of_fit_threshold: float | None = None,
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
    fit_method : str, optional
        Method to use for fitting the Gaussian. Options are "scipy" for
        scipy.optimize.curve_fit or "lmfit" for the lmfit library. The default is "lmfit".
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
    goodness_of_fit_threshold : float or None, optional
        If not None, the minimum Pearson's r value for the fit to be considered successful.
        If None, no r value cut is applied. The default is None.
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

    log = log.bind(func_name=f"{fit_method}_2d_gaussian_fit")
    preamble = f"sources.fitting.{fit_method}_2d_gaussian_fit."
    fit_sources = []
    for i in tqdm(
        range(len(source_list)),
        desc=f"Cutting thumbnails and fitting sources w 2D Gaussian ({fit_method})",
    ):
        source = source_list[i]

        source_flags = []
        for flag in flags:
            if flags[flag][i]:
                source_flags.append(flag)
        ## apply pointing residuals to source position
        source_pos = SkyCoord(ra=source.ra, dec=source.dec)
        if pointing_residuals is not None:
            source_pos = pointing_residuals.apply_offset_at_position(
                source_pos, data=pointing_offset_data
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
            fit_method=f"{fit_method}_2d_gaussian",
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

        forced_source.observation_start_time = t_start
        forced_source.observation_mean_time = t_mean
        forced_source.observation_end_time = t_end

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
            forced_source.fit_failed = True
            forced_source.fit_failure_reason = "extract_thumbnail_failed"
            fit_sources.append(forced_source)
            continue

        if np.any(np.isnan(forced_source.thumbnail)):
            log.warning(f"{preamble}flux_thumb_has_nan", source=source_name)
            forced_source.fit_failed = True
            forced_source.fit_failure_reason = "flux_thumb_has_nan"
            fit_sources.append(forced_source)
            continue

        fitter = Gaussian2DFitter(
            forced_source,
            fit_method=fit_method,
            reprojected=reproject_thumb,
            fwhm_guess=fwhm
            if fwhm is not None
            else get_fwhm(
                freq=input_map.frequency,
                arr=input_map.array,
                instrument=input_map.instrument,
            ),
            force_center=forced_source.flux < flux_lim_fit_centroid
            if forced_source.flux is not None
            else False,
            allowable_center_offset=allowable_center_offset,
            log=log,
        )

        fitter.initialize_model()
        fit = fitter.fit()
        if debug:
            log.debug(f"{preamble}gauss_fit", source=source_name, fit=fit)

        if (
            goodness_of_fit_threshold is not None
            and fit.goodness_of_fit is not None
            and fit.goodness_of_fit < goodness_of_fit_threshold
        ):
            fit.failed = True
            fit.failure_reason = "goodness_of_fit_below_threshold"

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
