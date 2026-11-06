from __future__ import annotations

import math
from typing import TYPE_CHECKING

import lmfit
import numpy as np
import structlog
import uuid7 as uuid
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropydantic import AstroPydanticQuantity
from numpy.typing import ArrayLike
from pydantic import BaseModel
from structlog import get_logger
from structlog.types import FilteringBoundLogger
from tqdm import tqdm

from sotrplib.maps.core import ProcessableMap
from sotrplib.sources.sources import (
    CrossMatch,
    MeasuredSource,
    RegisteredSource,
)
from sotrplib.utils.utils import (
    SIGMA_TO_FWHM,
    get_frequency,
    get_fwhm,
    rescale_gaussian_axes,
)

if TYPE_CHECKING:
    from sotrplib.maps.pointing import PointingModel


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


class LmFitGaussian2DFitter:
    """
    provide class for fitting a 2D gaussian to a source thumbnail using LmFit.

    Parameters
    ----------
    source : MeasuredSource
        The source object containing the thumbnail to fit.
    fwhm_guess : AstroPydanticQuantity[u.arcmin], optional
        Initial guess for the FWHM of the Gaussian, by default u.Quantity(2.2, "arcmin")
    force_center : bool, optional
        If True, the Gaussian center is at the thumbnail center (the source
        position) and the beam shape is fixed: only the amplitude and the
        background are fit. If False, the fit
        also starts at the thumbnail center, but the center can move by
        allowable_center_offset from that position, and the shape is free.
        By default False.
    reprojected : bool, optional
        Whether the thumbnail was reprojected (affects RA offset sign), by default False
    allowable_center_offset : AstroPydanticQuantity[u.arcmin] or None, optional
        Maximum allowable offset from the thumbnail center for the fit, if
        force_center is False. If None, use fwhm_guess. By default None.
    fwhm_prior_width : float or None, optional
        Width of a Gaussian prior on the natural log of each fitted width,
        centered on fwhm_guess, if force_center is False. If None, no prior.
        By default 0.35.
    position_prior_width : float or None, optional
        Width of a Gaussian prior on the fitted position, in units of
        fwhm_guess and centered on the source position, if force_center is
        False. If None, no prior. By default 0.5.
    """

    def __init__(
        self,
        source: MeasuredSource,
        fwhm_guess: AstroPydanticQuantity[u.arcmin] = u.Quantity(2.2, "arcmin"),
        force_center: bool = False,
        reprojected: bool = False,
        allowable_center_offset: AstroPydanticQuantity[u.arcmin] | None = None,
        fwhm_prior_width: float | None = 0.35,
        position_prior_width: float | None = 0.5,
        log: FilteringBoundLogger | None = None,
    ):
        self.log = log or get_logger()
        self.log = self.log.bind(func_name="Gaussian2DFitter")
        self.source = source
        self.fwhm_guess = fwhm_guess
        self.force_center = force_center
        self.reprojected = reprojected
        self.allowable_center_offset = (
            allowable_center_offset
            if allowable_center_offset is not None
            else fwhm_guess
        )
        self.fwhm_prior_width = fwhm_prior_width
        self.position_prior_width = position_prior_width
        self.fit_method = "lmfit"
        self.model = None

    def _ra_pixel_scale(self) -> float:
        """
        The size on the sky of a thumbnail pixel in RA, relative to its size
        in the map coordinates.
        """
        return 1.0 if self.reprojected else math.cos(self.source.dec.to(u.rad).value)

    def initialize_model(self):
        """
        Docstring for initialize_model

        Initialize the 2D Gaussian model and initial parameter guess

        Forcing center removes x0, y0, sigma_x, sigma_y and theta from the
        fit parameters: the fit is a beam at the source position.
        """
        self.model = lmfit.Model(gaussian_2d_lmfit, independent_vars=["x", "y"])
        ny, nx = self.source.thumbnail.shape

        # Define coordinate system centered at the source position. This is
        # the middle of the thumbnail, if the thumbnail does not give a position.
        y_source, x_source = self.source.thumbnail_source_pixel or (
            (ny - 1) / 2,
            (nx - 1) / 2,
        )
        x = np.arange(nx) - x_source
        y = np.arange(ny) - y_source
        X, Y = np.meshgrid(x, y)
        self.X = X
        self.Y = Y
        xy = (X.ravel(), Y.ravel())
        self.xy = xy

        # The beam is round on the sky. Pixels in RA are smaller on the sky,
        # if the thumbnail is not reprojected.
        pixel_on_sky = abs(self.source.thumbnail_res.to(u.arcmin).value) * [
            self._ra_pixel_scale(),
            1.0,
        ]
        self.pixel_on_sky = pixel_on_sky
        sigma_guess = (
            self.fwhm_guess.to(u.arcmin).value / pixel_on_sky / SIGMA_TO_FWHM
        )  # Rough width estimate, in pixels
        self.data_weight = self._data_weight(sigma_guess)
        theta_guess = 0
        offset_guess = 0

        allowable_center_offset_pixels = (
            self.allowable_center_offset.to(u.arcmin).value / pixel_on_sky
        )

        # Initial guess for parameters.
        x0_guess, y0_guess = 0.0, 0.0
        amplitude_guess = self.source.thumbnail[
            np.clip(round(y_source), 0, ny - 1), np.clip(round(x_source), 0, nx - 1)
        ]

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
            # Also keep the beam shape fixed.
            self.initial_guess["x0"].set(vary=False)
            self.initial_guess["y0"].set(vary=False)
            self.initial_guess["sigma_x"].set(vary=False)
            self.initial_guess["sigma_y"].set(vary=False)
            self.initial_guess["theta"].set(vary=False)
        else:
            # A round start has no gradient in theta. The log keeps the
            # widths positive with no limits.
            self.initial_guess.add("ln_sigma_x", value=np.log(1.2 * sigma_guess[0]))
            self.initial_guess.add("ln_sigma_y", value=np.log(0.8 * sigma_guess[1]))
            self.initial_guess["sigma_x"].set(expr="exp(ln_sigma_x)")
            self.initial_guess["sigma_y"].set(expr="exp(ln_sigma_y)")
            self.initial_guess["theta"].set(value=np.pi / 4)
            self.initial_guess["x0"].set(
                min=x0_guess - allowable_center_offset_pixels[0],
                max=x0_guess + allowable_center_offset_pixels[0],
            )
            self.initial_guess["y0"].set(
                min=y0_guess - allowable_center_offset_pixels[1],
                max=y0_guess + allowable_center_offset_pixels[1],
            )

        self.log.debug(
            "Gaussian2DFitter.model_initialized",
            fit_method=self.fit_method,
            model=self.model,
            initial_guess=self.initial_guess,
        )

    def _data_weight(self, sigma_pixels: np.ndarray) -> float | None:
        """
        The weight of the thumbnail residuals against the priors: one over the
        thumbnail noise and the square root of the pixels per beam, because
        the pixels are correlated over the beam. None if the thumbnail has no
        noise, or if there are no priors.
        """
        if self.force_center or (
            self.fwhm_prior_width is None and self.position_prior_width is None
        ):
            return None
        thumbnail = np.asarray(self.source.thumbnail)
        noise = 1.4826 * np.median(np.abs(thumbnail - np.median(thumbnail)))
        if not np.isfinite(noise) or noise <= 0:
            return None
        return 1.0 / (noise * np.sqrt(2 * np.pi * np.prod(sigma_pixels)))

    def _image_residual(self, params: lmfit.Parameters) -> np.ndarray:
        return (
            np.asarray(self.source.thumbnail)
            - self.model.eval(params, x=self.X, y=self.Y).reshape(
                self.source.thumbnail.shape
            )
        ).ravel()

    def _residual(self, params: lmfit.Parameters) -> np.ndarray:
        image = self._image_residual(params)
        if self.data_weight is None:
            return image
        terms = [image * self.data_weight]
        fwhm = self.fwhm_guess.to_value(u.arcmin)
        if self.fwhm_prior_width is not None:
            sky_x, sky_y, _ = rescale_gaussian_axes(
                params["sigma_x"].value,
                params["sigma_y"].value,
                params["theta"].value,
                *self.pixel_on_sky,
            )
            terms.append(
                np.log(np.array([sky_x, sky_y]) * SIGMA_TO_FWHM / fwhm)
                / self.fwhm_prior_width
            )
        if self.position_prior_width is not None:
            position = np.array([params["x0"].value, params["y0"].value])
            terms.append(
                position * self.pixel_on_sky / (self.position_prior_width * fwhm)
            )
        return np.concatenate(terms)

    def fit(self) -> GaussianFitParameters:
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
            result = lmfit.minimize(self._residual, self.initial_guess)
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
        ## Fit statistics of the thumbnail only, without the priors.
        image_residual = self._image_residual(params)
        thumbnail = np.asarray(self.source.thumbnail)
        chisquared = float(np.sum(image_residual**2))
        reduced_chisquared = chisquared / max(image_residual.size - result.nvarys, 1)
        goodness_of_fit = 1.0 - chisquared / np.sum((thumbnail - thumbnail.mean()) ** 2)

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

        ## Principal widths and angle on the sky.
        res_x, res_y = abs(self.source.thumbnail_res)
        sky_x, sky_y, sky_theta = rescale_gaussian_axes(
            sigma_x,
            sigma_y,
            theta,
            res_x.to_value(u.arcmin) * self._ra_pixel_scale(),
            res_y.to_value(u.arcmin),
        )
        ## Report the angle in the sense of the model map, also if x is flipped.
        if self.source.thumbnail_res[0] * self.source.thumbnail_res[1] < 0:
            sky_theta = (np.pi - sky_theta) % np.pi
        fwhm_x = SIGMA_TO_FWHM * sky_x * u.arcmin
        fwhm_y = SIGMA_TO_FWHM * sky_y * u.arcmin
        fwhm_x_err = fwhm_x * sigma_x_err / sigma_x if sigma_x_err is not None else None
        fwhm_y_err = fwhm_y * sigma_y_err / sigma_y if sigma_y_err is not None else None

        ## The amplitude is the height above the fitted background.
        amplitude = AstroPydanticQuantity(amplitude, self.source.thumbnail_unit)
        amplitude_err = (
            AstroPydanticQuantity(amplitude_err, self.source.thumbnail_unit)
            if amplitude_err is not None
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
            theta=sky_theta * u.rad,
            theta_err=theta_err * u.rad if theta_err is not None else None,
            forced_center=self.force_center,
            chisquared=chisquared,
            reduced_chisquared=reduced_chisquared,
            goodness_of_fit=goodness_of_fit,
        )
        return self.fit_params


## A fit wider than this, in units of the beam FWHM, has the wide_fit flag.
WIDE_FIT_FWHM_RATIO = 3.0


def _fitted_position(
    source: MeasuredSource, fit: GaussianFitParameters
) -> tuple[float, float]:
    """
    The fitted (dec, ra) in radians: the source position, less the fit offsets
    (offset = source position - fitted position). If the fit has no offsets
    (forced center), this is the source position.
    """
    dec = source.dec.to_value(u.rad)
    ra = source.ra.to_value(u.rad)
    if fit.ra_offset is not None and fit.dec_offset is not None:
        dec -= fit.dec_offset.to_value(u.rad)
        ra -= fit.ra_offset.to_value(u.rad) / np.cos(dec)
    return dec, ra


def _map_flux_at_fit(
    input_map: ProcessableMap,
    source: MeasuredSource,
    fit: GaussianFitParameters,
) -> AstroPydanticQuantity | None:
    """
    Get the map flux at the fitted position, with cubic interpolation. Returns
    None if the fit failed, or if the flux is not finite near that position.
    """
    if fit.failed:
        return None
    dec, ra = _fitted_position(source, fit)
    y, x = np.round(input_map.flux.sky2pix([dec, ra])).astype(int)
    ## The spline filters all of the array it gets: use a small patch of the map.
    half = 8
    if y < half or x < half:
        return None
    patch = input_map.flux[y - half : y + half + 1, x - half : x + half + 1]
    if patch.shape != (2 * half + 1, 2 * half + 1) or not np.all(np.isfinite(patch)):
        return None
    flux = patch.at(np.array([[dec], [ra]]), mode="spline", order=3)[0]
    return AstroPydanticQuantity(float(flux), input_map.flux_units)


def _map_noise_at_fit(
    input_map: ProcessableMap,
    source: MeasuredSource,
    fit: GaussianFitParameters,
) -> AstroPydanticQuantity | None:
    """
    Get the map noise (flux / SNR) at the fitted source position.

    Returns None if the fit failed, or if the noise is not finite and
    positive at that position.
    """
    if fit.failed:
        return None
    dec, ra = _fitted_position(source, fit)
    ## noise = flux / SNR, as ProcessableMap.noise, but at one pixel only
    pos = np.array([[dec], [ra]])
    with np.errstate(divide="ignore", invalid="ignore"):
        noise = (
            input_map.flux.at(pos, mode="nn")[0] / input_map.snr.at(pos, mode="nn")[0]
        )
    if not np.isfinite(noise) or noise <= 0:
        return None
    return AstroPydanticQuantity(float(noise), input_map.flux_units)


def gaussian_fit(
    input_map: ProcessableMap,
    source_list: list[RegisteredSource],
    fit_method: str = "lmfit",
    fit_centroid: bool = False,
    fixed_fit_on_failure: bool = False,
    fwhm_prior_width: float | None = 0.35,
    position_prior_width: float | None = 0.5,
    thumbnail_half_width: u.Quantity = u.Quantity(10.0, "arcmin"),
    fwhm: u.Quantity | None = None,
    reproject_thumb: bool = False,
    pointing_model: PointingModel | None = None,
    allowable_center_offset: u.Quantity | None = None,
    goodness_of_fit_threshold: float | None = None,
    flags: dict = {},
    log: FilteringBoundLogger | None = None,
    debug: bool = False,
) -> list[MeasuredSource]:
    """
    Fit 2D Gaussian models to a list of registered sources on a map using lmfit.
    The flux of a source is the map value at the fitted position. The fit gives
    the position and the shape. A source with a fit wider than
    WIDE_FIT_FWHM_RATIO beams, on either axis, has the wide_fit flag.

    Parameters
    ----------
    input_map : ProcessableMap
        Map object containing the image data on which the sources are detected
        and to which the 2D Gaussian models are fitted.
    source_list : list[RegisteredSource]
        List of registered source objects whose positions are used as initial
        guesses for the Gaussian fits.
    fit_method : str, optional
        Kept for backwards compatibility; accepts any value but is ignored.
        lmfit is always used for fitting.
    fit_centroid : bool, optional
        If False, fit each source at its position with the beam shape fixed.
        If True, also fit the source position and shape, from the thumbnail
        center and in the limit of allowable_center_offset. A source with the
        nearby_source flag stays at its position. The default is False.
    fixed_fit_on_failure : bool, optional
        If True and a fit with a free position fails, fit again at the source
        position with the beam shape fixed. The default is False.
    fwhm_prior_width : float or None, optional
        Width of a Gaussian prior on the natural log of each fitted width,
        centered on the beam FWHM. If None, no prior. The default is 0.35.
    position_prior_width : float or None, optional
        Width of a Gaussian prior on the fitted position, in units of the beam
        FWHM and centered on the source position. If None, no prior. The
        default is 0.5.
    thumbnail_half_width : astropy.units.Quantity, optional
        Half-width of the thumbnail cut-out around each source position, in
        angular units. The default is 10 arcmin.
    fwhm : astropy.units.Quantity or None, optional
        Full-width at half maximum (FWHM) of the instrumental beam. If given,
        it is used to set the initial size (and/or bounds) of the Gaussian.
        If None, the FWHM is inferred from the map or other defaults.
     reproject_thumb : bool, optional
        If True, thumbnails are reprojected (e.g., to a local tangent-plane
        projection) before fitting. If False, thumbnails are extracted in the
        native projection. Defaults to False.
    pointing_model: PointingModel or None, optional
        If provided, this model is used to apply pointing corrections to the source positions
        before cutting thumbnails and fitting.
    allowable_center_offset : astropy.units.Quantity or None, optional
        Maximum offset of the fitted position from the source position, in RA
        and in Dec. If None, use the beam FWHM. The default is None.
    goodness_of_fit_threshold : float or None, optional
        If not None, the minimum R^2 value for the fit to be considered successful.
        If None, no R^2 value cut is applied. The default is None.
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

    log = log.bind(func_name="lmfit_2d_gaussian_fit")
    preamble = "sources.fitting.lmfit_2d_gaussian_fit."
    fit_sources = []
    for i in tqdm(
        range(len(source_list)),
        desc="Cutting thumbnails and fitting sources w 2D Gaussian (lmfit)",
    ):
        source = source_list[i]

        source_flags = []
        for flag in flags:
            if flags[flag][i]:
                source_flags.append(flag)
        ## apply pointing residuals to source position
        source_pos = SkyCoord(ra=source.ra, dec=source.dec)
        if pointing_model is not None:
            source_pos = pointing_model.predict(source_pos)

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

        # source_id is the catalog's human-readable name (e.g. "ACT-S
        # J1353.8+0151"), not a stable identifier. The catalog's own unique
        # id (e.g. socat's UUID) is already attached as catalog_idx on the
        # source's own crossmatch, set when it was fetched from the catalog
        # (see SOCat._sg_to_registered). Fall back to source_id only for
        # catalogs that don't pre-populate a crossmatch.
        existing_crossmatch = source.crossmatches[0] if source.crossmatches else None
        catalog_idx = (
            existing_crossmatch.catalog_idx
            if existing_crossmatch is not None
            else source.source_id
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
            fit_method="lmfit_2d_gaussian",
            crossmatches=[
                CrossMatch(
                    ra=source.ra,
                    dec=source.dec,
                    observation_time=None,
                    source_id=source.source_id,
                    source_type=source.source_type,
                    probability=1.0,  ## todo: set properly
                    catalog_name=source.catalog_name,
                    catalog_idx=catalog_idx,
                    flux=source.flux,
                    angular_separation=None,
                )
            ],
        )
        forced_source.set_map(input_map)
        if np.any(np.isnan(pix)):
            log.warning(
                f"{preamble}source_position_has_nan_pix",
                source=source_name,
            )
            forced_source.fit_failed = True
            forced_source.fit_failure_reason = "source_position_has_nan_pix"
            fit_sources.append(forced_source)
            continue
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

        beam_fwhm = (
            fwhm
            if fwhm is not None
            else get_fwhm(
                freq=input_map.frequency,
                arr=input_map.array,
                instrument=input_map.instrument,
            )
        )

        def run_fit(force_center: bool) -> GaussianFitParameters:
            fitter = LmFitGaussian2DFitter(
                forced_source,
                reprojected=reproject_thumb,
                fwhm_guess=beam_fwhm,
                force_center=force_center,
                allowable_center_offset=allowable_center_offset,
                fwhm_prior_width=fwhm_prior_width,
                position_prior_width=position_prior_width,
                log=log,
            )
            fitter.initialize_model()
            return fitter.fit()

        force_center = not fit_centroid or "nearby_source" in source_flags
        fit = run_fit(force_center=force_center)
        if fit.failed and fixed_fit_on_failure and not force_center:
            fit = run_fit(force_center=True)
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
        map_flux = _map_flux_at_fit(input_map, forced_source, fit)
        forced_source.flux = map_flux if map_flux is not None else fit.amplitude
        map_noise = _map_noise_at_fit(input_map, forced_source, fit)
        forced_source.err_flux = (
            map_noise if map_noise is not None else fit.amplitude_err
        )
        forced_source.snr = (
            (forced_source.flux / forced_source.err_flux).to_value(
                u.dimensionless_unscaled
            )
            if (forced_source.err_flux is not None) and (forced_source.flux is not None)
            else None
        )
        if (
            not fit.failed
            and max(fit.fwhm_ra, fit.fwhm_dec) > WIDE_FIT_FWHM_RATIO * beam_fwhm
        ):
            forced_source.flags = [*forced_source.flags, "wide_fit"]

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


def convert_dict_catalog_to_registered_source_objects(
    catalog_sources: dict,
    source_type: str | None = None,
    log: FilteringBoundLogger | None = None,
):
    """
    take each source in the catalog and convert to a list
    of RegisteredSource objects.

    """
    log = log or structlog.get_logger()
    log = log.bind(func_name="convert_dict_catalog_to_registered_source_objects")

    if isinstance(catalog_sources, list):
        log.info(
            "convert_dict_catalog_to_registered_source_objects.catalog_is_list",
            num_sources=len(catalog_sources),
        )
        return catalog_sources
    if not catalog_sources:
        log.warning(
            "convert_dict_catalog_to_registered_source_objects.catalog_is_empty",
            num_sources=0,
        )
        return []

    known_sources = []
    for i in range(len(catalog_sources["name"])):
        source_ra = catalog_sources["RADeg"][i] % 360
        source_dec = catalog_sources["decDeg"][i]
        source_id = uuid.create()
        cs = RegisteredSource(
            ra=source_ra * u.deg,
            dec=source_dec * u.deg,
            flux=catalog_sources["fluxJy"][i] * u.Jy,
            source_id=source_id,
            source_type=source_type,
        )
        if "ra_offset_arcmin" in catalog_sources:
            if catalog_sources["ra_offset_arcmin"][i]:
                cs.err_ra = catalog_sources["err_ra_offset_arcmin"][i] * u.arcmin
                cs.err_dec = catalog_sources["err_dec_offset_arcmin"][i] * u.arcmin

        cs.add_crossmatch(
            CrossMatch(
                source_id=source_id,
                probability=1.0,
                catalog_idx=source_id,
            )
        )
        known_sources.append(cs)
    log.info(
        "convert_dict_catalog_to_registered_source_objects.catalog_converted",
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
