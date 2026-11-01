"""
Robust pointing-residual models for depth-1 maps.

The models in this module find the pointing residual of a map from the
measured offsets of bright catalogue sources. They are made to stay correct
when:

- some fits converge on noise and give offsets of several arcminutes;
- parts of the map have no sources;
- the map crosses RA = 0.

The procedure has four steps:

1. Gate. Remove sources with a failed fit, a low SNR, a non-finite offset, or
   an offset larger than ``max_offset``.
2. Weight. Give each source a positional uncertainty from the beam size and
   the SNR (Condon 1997): ``sigma = 0.6 * FWHM / SNR``, added in quadrature to
   a systematic floor.
3. Clip. Remove outliers one at a time. Use the 2D normalised residual of each
   source with respect to the model that the other sources give (leave-one-out).
4. Fit. Use a constant offset (inverse-variance weighted mean), or a constant
   plus a smooth Gaussian-process (GP) correction. The GP correction goes to
   zero far from the sources. Thus, the model goes back to the constant offset
   in regions without sources and cannot become large.

The GP method is the same as the method that DES uses for atmospheric
astrometric residuals (Fortino et al. 2021, AJ 162, 106). It is also related
to the smoothed radial-basis-function warp of GLEAM (fits_warp, Hurley-Walker
& Hancock 2018).
"""

from typing import TYPE_CHECKING, Literal

import numpy as np
import structlog
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropydantic import AstroPydanticQuantity
from mapcat.pointing.base import PointingModelProtocol, PointingModelStats
from mapcat.pointing.const import ConstantPointingModel
from structlog.types import FilteringBoundLogger

if TYPE_CHECKING:
    from sotrplib.sources.sources import MeasuredSource

# Condon (1997): sigma_x = sqrt(2) * sigma_beam / SNR = 0.6 * FWHM / SNR.
CONDON_FACTOR = np.sqrt(2.0) / np.sqrt(8.0 * np.log(2.0))


def _unit_vectors(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    """Return the 3D unit vectors of sky positions. RA wrap has no effect."""
    ra = np.deg2rad(np.asarray(ra_deg, dtype=float))
    dec = np.deg2rad(np.asarray(dec_deg, dtype=float))
    return np.stack(
        [np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], axis=-1
    )


def _angular_distance_deg(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Return the angular distance in degrees between two sets of unit vectors."""
    chord = np.linalg.norm(v1[:, None, :] - v2[None, :, :], axis=-1)
    return np.rad2deg(2.0 * np.arcsin(np.clip(chord / 2.0, 0.0, 1.0)))


class GaussianProcessPointingModel(PointingModelProtocol):
    """
    Pointing model: a constant offset plus a smooth GP correction.

    The kernel is ``tau**2 * exp(-d**2 / (2 * length_scale**2))``, where ``d``
    is the angular distance. The model keeps the support positions and the
    GP weights (``alpha``) of each axis. Far from all support positions, the
    model gives the constant offset.

    The offsets have the same sign convention as ``ConstantPointingModel``:
    ``predict`` subtracts the offset from the input position.
    """

    model_type: Literal["gaussian_process"] = "gaussian_process"

    ra_offset: AstroPydanticQuantity[u.deg]
    dec_offset: AstroPydanticQuantity[u.deg]
    length_scale: AstroPydanticQuantity[u.deg]
    amplitude: AstroPydanticQuantity[u.deg]
    max_correction: AstroPydanticQuantity[u.deg]
    support_ra: list[float]
    support_dec: list[float]
    alpha_ra: list[float]
    alpha_dec: list[float]

    def offsets_at(self, pos: SkyCoord) -> tuple[u.Quantity, u.Quantity]:
        """Return the RA and Dec offsets of the model at the positions."""
        ra = np.atleast_1d(pos.ra.to_value(u.deg))
        dec = np.atleast_1d(pos.dec.to_value(u.deg))
        if len(self.support_ra) == 0:
            corr_ra = np.zeros_like(ra)
            corr_dec = np.zeros_like(ra)
        else:
            d = _angular_distance_deg(
                _unit_vectors(ra, dec),
                _unit_vectors(np.array(self.support_ra), np.array(self.support_dec)),
            )
            ell = self.length_scale.to_value(u.deg)
            tau2 = self.amplitude.to_value(u.deg) ** 2
            k = tau2 * np.exp(-0.5 * (d / ell) ** 2)
            corr_ra = k @ np.asarray(self.alpha_ra)
            corr_dec = k @ np.asarray(self.alpha_dec)
            cmax = self.max_correction.to_value(u.deg)
            corr_ra = np.clip(corr_ra, -cmax, cmax)
            corr_dec = np.clip(corr_dec, -cmax, cmax)
        off_ra = (self.ra_offset.to_value(u.deg) + corr_ra) * u.deg
        off_dec = (self.dec_offset.to_value(u.deg) + corr_dec) * u.deg
        if pos.isscalar:
            return off_ra[0], off_dec[0]
        return off_ra, off_dec

    def predict(self, pos: SkyCoord) -> SkyCoord:
        off_ra, off_dec = self.offsets_at(pos)
        return SkyCoord(ra=pos.ra - off_ra, dec=pos.dec - off_dec, frame=pos.frame)

    def as_constant(self) -> ConstantPointingModel:
        """Return the constant part of the model."""
        return ConstantPointingModel(
            ra_offset=self.ra_offset, dec_offset=self.dec_offset
        )


def _gp_fit(
    v: np.ndarray,
    y: np.ndarray,
    sig: np.ndarray,
    tau: float,
    ell: float,
    const_prior: float,
):
    """
    Fit one axis with a GP and return the values that the clipping and the
    model need.

    The kernel is ``const_prior**2 + tau**2 * SE(ell)``. The constant term
    marginalises the mean offset. If ``tau`` is 0, the fit is the
    inverse-variance weighted mean.

    Returns
    -------
    mean : float
        The generalised-least-squares constant offset.
    alpha : np.ndarray
        The GP weights of the SE part, about ``mean``.
    loo_resid : np.ndarray
        The leave-one-out residuals.
    loo_var : np.ndarray
        The leave-one-out predictive variance plus the noise variance.
    """
    n = len(y)
    d = _angular_distance_deg(v, v)
    kse = (tau**2) * np.exp(-0.5 * (d / ell) ** 2) if tau > 0 else np.zeros((n, n))
    kfull = const_prior**2 + kse + np.diag(sig**2)
    kinv = np.linalg.inv(kfull)
    a_full = kinv @ y
    diag = np.diag(kinv)
    loo_resid = a_full / diag
    loo_var = 1.0 / diag

    # Split the constant part from the SE part. The constant is the GLS mean
    # with the covariance kse + noise.
    c = kse + np.diag(sig**2)
    cinv = np.linalg.inv(c)
    one = np.ones(n)
    mean = float(one @ cinv @ y / (one @ cinv @ one))
    alpha = cinv @ (y - mean)
    return mean, alpha, loo_resid, loo_var


class RobustPointingOffset:
    """
    Robust pointing-residual estimator.

    Parameters
    ----------
    method : {"constant", "gp"}
        The model to fit. "constant" gives a ``ConstantPointingModel``. "gp"
        gives a ``GaussianProcessPointingModel``.
    min_snr : float
        Minimum source SNR.
    min_num : int
        Minimum number of sources after the gate and the clip. With fewer
        sources, the estimator gives a zero offset.
    max_offset : Quantity
        Sources with a larger total offset are removed before the fit.
    max_model_offset : Quantity
        Largest permitted constant offset of the model. If the fit gives a
        larger offset, the estimator gives a zero offset and a warning.
    clip_sigma : float
        Clip threshold, in units of the 2D normalised leave-one-out residual.
    sys_floor : Quantity
        Systematic positional uncertainty per axis, added in quadrature.
    fwhm : Quantity | None
        Beam FWHM for the uncertainty. If None, use the median fitted FWHM of
        the sources with SNR >= ``fwhm_snr``.
    fwhm_snr : float
        Minimum SNR of the sources that give the median FWHM.
    length_scale : Quantity
        GP correlation length.
    amplitude : Quantity
        GP rms amplitude per axis.
    max_correction : Quantity
        Largest permitted GP correction about the constant offset.
    """

    def __init__(
        self,
        method: Literal["constant", "gp"] = "gp",
        min_snr: float = 5.0,
        min_num: int = 3,
        max_offset: u.Quantity = 5.0 * u.arcmin,
        max_model_offset: u.Quantity = 5.0 * u.arcmin,
        clip_sigma: float = 4.0,
        sys_floor: u.Quantity = 2.0 * u.arcsec,
        fwhm: u.Quantity | None = None,
        fwhm_snr: float = 10.0,
        length_scale: u.Quantity = 5.0 * u.deg,
        amplitude: u.Quantity = 2.5 * u.arcsec,
        max_correction: u.Quantity = 15.0 * u.arcsec,
        log: FilteringBoundLogger | None = None,
    ):
        self.method = method
        self.min_snr = min_snr
        self.min_num = min_num
        self.max_offset = max_offset
        self.max_model_offset = max_model_offset
        self.clip_sigma = clip_sigma
        self.sys_floor = sys_floor
        self.fwhm = fwhm
        self.fwhm_snr = fwhm_snr
        self.length_scale = length_scale
        self.amplitude = amplitude
        self.max_correction = max_correction
        self.pointing_sources: list = []
        self.pointing_model = None
        self.pointing_stats = None
        self.log = log or structlog.get_logger()

    # --- Array interface (also used by the offline tests) ---------------------

    def fit_arrays(
        self,
        ra: np.ndarray,
        dec: np.ndarray,
        dra: np.ndarray,
        ddec: np.ndarray,
        snr: np.ndarray,
        fwhm: np.ndarray | None = None,
    ) -> tuple[
        ConstantPointingModel | GaussianProcessPointingModel,
        PointingModelStats,
        np.ndarray,
    ]:
        """
        Fit the model to arrays.

        Parameters
        ----------
        ra, dec : np.ndarray
            Source positions in degrees.
        dra, ddec : np.ndarray
            Offsets in arcsec, with the sotrp sign convention.
        snr : np.ndarray
            Source SNR.
        fwhm : np.ndarray | None
            Fitted FWHM of each source, in arcsec.

        Returns
        -------
        model, stats, used : the model, its statistics and a mask of the
        sources that the fit used.
        """
        log = self.log.bind(func="RobustPointingOffset.fit_arrays", method=self.method)
        ra, dec, dra, ddec, snr = (
            np.asarray(x, dtype=float) for x in (ra, dec, dra, ddec, snr)
        )
        n_in = len(ra)
        zero = ConstantPointingModel(ra_offset=0.0 * u.deg, dec_offset=0.0 * u.deg)

        max_off = self.max_offset.to_value(u.arcsec)
        ok = (
            np.isfinite(ra)
            & np.isfinite(dec)
            & np.isfinite(dra)
            & np.isfinite(ddec)
            & np.isfinite(snr)
            & (snr >= self.min_snr)
            & (np.hypot(dra, ddec) <= max_off)
        )

        # Beam FWHM for the uncertainties.
        if self.fwhm is not None:
            beam = self.fwhm.to_value(u.arcsec)
        else:
            beam = np.nan
            if fwhm is not None:
                fw = np.asarray(fwhm, dtype=float)
                good = ok & np.isfinite(fw) & (snr >= self.fwhm_snr)
                if good.sum() >= 3:
                    beam = float(np.median(fw[good]))
            if not np.isfinite(beam):
                log.warning("RobustPointingOffset.no_fwhm_estimate")
                beam = 120.0
        sig = np.hypot(
            CONDON_FACTOR * beam / np.maximum(snr, 1e-3),
            self.sys_floor.to_value(u.arcsec),
        )

        # Pre-clip with respect to the weighted median. This removes gross
        # outliers before the leave-one-out step, which is not robust to many
        # gross outliers.
        if ok.sum() >= self.min_num:
            med_ra = np.median(dra[ok])
            med_dec = np.median(ddec[ok])
            z = np.hypot((dra - med_ra) / sig, (ddec - med_dec) / sig)
            ok &= z <= 3.0 * self.clip_sigma

        if ok.sum() < self.min_num:
            log.warning(
                "RobustPointingOffset.not_enough_sources",
                n_valid=int(ok.sum()),
                n_in=n_in,
            )
            return zero, PointingModelStats(n_sources=int(ok.sum())), ok

        v_all = _unit_vectors(ra, dec)
        tau = self.amplitude.to_value(u.arcsec) if self.method == "gp" else 0.0
        ell = self.length_scale.to_value(u.deg)
        const_prior = 2.0 * max_off

        # Remove the worst outlier, then fit again, until no source is above
        # the clip threshold.
        while True:
            idx = np.flatnonzero(ok)
            fits = [
                _gp_fit(v_all[idx], y[idx], sig[idx], tau, ell, const_prior)
                for y in (dra, ddec)
            ]
            z2 = sum(f[2] ** 2 / f[3] for f in fits)
            worst = int(np.argmax(z2))
            if z2[worst] <= self.clip_sigma**2 or len(idx) <= self.min_num:
                break
            ok[idx[worst]] = False

        (m_ra, a_ra, r_ra, _), (m_dec, a_dec, r_dec, _) = fits
        n_used = len(idx)
        if np.hypot(m_ra, m_dec) > self.max_model_offset.to_value(u.arcsec):
            log.warning(
                "RobustPointingOffset.model_offset_above_limit",
                ra_offset=f"{m_ra:.1f} arcsec",
                dec_offset=f"{m_dec:.1f} arcsec",
                limit=f"{self.max_model_offset.to_value(u.arcsec):.0f} arcsec",
            )
            return zero, PointingModelStats(n_sources=n_used), ok
        w = 1.0 / sig[idx] ** 2
        err_ra = np.sqrt(1.0 / w.sum())

        if self.method == "gp":
            model = GaussianProcessPointingModel(
                ra_offset=(m_ra * u.arcsec).to(u.deg),
                dec_offset=(m_dec * u.arcsec).to(u.deg),
                length_scale=self.length_scale.to(u.deg),
                amplitude=self.amplitude.to(u.deg),
                max_correction=self.max_correction.to(u.deg),
                support_ra=ra[idx].tolist(),
                support_dec=dec[idx].tolist(),
                alpha_ra=(a_ra / 3600.0**2).tolist(),
                alpha_dec=(a_dec / 3600.0**2).tolist(),
            )
        else:
            model = ConstantPointingModel(
                ra_offset=(m_ra * u.arcsec).to(u.deg),
                dec_offset=(m_dec * u.arcsec).to(u.deg),
            )

        stats = PointingModelStats(
            mean_ra_offset=(m_ra * u.arcsec).to(u.deg),
            mean_dec_offset=(m_dec * u.arcsec).to(u.deg),
            stddev_ra_offset=(np.std(r_ra) * u.arcsec).to(u.deg),
            stddev_dec_offset=(np.std(r_dec) * u.arcsec).to(u.deg),
            n_sources=n_used,
        )
        log.info(
            "RobustPointingOffset.fit",
            n_in=n_in,
            n_used=n_used,
            beam_fwhm=f"{beam:.1f} arcsec",
            ra_offset=f"{m_ra:.1f} arcsec",
            dec_offset=f"{m_dec:.1f} arcsec",
            offset_err=f"{err_ra:.1f} arcsec",
            loo_rms_ra=f"{np.std(r_ra):.1f} arcsec",
            loo_rms_dec=f"{np.std(r_dec):.1f} arcsec",
        )
        return model, stats, ok

    # --- MapPointingOffset interface ----------------------------------------------

    def build_model(self, pointing_sources: list["MeasuredSource"] | None = None):
        """Fit the model to the measured pointing sources."""
        srcs = [
            s
            for s in (pointing_sources or self.pointing_sources)
            if s.snr is not None
            and s.offset_ra is not None
            and s.offset_dec is not None
            and not getattr(s, "fit_failed", False)
        ]

        def q(s, name, unit):
            v = getattr(s, name, None)
            return np.nan if v is None else v.to_value(unit)

        fw = np.array(
            [
                np.nanmean([q(s, "fwhm_ra", u.arcsec), q(s, "fwhm_dec", u.arcsec)])
                if (s.fwhm_ra is not None or s.fwhm_dec is not None)
                else np.nan
                for s in srcs
            ]
        )
        model, stats, _ = self.fit_arrays(
            ra=np.array([q(s, "ra", u.deg) for s in srcs]),
            dec=np.array([q(s, "dec", u.deg) for s in srcs]),
            dra=np.array([q(s, "offset_ra", u.arcsec) for s in srcs]),
            ddec=np.array([q(s, "offset_dec", u.arcsec) for s in srcs]),
            snr=np.array([s.snr for s in srcs], dtype=float),
            fwhm=fw,
        )
        self.pointing_model = model
        self.pointing_stats = stats
        return model, stats

    def apply_offset_at_position(self, pos: SkyCoord) -> SkyCoord:
        if self.pointing_model is None:
            return pos
        return self.pointing_model.predict(pos)
