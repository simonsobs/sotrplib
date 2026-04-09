"""
Dependencies for map pointing calculations. Sets MapPointingOffset object.
These operations happen after forced photometry and before source subtraction.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import structlog
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clip, sigma_clipped_stats
from pixell import enmap
from structlog.types import FilteringBoundLogger

if TYPE_CHECKING:
    from sotrplib.sources.sources import RegisteredSource


@dataclass
class PointingModelCoefficients:
    """
    These are used to build a polynomial model of the pointing offsets.
    An array where each element corresponds to a coefficient of the polynomial
    used to model the RA and Dec pointing offsets as a function of position on
    the sky.

    A single entry corresponds to the mean or median value, depending on how the
    constant model is defined.
    """

    model: Literal["mean", "median", "polynomial"]
    poly_order: int | None
    coeffs_ra: u.Quantity
    coeffs_dec: u.Quantity
    residual_ra_rms: u.Quantity | None
    residual_dec_rms: u.Quantity | None


class MapPointingOffset(ABC):
    """
    Base class for map pointing offsets.
    Allow for calculation of offsets from sources, and application of
    the inverse offsets to recover true positions.
    """

    pointing_sources: list["RegisteredSource"] | None = None
    coeffs: PointingModelCoefficients | None = None

    def get_offset(self, pointing_sources: list["RegisteredSource"] | None = None):
        """Calculate the pointing offset based on the provided sources."""
        pass

    def apply_offset_at_position(self, position: SkyCoord) -> SkyCoord:
        """
        Apply inverse offsets to one or more SkyCoord positions.

        Parameters
        ----------
        position : SkyCoord
            Input coordinate(s) to which the offsets will be removed.


        Returns
        -------
        SkyCoord
            New coordinates with the offsets removed.
        """
        if self.coeffs is None:
            return position
        ## returns an array of offsets, but only apply to single position.
        dra = self.ra_model(position.ra, position.dec)
        ddec = self.dec_model(position.ra, position.dec)

        ## if offset > 5arcmin , warn and skip.
        if abs(dra) > 5 * u.arcmin or abs(ddec) > 5 * u.arcmin:
            self.log.warn(
                f"{self.__class__.__name__}.large_offset_warning",
                ra=position.ra,
                dec=position.dec,
                dra=dra,
                ddec=ddec,
            )
            return position

        return SkyCoord(
            ra=position.ra - dra,
            dec=position.dec - ddec,
            frame=position.frame,
        )

    @abstractmethod
    def ra_model(self, ra: u.Quantity, dec: u.Quantity) -> u.Quantity:
        """
        Model function to compute RA offsets at given positions.

        Parameters
        ----------
        ra : u.Quantity
            Array of RA positions in degrees.
        dec : u.Quantity
            Array of Dec positions in degrees.

        Returns
        -------
        u.Quantity
            Array of RA offsets in degrees.
        """
        return

    @abstractmethod
    def dec_model(self, ra: u.Quantity, dec: u.Quantity) -> u.Quantity:
        """
        Model function to compute dec offsets at given positions.

        Parameters
        ----------
        ra : u.Quantity
            Array of RA positions in degrees.
        dec : u.Quantity
            Array of Dec positions in degrees.

        Returns
        -------
        u.Quantity
            Array of dec offsets in degrees.
        """
        return


class EmptyPointingOffset(MapPointingOffset):
    def get_offset(self, pointing_sources: list["RegisteredSource"] | None = None):
        # no need for a deepcopy as apply offset is a noop
        pass

    def dec_model(self, ra: u.Quantity, dec: u.Quantity) -> u.Quantity:
        return 0.0 * u.deg

    def ra_model(self, ra: u.Quantity, dec: u.Quantity) -> u.Quantity:
        return 0.0 * u.deg


class ConstantPointingOffset(MapPointingOffset):
    def __init__(
        self,
        min_snr: float = 5.0,
        min_num: int = 5,
        sigma_clip_level: float = 3.0,
        pointing_sources: list["RegisteredSource"] | None = None,
        avg_method: Literal["mean", "median"] = "mean",
        coeffs: PointingModelCoefficients | None = None,
        log: FilteringBoundLogger | None = None,
    ):
        self.ra_offset = 0.0 * u.deg
        self.dec_offset = 0.0 * u.deg
        self.min_snr = min_snr
        self.min_num = min_num
        self.sigma_clip_level = sigma_clip_level
        self.avg_method = avg_method
        self.pointing_sources = pointing_sources or []
        self.coeffs = (
            PointingModelCoefficients(
                model=self.avg_method,
                coeffs_ra=np.array(0.0) * u.deg,
                coeffs_dec=np.array(0.0) * u.deg,
                poly_order=None,
                residual_dec_rms=None,
                residual_ra_rms=None,
            )
            if coeffs is None
            else coeffs
        )
        self.log = log or structlog.get_logger()

    def get_offset(
        self,
        pointing_sources: list["RegisteredSource"] | None = None,
    ) -> PointingModelCoefficients:
        log = self.log or structlog.get_logger()
        log = log.bind(
            func="pointing.ConstantPointingOffset.get_offset",
            min_snr=self.min_snr,
            min_num=self.min_num,
        )
        # Implementation of median offset calculation
        snr = [
            src.snr
            for src in pointing_sources
            if (src.snr is not None)
            & (src.offset_ra is not None)
            & (src.offset_dec is not None)
        ]
        if len(snr) == 0:
            log.warn("pointing.ConstantPointingOffset.no_valid_sources")
            return PointingModelCoefficients(
                model=self.avg_method,
                coeffs_ra=np.array(0.0) * u.deg,
                coeffs_dec=np.array(0.0) * u.deg,
                poly_order=None,
                residual_dec_rms=None,
                residual_ra_rms=None,
            )

        ra_offsets = [
            src.offset_ra
            for src in pointing_sources
            if (src.snr is not None)
            & (src.offset_ra is not None)
            & (src.offset_dec is not None)
        ]
        dec_offsets = [
            src.offset_dec
            for src in pointing_sources
            if (src.snr is not None)
            & (src.offset_ra is not None)
            & (src.offset_dec is not None)
        ]
        if len(ra_offsets) == 0 or len(dec_offsets) == 0:
            log.warn("pointing.ConstantPointingOffset.no_valid_offsets")
            return PointingModelCoefficients(
                model=self.avg_method,
                coeffs_ra=np.array(0.0) * u.deg,
                coeffs_dec=np.array(0.0) * u.deg,
                poly_order=None,
                residual_dec_rms=None,
                residual_ra_rms=None,
            )

        # Compute median offsets where snr>min_snr
        ra_off_list = u.Quantity(
            [ra for ra, s in zip(ra_offsets, snr) if s >= self.min_snr]
        )

        dec_off_list = u.Quantity(
            [dec for dec, s in zip(dec_offsets, snr) if s >= self.min_snr]
        )

        if len(ra_off_list) < self.min_num or len(dec_off_list) < self.min_num:
            log.warn(
                "pointing.ConstantPointingOffset.not_enough_sources_above_snr",
                n_valid_sources=len(ra_off_list),
            )
            return PointingModelCoefficients(
                model=self.avg_method,
                coeffs_ra=np.array(0.0) * u.deg,
                coeffs_dec=np.array(0.0) * u.deg,
                poly_order=None,
                residual_dec_rms=None,
                residual_ra_rms=None,
            )

        mean_ra, median_ra, std_ra = sigma_clipped_stats(
            ra_off_list,
            sigma=self.sigma_clip_level,
        )
        mean_dec, median_dec, std_dec = sigma_clipped_stats(
            dec_off_list,
            sigma=self.sigma_clip_level,
        )

        ra_offset = median_ra if self.avg_method == "median" else mean_ra
        dec_offset = median_dec if self.avg_method == "median" else mean_dec

        log.info(
            "pointing.ConstantPointingOffset.offsets",
            num_sources=len(ra_off_list),
            min_snr=self.min_snr,
            ra_offset=f"{ra_offset.to_value(u.arcsec):.1f} arcsec",
            dec_offset=f"{dec_offset.to_value(u.arcsec):.1f} arcsec",
            ra_offset_rms=f"{std_ra.to_value(u.arcsec):.1f} arcsec",
            dec_offset_rms=f"{std_dec.to_value(u.arcsec):.1f} arcsec",
        )
        self.coeffs = PointingModelCoefficients(
            model=self.avg_method,
            coeffs_ra=ra_offset,
            coeffs_dec=dec_offset,
            residual_ra_rms=std_ra,
            residual_dec_rms=std_dec,
            poly_order=None,
        )

        return self.coeffs

    def ra_model(self, ra: u.Quantity, dec: u.Quantity) -> u.Quantity:
        return self.coeffs.coeffs_ra

    def dec_model(self, ra: u.Quantity, dec: u.Quantity) -> u.Quantity:
        return self.coeffs.coeffs_dec


class PolynomialPointingOffset(MapPointingOffset):
    def __init__(
        self,
        min_snr: float = 5.0,
        min_num: int = 5,
        sigma_clip_level: float = 3.0,
        pointing_sources: list["RegisteredSource"] | None = None,
        poly_order: int = 3,
        max_snr_weight: float = 100.0,
        coeffs: PointingModelCoefficients | None = None,
        log: FilteringBoundLogger | None = None,
    ):
        self.min_snr = min_snr
        self.min_num = min_num
        self.sigma_clip_level = sigma_clip_level
        self.poly_order = poly_order
        self.max_snr_weight = max_snr_weight
        self.pointing_sources = pointing_sources or []
        self.log = log or structlog.get_logger()
        self.coeffs = (
            PointingModelCoefficients(
                model="polynomial",
                coeffs_ra=np.zeros((self.poly_order + 1) * (self.poly_order + 2) // 2)
                * u.deg,
                coeffs_dec=np.zeros((self.poly_order + 1) * (self.poly_order + 2) // 2)
                * u.deg,
                poly_order=self.poly_order,
                residual_ra_rms=None,
                residual_dec_rms=None,
            )
            if coeffs is None
            else coeffs
        )

    ## Basis terms for 2D polynomial fit
    def _poly_terms(self, x, y):
        terms = []
        for i in range(self.poly_order + 1):
            for j in range(self.poly_order + 1 - i):
                terms.append((x**i) * (y**j))
        return np.vstack(terms).T

    def get_offset(
        self, pointing_sources: list["RegisteredSource"] | None = None
    ) -> PointingModelCoefficients:
        log = self.log.bind(func="PolynomialPointingOffset.get_offset")
        pointing_sources = pointing_sources or self.pointing_sources

        # gather valid records
        valid = [
            s
            for s in pointing_sources
            if s.snr is not None
            and s.offset_ra is not None
            and s.offset_dec is not None
        ]

        if len(valid) == 0:
            log.warn("PolynomialPointingOffset.no_valid_sources")
            n_terms = (self.poly_order + 1) * (self.poly_order + 2) // 2
            return PointingModelCoefficients(
                model="polynomial",
                poly_order=self.poly_order,
                coeffs_ra=np.zeros(n_terms) * u.deg,
                coeffs_dec=np.zeros(n_terms) * u.deg,
                residual_ra_rms=None,
                residual_dec_rms=None,
            )

        ras = np.array([s.ra.to_value(u.deg) for s in valid])
        decs = np.array([s.dec.to_value(u.deg) for s in valid])
        dra = np.array([s.offset_ra.to_value(u.deg) for s in valid])
        ddec = np.array([s.offset_dec.to_value(u.deg) for s in valid])
        snr = np.array([s.snr for s in valid])

        # SNR cut
        mask = snr >= self.min_snr
        ras, decs, dra, ddec, snr = (
            ras[mask],
            decs[mask],
            dra[mask],
            ddec[mask],
            snr[mask],
        )

        if len(ras) < self.min_num:
            log.warn(
                "PolynomialPointingOffset.not_enough_sources_above_snr",
                n_valid_sources=len(ras),
            )
            n_terms = (self.poly_order + 1) * (self.poly_order + 2) // 2
            return PointingModelCoefficients(
                model="polynomial",
                poly_order=self.poly_order,
                coeffs_ra=np.zeros(n_terms) * u.deg,
                coeffs_dec=np.zeros(n_terms) * u.deg,
                residual_ra_rms=None,
                residual_dec_rms=None,
            )

        # sigma clip independently
        dra_clipped = sigma_clip(dra, sigma=self.sigma_clip_level, masked=True)
        ddec_clipped = sigma_clip(ddec, sigma=self.sigma_clip_level, masked=True)
        mask_ra = ~dra_clipped.mask
        mask_dec = ~ddec_clipped.mask

        ## Weighted least squares
        # weights = min(snr, max_snr_weight)
        w = np.minimum(snr, self.max_snr_weight)
        w_sqrt = np.sqrt(w)  # used to weight design matrix and data

        ## RA polynomial fit
        A_ra = self._poly_terms(ras[mask_ra], decs[mask_ra])
        y_ra = dra[mask_ra]
        w_ra = w_sqrt[mask_ra]

        ## Apply weights
        Aw = A_ra * w_ra[:, None]
        yw = y_ra * w_ra
        coeffs_ra, *_ = np.linalg.lstsq(Aw, yw, rcond=None)

        ## Dec polynomial fit
        A_dec = self._poly_terms(ras[mask_dec], decs[mask_dec])
        y_dec = ddec[mask_dec]
        w_dec = w_sqrt[mask_dec]

        Aw = A_dec * w_dec[:, None]
        yw = y_dec * w_dec
        coeffs_dec, *_ = np.linalg.lstsq(Aw, yw, rcond=None)

        residual_ra_rms = np.sqrt(np.mean((A_ra @ coeffs_ra - y_ra) ** 2)) * u.arcsec
        residual_dec_rms = (
            np.sqrt(np.mean((A_dec @ coeffs_dec - y_dec) ** 2)) * u.arcsec
        )

        log.info(
            "PolynomialPointingOffset.weighted_poly_fit",
            num_sources=len(ras),
            poly_order=self.poly_order,
            max_snr_weight=self.max_snr_weight,
            coeffs_ra=coeffs_ra,
            coeffs_dec=coeffs_dec,
            residual_ra_rms=residual_ra_rms,
            residual_dec_rms=residual_dec_rms,
        )

        self.coeffs = PointingModelCoefficients(
            model="polynomial",
            poly_order=self.poly_order,
            coeffs_ra=coeffs_ra * u.deg,
            coeffs_dec=coeffs_dec * u.deg,
            residual_ra_rms=residual_ra_rms,
            residual_dec_rms=residual_dec_rms,
        )
        return self.coeffs

    def model_fn(
        self, ra: u.Quantity, dec: u.Quantity, coeffs: u.Quantity
    ) -> u.Quantity:
        ra = np.atleast_1d(ra.to_value(u.deg))
        dec = np.atleast_1d(dec.to_value(u.deg))
        c = np.atleast_1d(coeffs.to_value(u.deg))
        T = self._poly_terms(ra, dec)
        return (T @ c) * u.deg

    def ra_model(self, ra: u.Quantity, dec: u.Quantity) -> u.Quantity:
        return self.model_fn(ra, dec, self.coeffs.coeffs_ra)

    def dec_model(self, ra: u.Quantity, dec: u.Quantity) -> u.Quantity:
        return self.model_fn(ra, dec, self.coeffs.coeffs_dec)

    def apply_offset_at_position(self, position: SkyCoord) -> SkyCoord:
        ## returns an array of offsets, but only apply to single position.
        dra = self.ra_model(position.ra, position.dec)[0]
        ddec = self.dec_model(position.ra, position.dec)[0]

        ## if offset > 5arcmin , warn and skip.
        if abs(dra) > 5 * u.arcmin or abs(ddec) > 5 * u.arcmin:
            self.log.warn(
                "PolynomialPointingOffset.large_offset_warning",
                ra=position.ra,
                dec=position.dec,
                dra=dra,
                ddec=ddec,
            )
            return position

        return SkyCoord(
            ra=position.ra - dra,
            dec=position.dec - ddec,
            frame=position.frame,
        )


def save_model_maps(
    pointing_model: MapPointingOffset,
    input_map: enmap.ndmap,
    filename_prefix: str,
    log: FilteringBoundLogger | None = None,
):
    """
    Save the pointing offset model as maps for visualization.

    Parameters
    ----------
    pointing_model : MapPointingOffset
        The pointing offset model to be saved.
    input_map : enmap.ndmap
        The input map used to define the shape, WCS and valid pixels of the output pointing offset maps.


    filename_prefix : str
        Prefix for the output map filenames.
    log : FilteringBoundLogger | None
        Logger for logging messages. If None, no logging is performed.
    """
    log = log or structlog.get_logger()
    log = log.bind(func="save_model_maps")

    if isinstance(pointing_model, (ConstantPointingOffset, EmptyPointingOffset)):
        log.warn("save_model_maps.Constant_or_EmptyModel.no_maps_saved")
        return
    pixmap = enmap.pixmap(input_map.shape, wcs=input_map.wcs)

    decmap, ramap = enmap.pix2sky(input_map.shape, input_map.wcs, pixmap)

    ra_offset_map = pointing_model.ra_model(
        (ramap.flatten() * u.rad), (decmap.flatten() * u.rad)
    ).to_value(u.arcsec)
    dec_offset_map = pointing_model.dec_model(
        (ramap.flatten() * u.rad), (decmap.flatten() * u.rad)
    ).to_value(u.arcsec)

    ra_offset_map = enmap.enmap(
        ra_offset_map.reshape(input_map.shape), wcs=input_map.wcs
    )
    dec_offset_map = enmap.enmap(
        dec_offset_map.reshape(input_map.shape), wcs=input_map.wcs
    )

    ra_offset_map[(input_map == 0) | (~np.isfinite(input_map))] = 0.0
    dec_offset_map[(input_map == 0) | (~np.isfinite(input_map))] = 0.0

    enmap.write_map(f"{filename_prefix}_ra_offset_map.fits", ra_offset_map)
    enmap.write_map(f"{filename_prefix}_dec_offset_map.fits", dec_offset_map)

    return
