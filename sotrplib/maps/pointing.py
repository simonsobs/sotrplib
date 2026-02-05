"""
Dependencies for map pointing calculations. Sets MapPointingOffset object.
These operations happen after forced photometry and before source subtraction.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import numpy as np
import structlog
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clip, sigma_clipped_stats
from pixell import enmap
from structlog.types import FilteringBoundLogger

from sotrplib.maps.core import ProcessableMap
from sotrplib.sources.sources import RegisteredSource


@dataclass
class ConstantPointingData:
    ra_offset: u.Quantity
    dec_offset: u.Quantity


@dataclass
class PolynomialPointingData:
    coeffs_ra: np.ndarray
    coeffs_dec: np.ndarray


PointingData = ConstantPointingData | PolynomialPointingData


class MapPointingOffset(ABC):
    """
    Base class for map pointing offsets.
    Allow for calculation of offsets from sources, and application of
    the inverse offsets to recover true positions.
    """

    pointing_sources: list[RegisteredSource] | None = None

    def get_offset(self, pointing_sources: list[RegisteredSource] | None = None):
        """Calculate the pointing offset based on the provided sources."""
        pass

    def apply_offset_at_position(
        self, pos: SkyCoord, data: PointingData | None
    ) -> SkyCoord:
        """
        Apply inverse offsets to one or more SkyCoord positions.

        Parameters
        ----------
        pos : SkyCoord
            Input coordinate(s) to which the offsets will be removed.

        Returns
        -------
        SkyCoord
            New coordinates with the offsets removed.
        """
        ## returns an array of offsets, but only apply to single position.
        dra = self.ra_model(pos.ra, pos.dec, data)
        ddec = self.dec_model(pos.ra, pos.dec, data)

        ## if offset > 5arcmin , warn and skip.
        if abs(dra) > 5 * u.arcmin or abs(ddec) > 5 * u.arcmin:
            self.log.warn(
                f"{self.__class__.__name__}.large_offset_warning",
                ra=pos.ra,
                dec=pos.dec,
                dra=dra,
                ddec=ddec,
            )
            return pos

        return SkyCoord(
            ra=pos.ra - dra,
            dec=pos.dec - ddec,
            frame=pos.frame,
        )

    @abstractmethod
    def ra_model(
        self, ra: u.Quantity, dec: u.Quantity, data: PointingData | None = None
    ) -> u.Quantity:
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
    def dec_model(
        self, ra: u.Quantity, dec: u.Quantity, data: PointingData | None = None
    ) -> u.Quantity:
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
    def get_offset(self, pointing_sources: list[RegisteredSource] | None = None):
        # no need for a deepcopy as apply offset is a noop
        pass

    def dec_model(
        self, ra: u.Quantity, dec: u.Quantity, data: None = None
    ) -> u.Quantity:
        return 0.0 * u.deg

    def ra_model(
        self, ra: u.Quantity, dec: u.Quantity, data: None = None
    ) -> u.Quantity:
        return 0.0 * u.deg


class ConstantPointingOffset(MapPointingOffset):
    def __init__(
        self,
        min_snr: float = 5.0,
        min_num: int = 5,
        sigma_clip_level: float = 3.0,
        pointing_sources: list[RegisteredSource] | None = None,
        avg_method: Literal["mean", "median"] = "mean",
        log: FilteringBoundLogger | None = None,
    ):
        self.ra_offset = 0.0 * u.deg
        self.dec_offset = 0.0 * u.deg
        self.min_snr = min_snr
        self.min_num = min_num
        self.sigma_clip_level = sigma_clip_level
        self.avg_method = avg_method
        self.pointing_sources = pointing_sources or []
        self.log = log or structlog.get_logger()

    def get_offset(
        self,
        pointing_sources: list[RegisteredSource] | None = None,
    ) -> ConstantPointingData:
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
            return ConstantPointingData(ra_offset=0.0 * u.deg, dec_offset=0.0 * u.deg)

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
            return ConstantPointingData(ra_offset=0.0 * u.deg, dec_offset=0.0 * u.deg)

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
            return ConstantPointingData(ra_offset=0.0 * u.deg, dec_offset=0.0 * u.deg)

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

        return ConstantPointingData(ra_offset=ra_offset, dec_offset=dec_offset)

    def ra_model(
        self, ra: u.Quantity, dec: u.Quantity, data: ConstantPointingData
    ) -> u.Quantity:
        return data.ra_offset

    def dec_model(
        self, ra: u.Quantity, dec: u.Quantity, data: ConstantPointingData
    ) -> u.Quantity:
        return data.dec_offset


class PolynomialPointingOffset(MapPointingOffset):
    def __init__(
        self,
        min_snr: float = 5.0,
        min_num: int = 5,
        sigma_clip_level: float = 3.0,
        pointing_sources: list[RegisteredSource] | None = None,
        poly_order: int = 3,
        max_snr_weight: float = 100.0,
        log: FilteringBoundLogger | None = None,
    ):
        self.min_snr = min_snr
        self.min_num = min_num
        self.sigma_clip_level = sigma_clip_level
        self.poly_order = poly_order
        self.max_snr_weight = max_snr_weight
        self.pointing_sources = pointing_sources or []
        self.log = log or structlog.get_logger()

    ## Basis terms for 2D polynomial fit
    def _poly_terms(self, x, y):
        terms = []
        for i in range(self.poly_order + 1):
            for j in range(self.poly_order + 1 - i):
                terms.append((x**i) * (y**j))
        return np.vstack(terms).T

    def get_offset(
        self, pointing_sources: list[RegisteredSource] | None = None
    ) -> PolynomialPointingData:
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
            return PolynomialPointingData(
                coeffs_ra=np.zeros(n_terms),
                coeffs_dec=np.zeros(n_terms),
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
            return PolynomialPointingData(
                coeffs_ra=np.zeros(n_terms),
                coeffs_dec=np.zeros(n_terms),
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

        log.info(
            "PolynomialPointingOffset.weighted_poly_fit",
            num_sources=len(ras),
            poly_order=self.poly_order,
            max_snr_weight=self.max_snr_weight,
            coeffs_ra=coeffs_ra,
            coeffs_dec=coeffs_dec,
        )

        return PolynomialPointingData(
            coeffs_ra=coeffs_ra,
            coeffs_dec=coeffs_dec,
        )

    def model_fn(
        self, ra: u.Quantity, dec: u.Quantity, coeffs: np.ndarray
    ) -> u.Quantity:
        ra = np.atleast_1d(ra.to_value(u.deg))
        dec = np.atleast_1d(dec.to_value(u.deg))
        T = self._poly_terms(ra, dec)
        return (T @ coeffs) * u.deg

    def ra_model(
        self, ra: u.Quantity, dec: u.Quantity, data: PolynomialPointingData
    ) -> u.Quantity:
        return self.model_fn(ra, dec, data.coeffs_ra)

    def dec_model(
        self, ra: u.Quantity, dec: u.Quantity, data: PolynomialPointingData
    ) -> u.Quantity:
        return self.model_fn(ra, dec, data.coeffs_dec)

    def apply_offset_at_position(
        self, pos: SkyCoord, data: PolynomialPointingData
    ) -> SkyCoord:
        ## returns an array of offsets, but only apply to single position.
        dra = self.ra_model(pos.ra, pos.dec, data)[0]
        ddec = self.dec_model(pos.ra, pos.dec, data)[0]

        ## if offset > 5arcmin , warn and skip.
        if abs(dra) > 5 * u.arcmin or abs(ddec) > 5 * u.arcmin:
            self.log.warn(
                "PolynomialPointingOffset.large_offset_warning",
                ra=pos.ra,
                dec=pos.dec,
                dra=dra,
                ddec=ddec,
            )
            return pos

        return SkyCoord(
            ra=pos.ra - dra,
            dec=pos.dec - ddec,
            frame=pos.frame,
        )


def save_model_maps(
    pointing_model: MapPointingOffset,
    pointing_data: PointingData | None,
    input_map: ProcessableMap,
    filename_prefix: str,
    log: FilteringBoundLogger | None = None,
):
    """
    Save the pointing offset model as maps for visualization.

    Parameters
    ----------
    pointing_model : MapPointingOffset
        The pointing offset model to be saved.
    input_map : ProcessableMap
        The input map used to define the shape and WCS of the output maps.
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
    pixmap = enmap.pixmap(input_map.flux.shape, wcs=input_map.flux.wcs)

    decmap, ramap = enmap.pix2sky(input_map.flux.shape, input_map.flux.wcs, pixmap)

    ra_offset_map = pointing_model.ra_model(
        (ramap.flatten() * u.rad).to_value(u.deg),
        (decmap.flatten() * u.rad).to_value(u.deg),
        data=pointing_data,
    ).to_value(u.arcsec)
    dec_offset_map = pointing_model.dec_model(
        (ramap.flatten() * u.rad).to_value(u.deg),
        (decmap.flatten() * u.rad).to_value(u.deg),
        data=pointing_data,
    ).to_value(u.arcsec)

    ra_offset_map = enmap.enmap(
        ra_offset_map.reshape(input_map.flux.shape), wcs=input_map.flux.wcs
    )
    dec_offset_map = enmap.enmap(
        dec_offset_map.reshape(input_map.flux.shape), wcs=input_map.flux.wcs
    )

    ra_offset_map[(input_map.flux == 0) | (~np.isfinite(input_map.flux))] = 0.0
    dec_offset_map[(input_map.flux == 0) | (~np.isfinite(input_map.flux))] = 0.0

    enmap.write_map(f"{filename_prefix}_ra_offset_map.fits", ra_offset_map)
    enmap.write_map(f"{filename_prefix}_dec_offset_map.fits", dec_offset_map)

    return
