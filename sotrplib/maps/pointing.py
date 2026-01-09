"""
Dependencies for map pointing calculations. Sets MapPointingOffset object.
These operations happen after forced photometry and before source subtraction.
"""

from abc import ABC, abstractmethod
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


class MapPointingOffset(ABC):
    """
    Base class for map pointing offsets.
    Allow for calculation of offsets from sources, and application of
    the inverse offsets to recover true positions.
    """

    pointing_sources: list[RegisteredSource] | None = None

    @abstractmethod
    def get_offset(self, pointing_sources: list[RegisteredSource] | None = None):
        """Calculate the pointing offset based on the provided sources."""
        return

    @abstractmethod
    def apply_offset_at_position(self, pos: SkyCoord) -> SkyCoord:
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
        return


class EmptyPointingOffset(MapPointingOffset):
    def get_offset(self, pointing_sources: list[RegisteredSource] | None = None):
        return

    def apply_offset_at_position(self, pos: SkyCoord) -> SkyCoord:
        return pos


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
    ):
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
            self.ra_offset = 0.0 * u.deg
            self.dec_offset = 0.0 * u.deg
            return (self.ra_offset, self.dec_offset)

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
            self.ra_offset = 0.0 * u.deg
            self.dec_offset = 0.0 * u.deg
            return

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
            self.ra_offset = 0.0 * u.deg
            self.dec_offset = 0.0 * u.deg
            return

        mean_ra, median_ra, std_ra = sigma_clipped_stats(
            ra_off_list,
            sigma=self.sigma_clip_level,
        )
        mean_dec, median_dec, std_dec = sigma_clipped_stats(
            dec_off_list,
            sigma=self.sigma_clip_level,
        )

        self.ra_offset = median_ra if self.avg_method == "median" else mean_ra
        self.dec_offset = median_dec if self.avg_method == "median" else mean_dec

        self.ra_offset_rms = std_ra
        self.dec_offset_rms = std_dec

        log.info(
            "pointing.ConstantPointingOffset.offsets",
            num_sources=len(ra_off_list),
            min_snr=self.min_snr,
            ra_offset=self.ra_offset,
            dec_offset=self.dec_offset,
            ra_offset_rms=self.ra_offset_rms,
            dec_offset_rms=self.dec_offset_rms,
        )

        return

    def apply_offset_at_position(self, pos: SkyCoord) -> SkyCoord:
        return SkyCoord(
            ra=pos.ra - self.ra_offset,
            dec=pos.dec - self.dec_offset,
            frame=pos.frame,
        )


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

        # identity (no correction)
        self.ra_model = lambda ra, dec: np.zeros_like(np.atleast_1d(ra)) * u.deg
        self.dec_model = lambda ra, dec: np.zeros_like(np.atleast_1d(dec)) * u.deg

    ## Basis terms for 2D polynomial fit
    def _poly_terms(self, x, y):
        terms = []
        for i in range(self.poly_order + 1):
            for j in range(self.poly_order + 1 - i):
                terms.append((x**i) * (y**j))
        return np.vstack(terms).T

    def get_offset(
        self,
        pointing_sources: list[RegisteredSource] | None = None,
    ):
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
            return

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
            return

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

        ## Store ra,dec poly fit models
        def ra_model_fn(ra, dec, c=coeffs_ra):
            ra = np.atleast_1d(ra)
            dec = np.atleast_1d(dec)
            T = self._poly_terms(ra, dec)
            return (T @ c) * u.deg

        def dec_model_fn(ra, dec, c=coeffs_dec):
            ra = np.atleast_1d(ra)
            dec = np.atleast_1d(dec)
            T = self._poly_terms(ra, dec)
            return (T @ c) * u.deg

        self.ra_model = ra_model_fn
        self.dec_model = dec_model_fn

        log.info(
            "PolynomialPointingOffset.weighted_poly_fit",
            num_sources=len(ras),
            poly_order=self.poly_order,
            max_snr_weight=self.max_snr_weight,
            coeffs_ra=coeffs_ra,
            coeffs_dec=coeffs_dec,
        )

        return

    def apply_offset_at_position(self, pos: SkyCoord) -> SkyCoord:
        ra = pos.ra.to_value(u.deg)
        dec = pos.dec.to_value(u.deg)
        ## returns an array of offsets, but only apply to single position.
        dra = self.ra_model(ra, dec)[0]
        ddec = self.dec_model(ra, dec)[0]

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
    pointing_model: MapPointingOffset, input_map: ProcessableMap, filename_prefix: str
):
    """
    Save the pointing offset model as maps for visualization.

    Parameters
    ----------
    filename_prefix : str
        Prefix for the output map filenames.
    """
    pixmap = enmap.pixmap(input_map.flux.shape, wcs=input_map.flux.wcs)

    decmap, ramap = enmap.pix2sky(input_map.flux.shape, input_map.flux.wcs, pixmap)

    ra_offset_map = pointing_model.ra_model(
        (ramap.flatten() * u.rad).to_value(u.deg),
        (decmap.flatten() * u.rad).to_value(u.deg),
    ).to_value(u.arcsec)
    dec_offset_map = pointing_model.dec_model(
        (ramap.flatten() * u.rad).to_value(u.deg),
        (decmap.flatten() * u.rad).to_value(u.deg),
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
