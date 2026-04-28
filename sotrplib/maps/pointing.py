"""
Dependencies for map pointing calculations. Sets MapPointingOffset object.
These operations happen after forced photometry and before source subtraction.
"""

from abc import ABC
from typing import TYPE_CHECKING, Literal

import numpy as np
import structlog
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clip, sigma_clipped_stats
from mapcat.pointing.base import PointingModelStats
from mapcat.pointing.const import ConstantPointingModel
from mapcat.pointing.poly import PolynomialPointingModel
from pixell import enmap
from structlog.types import FilteringBoundLogger

if TYPE_CHECKING:
    from sotrplib.maps.core import ProcessableMap

from sotrplib.sources.sources import RegisteredSource

PointingModel = ConstantPointingModel | PolynomialPointingModel


class MapPointingOffset(ABC):
    """
    Base class for map pointing offsets.
    Allow for calculation of offsets from sources, and application of
    the inverse offsets to recover true positions.
    """

    pointing_sources: list[RegisteredSource] | None = None
    pointing_model: PointingModel | None = None

    def calculate_model(self, pointing_sources: list[RegisteredSource] | None = None):
        """Calculate the pointing offset model based on the provided sources."""
        pass

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
        if self.pointing_model is None:
            return pos

        return self.pointing_model.predict(pos)


class EmptyPointingOffset(MapPointingOffset):
    def calculate_model(self, pointing_sources: list[RegisteredSource] | None = None):
        # no need for a deepcopy as apply offset is a noop
        pass

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
        self.pointing_model = None
        self.pointing_stats = None
        self.log = log or structlog.get_logger()

    def calculate_model(
        self, pointing_sources: list[RegisteredSource] | None = None
    ) -> tuple[ConstantPointingModel, PointingModelStats]:
        log = self.log or structlog.get_logger()
        log = log.bind(
            func="pointing.ConstantPointingOffset.get_offset",
            min_snr=self.min_snr,
            min_num=self.min_num,
        )

        model = ConstantPointingModel(ra_offset=0.0 * u.deg, dec_offset=0.0 * u.deg)
        stats = PointingModelStats()

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
            return model, stats

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
            return model, stats

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
            return model, stats

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

        model = ConstantPointingModel(ra_offset=ra_offset, dec_offset=dec_offset)
        stats = PointingModelStats(
            mean_ra_offset=mean_ra,
            mean_dec_offset=mean_dec,
            stddev_ra_offset=std_ra,
            stddev_dec_offset=std_dec,
            n_sources=len(ra_off_list),
        )
        self.pointing_model = model
        self.pointing_stats = stats
        return model, stats

    def apply_offset_at_position(self, pos: SkyCoord) -> SkyCoord:
        return super().apply_offset_at_position(pos)


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
        self.pointing_model = None
        self.pointing_stats = None
        self.log = log or structlog.get_logger()

    def calculate_model(
        self, pointing_sources: list[RegisteredSource] | None = None
    ) -> PolynomialPointingModel:
        log = self.log.bind(func="PolynomialPointingOffset.calculate_model")
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
            log.warn("PolynomialPointingOffset.calculate_model.no_valid_sources")
            self.pointing_model = None
            return None

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
                "PolynomialPointingOffset.calculate_model.not_enough_sources_above_snr",
                n_valid_sources=len(ras),
            )
            return None

        # sigma clip independently
        dra_clipped = sigma_clip(dra, sigma=self.sigma_clip_level, masked=True)
        ddec_clipped = sigma_clip(ddec, sigma=self.sigma_clip_level, masked=True)
        mask_ra = ~dra_clipped.mask
        mask_dec = ~ddec_clipped.mask

        masked = mask_ra | mask_dec
        self.pointing_model = PolynomialPointingModel(poly_order=self.poly_order)
        meas_pos = SkyCoord(
            ra=ras[~masked] * u.deg, dec=decs[~masked] * u.deg, frame="icrs"
        )
        exp_pos = SkyCoord(
            ra=(ras[~masked] - dra[~masked]) * u.deg,
            dec=(decs[~masked] - ddec[~masked]) * u.deg,
            frame="icrs",
        )
        self.pointing_model.calculate_model(meas_pos, exp_pos, weights=snr[~masked])

        log.info(
            "PolynomialPointingOffset.calculate_model.weighted_poly_fit",
            num_sources=len(ras),
            poly_order=self.poly_order,
            max_snr_weight=self.max_snr_weight,
            coeffs_ra=self.pointing_model.ra_model_coefficients.coeffs,
            coeffs_dec=self.pointing_model.dec_model_coefficients.coeffs,
        )

        self.pointing_stats = self.pointing_model.calculate_statistics(meas_pos)

        return self.pointing_model, self.pointing_stats

    def apply_offset_at_position(self, pos: SkyCoord) -> SkyCoord:
        return super().apply_offset_at_position(pos)


def save_model_maps(
    pointing_model: PointingModel,
    input_map: "ProcessableMap",
    filename_prefix: str,
    log: FilteringBoundLogger | None = None,
):
    """
    Save the pointing offset model as maps for visualization.

    Parameters
    ----------
    pointing_model : PointingModel
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
        (ramap.flatten() * u.rad), (decmap.flatten() * u.rad)
    ).to_value(u.arcsec)
    dec_offset_map = pointing_model.dec_model(
        (ramap.flatten() * u.rad), (decmap.flatten() * u.rad)
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
