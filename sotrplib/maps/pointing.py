"""
Dependencies for map pointing calculations. Sets MapPointingOffset object.
These operations happen after forced photometry and before source subtraction.
"""

from abc import ABC, abstractmethod
from typing import Literal

import structlog
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clipped_stats
from astropydantic import AstroPydanticQuantity
from structlog.types import FilteringBoundLogger

from sotrplib.sources.sources import RegisteredSource


class MapPointingOffset(ABC):
    """
    Base class for map pointing offsets.
    Allow for calculation of offsets from sources, and application of
    the inverse offsets to recover true positions.
    """

    pointing_sources: list[RegisteredSource] | None = None

    @abstractmethod
    def get_offset(
        self, pointing_sources: list[RegisteredSource] | None = None
    ) -> AstroPydanticQuantity[u.deg]:
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
    def get_offset(
        self, pointing_sources: list[RegisteredSource] | None = None
    ) -> AstroPydanticQuantity[u.deg]:
        return (0.0 * u.deg, 0.0 * u.deg)

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
    ) -> AstroPydanticQuantity[u.deg]:
        log = self.log or structlog.get_logger()
        log = log.bind(
            func="pointing.calculate_mean_offsets",
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
            return (self.ra_offset, self.dec_offset)

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
            return (self.ra_offset, self.dec_offset)

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

        return (self.ra_offset, self.dec_offset)

    def apply_offset_at_position(self, pos: SkyCoord) -> SkyCoord:
        return SkyCoord(
            ra=pos.ra - self.ra_offset,
            dec=pos.dec - self.dec_offset,
            frame=pos.frame,
        )
