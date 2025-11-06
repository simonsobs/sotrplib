"""
Dependencies for map pointing calculations. Sets MapPointingOffset object.
These operations happen after forced photometry and before source subtraction.
"""

from abc import ABC, abstractmethod
from typing import Literal

import structlog
from astropy import units as u
from astropy.stats import sigma_clipped_stats
from astropydantic import AstroPydanticQuantity
from structlog.types import FilteringBoundLogger

from sotrplib.maps.core import ProcessableMap
from sotrplib.maps.maps import shift_map_radec
from sotrplib.sources.sources import RegisteredSource


class MapPointingOffset(ABC):
    pointing_sources: list[RegisteredSource] | None = None

    @abstractmethod
    def get_offset(
        self, pointing_sources: list[RegisteredSource] | None = None
    ) -> AstroPydanticQuantity[u.deg]:
        return

    @abstractmethod
    def apply_offset(self, input_map: ProcessableMap) -> ProcessableMap:
        return


class EmptyPointingOffset(MapPointingOffset):
    def get_offset(
        self, pointing_sources: list[RegisteredSource] | None = None
    ) -> AstroPydanticQuantity[u.deg]:
        return (0.0 * u.deg, 0.0 * u.deg)

    def apply_offset(self, input_map: ProcessableMap) -> ProcessableMap:
        return input_map


class LinearPointingOffset(MapPointingOffset):
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
            log.warn("pointing.LinearPointingOffsets.no_valid_sources")
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
            log.warn("pointing.LinearPointingOffsets.no_valid_offsets")
            self.ra_offset = 0.0 * u.deg
            self.dec_offset = 0.0 * u.deg
            return (self.ra_offset, self.dec_offset)

        # Compute median offsets where snr>min_snr
        ra_off_list = [
            ra.to_value(u.arcmin) for ra, s in zip(ra_offsets, snr) if s >= self.min_snr
        ]
        dec_off_list = [
            dec.to_value(u.arcmin)
            for dec, s in zip(dec_offsets, snr)
            if s >= self.min_snr
        ]

        if len(ra_off_list) < self.min_num or len(dec_off_list) < self.min_num:
            log.warn(
                "pointing.LinearPointingOffsets.not_enough_sources_above_snr",
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

        if self.avg_method == "median":
            self.ra_offset = median_ra * u.arcmin
            self.dec_offset = median_dec * u.arcmin

        else:
            self.ra_offset = mean_ra * u.arcmin
            self.dec_offset = mean_dec * u.arcmin

        self.ra_offset_rms = std_ra * u.arcmin
        self.dec_offset_rms = std_dec * u.arcmin

        log.info(
            "pointing.LinearPointingOffsets.offsets",
            num_sources=len(ra_off_list),
            min_snr=self.min_snr,
            ra_offset=self.ra_offset,
            dec_offset=self.dec_offset,
            ra_offset_rms=self.ra_offset_rms,
            dec_offset_rms=self.dec_offset_rms,
        )

        return (self.ra_offset, self.dec_offset)

    def apply_offset(self, input_map: ProcessableMap) -> ProcessableMap:
        return shift_map_radec(input_map, self.ra_offset, self.dec_offset)
