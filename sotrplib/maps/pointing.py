"""
Dependencies for map pointing calculations. Sets MapPointingOffset object.
These operations happen after forced photometry and before source subtraction.
"""

from abc import ABC, abstractmethod

import numpy as np
import structlog
from astropy import units as u
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


class LinearMedianPointingOffset(MapPointingOffset):
    def __init__(
        self,
        min_snr: float = 5.0,
        min_num: int = 5,
        pointing_sources: list[RegisteredSource] | None = None,
        log: FilteringBoundLogger | None = None,
    ):
        self.ra_offset = 0.0 * u.deg
        self.dec_offset = 0.0 * u.deg
        self.min_snr = min_snr
        self.min_num = min_num
        self.pointing_sources = pointing_sources or []
        self.log = log or structlog.get_logger()

    def get_offset(
        self,
        pointing_sources: list[RegisteredSource] | None = None,
    ) -> AstroPydanticQuantity[u.deg]:
        log = self.log or structlog.get_logger()
        log.bind(
            func="pointing.calculate_median_offsets",
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
        # Compute median offsets where snr>min_snr
        self.ra_offset = (
            np.nanmedian(
                [
                    ra.to_value(u.arcmin)
                    for ra, s in zip(ra_offsets, snr)
                    if s >= self.min_snr
                ]
            )
            * u.arcmin
        )
        self.dec_offset = (
            np.nanmedian(
                [
                    dec.to_value(u.arcmin)
                    for dec, s in zip(dec_offsets, snr)
                    if s >= self.min_snr
                ]
            )
            * u.arcmin
        )
        # Check if enough sources
        num_sources = sum(1 for s in snr if s >= self.min_snr)
        if num_sources < self.min_num:
            log.warn("pointing.calculate_median_offsets.notenough_sources")
            self.ra_offset = 0.0 * u.deg
            self.dec_offset = 0.0 * u.deg
        log.info(
            "pointing.calculate_median_offsets.offsets",
            num_sources=num_sources,
            min_snr=self.min_snr,
            ra_offset=self.ra_offset,
            dec_offset=self.dec_offset,
        )

        return (self.ra_offset, self.dec_offset)

    def apply_offset(self, input_map: ProcessableMap) -> ProcessableMap:
        return shift_map_radec(input_map, self.ra_offset, self.dec_offset)
