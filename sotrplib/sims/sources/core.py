"""
Source simulations
"""

import numpy as np
from astropy.coordinates import SkyCoord
from pixell import enmap

from sotrplib.maps.core import ProcessableMap


class ProcessableMapWithSimulatedSources(ProcessableMap):
    """
    A ProcessableMap that has its sources simulated.
    """

    def __init__(
        self,
        flux: enmap.ndmap,
        snr: enmap.ndmap,
        time: enmap.ndmap,
        original_map: ProcessableMap,
    ):
        self.flux = flux
        self.snr = snr

        self.time_first = np.minimum(time, original_map.time_first)
        self.time_mean = time
        self.time_last = np.maximum(time, original_map.time_first)

        self.original_map = original_map

        self.observation_length = original_map.observation_length
        self.observation_start = original_map.observation_start
        self.observation_end = original_map.observation_end
        self.observation_time = original_map.observation_time

        self.flux_units = original_map.flux_units
        self.frequency = original_map.frequency
        self.instrument = original_map.instrument
        self.array = original_map.array
        self.finalized = original_map.finalized
        self.map_resolution = original_map.map_resolution
        self.box = original_map.box

        return

    def build(self):
        return

    def _compute_hits(self):
        return (abs(self.flux) > 0).astype(np.int32)

    def _compute_valid_pixel_mask(self):
        bool_map = (abs(self.flux) > 0).astype(np.int32) & (np.isfinite(self.flux))
        if self.mask is not None:
            bool_map = bool_map & (self.mask > 0)
        return bool_map

    def filter_sources(self, source_positions: SkyCoord):
        bool_map = (abs(self.flux) > 0).astype(np.int32) & (np.isfinite(self.flux))
        if self.mask is not None:
            bool_map *= self.mask

        return self._core_filter_sources(source_positions, bool_map)

    def get_pixel_times(self, pix):
        return super().get_pixel_times(pix)

    def apply_mask(self):
        return super().apply_mask()

    def finalize(self):
        super().finalize()
        return
