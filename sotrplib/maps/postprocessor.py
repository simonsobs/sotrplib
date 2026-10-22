"""
Dependencies for post-processing (e.g. flat-fielding). Affects
maps in-place. These operations happen after 'finalize', i.e. once
the snr and flux arrays already exist.
"""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import structlog
from astropy import units as u
from astropydantic import AstroPydanticQuantity
from pixell import enmap
from structlog.types import FilteringBoundLogger

from sotrplib.maps.core import ProcessableMap
from sotrplib.maps.maps import (
    flat_field_using_photutils,
    flat_field_using_pixell,
    snr_rms_tile_mask,
)
from sotrplib.maps.masks import mask_dustgal


class MapPostprocessor(ABC):
    @abstractmethod
    def postprocess(self, input_map: ProcessableMap) -> ProcessableMap:
        """
        Provides postprocessing for the input map
        """
        return input_map


class GalaxyMask(MapPostprocessor):
    mask_path: Path
    invert: bool = False
    log: FilteringBoundLogger | None = None

    def __init__(
        self,
        mask_path: Path,
        invert: bool = False,
        log: FilteringBoundLogger | None = None,
    ):
        self.mask_path = mask_path
        self.invert = invert
        self.log = log or structlog.get_logger()

    def postprocess(self, input_map: ProcessableMap) -> ProcessableMap:
        galaxy_mask = mask_dustgal(
            imap=input_map.flux,
            galmask=enmap.read_map(
                fname=str(self.mask_path),
                box=input_map.bbox,
            ),
        )
        if self.invert:
            galaxy_mask = 1 - galaxy_mask

        input_map.flux *= galaxy_mask
        input_map.snr *= galaxy_mask

        return input_map


class SnrRmsMask(MapPostprocessor):
    """
    Mask the tiles of the map where the robust rms of the snr is outside
    [min_rms, max_rms]. In these tiles the noise model of the matched filter
    is not correct (e.g. scan stripes), thus the snr is not reliable.

    The postprocessor sets the mask of the map to 0 in the bad tiles and sets
    the snr and the flux to 0 there (as apply_mask). Thus, forced photometry
    and the blind search do not use these tiles.

    Put this postprocessor before the flatfield. The flatfield divides the snr
    by the local rms, thus after the flatfield the rms is approximately 1 in
    all tiles.
    """

    def __init__(
        self,
        max_rms: float | None = 1.2,
        min_rms: float | None = None,
        tile_size: u.Quantity = 0.5 * u.deg,
        grow_tiles: int = 0,
        log: FilteringBoundLogger | None = None,
    ):
        self.max_rms = max_rms
        self.min_rms = min_rms
        self.tile_size = tile_size
        self.grow_tiles = grow_tiles
        self.log = log or structlog.get_logger()

    def postprocess(self, input_map: ProcessableMap) -> ProcessableMap:
        log = self.log.bind(
            func_name="snr_rms_mask.postprocess", map_name=input_map.map_name
        )
        snr = np.asarray(input_map.snr)
        valid = np.isfinite(snr) & (snr != 0)
        if input_map.mask is not None:
            valid &= np.asarray(input_map.mask) > 0
        tile_pix = max(
            1,
            int(round((self.tile_size / input_map.res).to_value(""))),
        )
        bad, rms = snr_rms_tile_mask(
            snr,
            valid,
            tile_pix,
            max_rms=self.max_rms,
            min_rms=self.min_rms,
            grow_tiles=self.grow_tiles,
        )
        finite = np.isfinite(rms)
        log.info(
            "snr_rms_mask.tiles",
            n_tiles=int(finite.sum()),
            tile_rms_median=float(np.median(rms[finite])) if finite.any() else None,
            n_above_max=int((rms[finite] > self.max_rms).sum())
            if self.max_rms is not None
            else 0,
            n_below_min=int((rms[finite] < self.min_rms).sum())
            if self.min_rms is not None
            else 0,
            masked_fraction=round(float(bad[valid].mean()), 4) if valid.any() else 0.0,
        )
        if not bad.any():
            return input_map

        good = enmap.ndmap((~bad).astype(np.float32), input_map.snr.wcs)
        input_map.mask = good if input_map.mask is None else input_map.mask * good
        input_map.snr *= good
        input_map.flux *= good
        return input_map


class PixellFlatFielder(MapPostprocessor):
    tile_size: AstroPydanticQuantity = AstroPydanticQuantity(1.0 * u.deg)
    sigma_val: float = 5.0

    def __init__(
        self,
        sigma_val: float = 5.0,
        tile_size: u.Quantity = 1.0 * u.deg,
        mask: enmap.ndmap | None = None,
        log: FilteringBoundLogger | None = None,
    ):
        self.sigma_val = sigma_val
        self.tile_size = tile_size
        self.mask = mask
        self.log = log or structlog.get_logger()

    def postprocess(self, input_map: ProcessableMap) -> ProcessableMap:
        return flat_field_using_pixell(
            mapdata=input_map,
            tilegrid=self.tile_size.to_value(u.deg),
            log=self.log,
        )


class PhotutilsFlatFielder(MapPostprocessor):
    tile_size: AstroPydanticQuantity = AstroPydanticQuantity(1.0 * u.deg)
    sigma_val: float = 5.0

    def __init__(
        self,
        sigma_val: float = 5.0,
        tile_size: u.Quantity = 1.0 * u.deg,
        mask: enmap.ndmap | None = None,
        log: FilteringBoundLogger | None = None,
    ):
        self.sigma_val = sigma_val
        self.tile_size = tile_size
        self.mask = mask
        self.log = log or structlog.get_logger()

    def postprocess(self, input_map: ProcessableMap) -> ProcessableMap:
        if self.mask is not None and input_map.mask is not None:
            combined_mask = self.mask * input_map.mask
        elif self.mask is not None:
            combined_mask = self.mask
        elif input_map.mask is not None:
            combined_mask = input_map.mask
        else:
            combined_mask = None
        return flat_field_using_photutils(
            mapdata=input_map,
            tilegrid=self.tile_size,
            mask=combined_mask,
            sigmaclip=self.sigma_val,
            log=self.log,
        )
