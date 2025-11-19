"""
Dependencies for post-processing (e.g. flat-fielding). Affects
maps in-place. These operations happen after 'finalize', i.e. once
the snr and flux arrays already exist.
"""

from abc import ABC, abstractmethod
from pathlib import Path

import structlog
from astropy import units as u
from astropydantic import AstroPydanticQuantity
from pixell import enmap
from structlog.types import FilteringBoundLogger

from sotrplib.maps.core import ProcessableMap
from sotrplib.maps.maps import flat_field_using_photutils, flat_field_using_pixell
from sotrplib.maps.masks import mask_dustgal


class MapPostprocessor(ABC):
    @abstractmethod
    def postprocess(self, input_map: ProcessableMap) -> ProcessableMap:
        """
        Provides postprocessing for the input map
        """
        return input_map


class GalaxyMask(MapPostprocessor):
    mask_filename: Path

    def __init__(self, mask_filename: Path):
        self.mask_filename = mask_filename

    def postprocess(self, input_map: ProcessableMap) -> ProcessableMap:
        galaxy_mask = mask_dustgal(
            imap=input_map.flux,
            galmask=enmap.read_map(
                fname=self.mask_filename,
                box=input_map.box,
            ),
        )

        input_map.flux *= galaxy_mask
        input_map.snr *= galaxy_mask

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
        return flat_field_using_photutils(
            mapdata=input_map,
            tilegrid=self.tile_size,
            mask=self.mask,
            sigmaclip=self.sigma_val,
            log=self.log,
        )
