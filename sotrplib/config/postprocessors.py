from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

from astropy import units as u
from astropydantic import AstroPydanticQuantity
from pydantic import BaseModel
from structlog.types import FilteringBoundLogger

from sotrplib.maps.postprocessor import (
    GalaxyMask,
    MapPostprocessor,
    PhotutilsFlatFielder,
    SnrRmsMask,
)


class PostprocessorConfig(BaseModel, ABC):
    postprocessor_type: str

    @abstractmethod
    def to_postprocessor(
        self, log: FilteringBoundLogger | None = None
    ) -> MapPostprocessor:
        return


class GalaxyMaskConfig(PostprocessorConfig):
    postprocessor_type: Literal["galaxy_mask"] = "galaxy_mask"
    mask_path: Path
    invert: bool = False

    def to_postprocessor(self, log: FilteringBoundLogger | None = None) -> GalaxyMask:
        return GalaxyMask(mask_path=self.mask_path, invert=self.invert, log=log)


class FlatfieldConfig(PostprocessorConfig):
    postprocessor_type: Literal["flatfield"] = "flatfield"
    sigma_val: float = 5.0
    tile_size: AstroPydanticQuantity[u.deg] = AstroPydanticQuantity(1.0 * u.deg)

    def to_postprocessor(
        self, log: FilteringBoundLogger | None = None
    ) -> MapPostprocessor:
        return PhotutilsFlatFielder(
            sigma_val=self.sigma_val, tile_size=self.tile_size, log=log
        )


class SnrRmsMaskConfig(PostprocessorConfig):
    """
    Mask the tiles where the robust rms of the snr is outside
    [min_rms, max_rms]. Put it before the flatfield.
    """

    postprocessor_type: Literal["snr_rms_mask"] = "snr_rms_mask"
    max_rms: float | None = 1.2
    min_rms: float | None = None
    tile_size: AstroPydanticQuantity[u.deg] = AstroPydanticQuantity(0.5 * u.deg)
    grow_tiles: int = 0

    def to_postprocessor(self, log: FilteringBoundLogger | None = None) -> SnrRmsMask:
        return SnrRmsMask(
            max_rms=self.max_rms,
            min_rms=self.min_rms,
            tile_size=self.tile_size,
            grow_tiles=self.grow_tiles,
            log=log,
        )


AllPostprocessorConfigTypes = GalaxyMaskConfig | FlatfieldConfig | SnrRmsMaskConfig
