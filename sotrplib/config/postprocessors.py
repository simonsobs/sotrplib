from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

from astropy import units as u
from astropydantic import AstroPydanticQuantity
from pydantic import BaseModel
from structlog.types import FilteringBoundLogger

from sotrplib.maps.postprocessor import (
    GalaxyMask,
    # EdgeMask,
    # PixellFlatfield,
    # PhotutilsFlatField,
    MapPostprocessor,
    PhotutilsFlatFielder,
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
    mask_filename: Path

    def to_postprocessor(self, log: FilteringBoundLogger | None = None) -> GalaxyMask:
        return GalaxyMask(mask_filename=self.mask_filename)


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


AllPostprocessorConfigTypes = GalaxyMaskConfig | FlatfieldConfig
