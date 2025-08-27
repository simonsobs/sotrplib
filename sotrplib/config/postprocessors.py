from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

from pydantic import BaseModel
from structlog.types import FilteringBoundLogger

from sotrplib.maps.postprocessor import (
    GalaxyMask,
    # EdgeMask,
    # PixellFlatfield,
    # PhotutilsFlatField,
    MapPostprocessor,
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


AllPostprocessorConfigTypes = GalaxyMaskConfig
