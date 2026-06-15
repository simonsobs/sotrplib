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
)


class PostprocessorConfig(BaseModel, ABC):
    """Abstract base for postprocessor configuration objects.

    Each subclass defines a ``postprocessor_type`` discriminator and implements
    ``to_postprocessor`` to construct a ``MapPostprocessor``.
    """

    postprocessor_type: str

    @abstractmethod
    def to_postprocessor(
        self, log: FilteringBoundLogger | None = None
    ) -> MapPostprocessor:
        """Construct the configured postprocessor.

        Parameters
        ----------
        log : FilteringBoundLogger, optional
            Structured logger.

        Returns
        -------
        MapPostprocessor
            The constructed postprocessor instance.
        """
        return


class GalaxyMaskConfig(PostprocessorConfig):
    """Configuration for the galactic-dust mask postprocessor.

    Fields
    ------
    mask_path : Path
        Path to the FITS galactic mask file.
    invert : bool
        If ``True``, invert the mask (default ``False``).
    """

    postprocessor_type: Literal["galaxy_mask"] = "galaxy_mask"
    mask_path: Path
    invert: bool = False

    def to_postprocessor(self, log: FilteringBoundLogger | None = None) -> GalaxyMask:
        return GalaxyMask(mask_path=self.mask_path, invert=self.invert, log=log)


class FlatfieldConfig(PostprocessorConfig):
    """Configuration for the photutils flat-fielding postprocessor.

    Fields
    ------
    sigma_val : float
        Sigma-clipping threshold for background estimation (default 5.0).
    tile_size : Quantity[deg]
        Tile size for the background grid (default 1 deg).
    """

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
