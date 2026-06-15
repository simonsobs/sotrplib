from abc import ABC, abstractmethod
from typing import Literal

import astropy.units as u
from astropydantic import AstroPydanticQuantity
from numpydantic import NDArray
from pydantic import BaseModel
from structlog.types import FilteringBoundLogger

from sotrplib.sources.blind import (
    BlindSearchParameters,
    BlindSearchProvider,
    EmptyBlindSearch,
    SigmaClipBlindSearch,
)


class BlindSearchConfig(BaseModel, ABC):
    """Abstract base for blind-search configuration objects.

    Each subclass defines a ``search_type`` discriminator and implements
    ``to_search_provider`` to construct a ``BlindSearchProvider``.
    """

    search_type: str

    @abstractmethod
    def to_search_provider(
        self, log: FilteringBoundLogger | None = None
    ) -> BlindSearchProvider:
        """Construct the configured blind-search provider.

        Parameters
        ----------
        log : FilteringBoundLogger, optional
            Structured logger.

        Returns
        -------
        BlindSearchProvider
            The constructed search provider instance.
        """
        return


class EmptyBlindSearchConfig(BlindSearchConfig):
    """Configuration for a no-op blind-search provider."""

    search_type: Literal["empty"] = "empty"

    def to_search_provider(
        self, log: FilteringBoundLogger | None = None
    ) -> EmptyBlindSearch:
        return EmptyBlindSearch()


class PhotutilsBlindSearchConfig(BlindSearchConfig):
    """Configuration for the sigma-clip blind source finder.

    Fields
    ------
    parameters : BlindSearchParameters or None
        Detection thresholds and fitting parameters.  Uses defaults if ``None``.
    pixel_mask : NDArray or None
        Additional pixel mask applied before source detection.
    thumbnail_half_width : Quantity[deg] or None
        Half-width of thumbnails stored for each detection.
    """

    search_type: Literal["photutils"] = "photutils"
    parameters: BlindSearchParameters | None = None
    pixel_mask: NDArray | None = None
    thumbnail_half_width: AstroPydanticQuantity[u.deg] | None = None

    def to_search_provider(
        self, log: FilteringBoundLogger | None = None
    ) -> SigmaClipBlindSearch:
        return SigmaClipBlindSearch(
            parameters=self.parameters,
            pixel_mask=self.pixel_mask,
            thumbnail_half_width=self.thumbnail_half_width,
            log=log,
        )


AllBlindSearchConfigTypes = EmptyBlindSearchConfig | PhotutilsBlindSearchConfig
