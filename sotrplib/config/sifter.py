from abc import ABC, abstractmethod
from typing import Literal

from astropy import units as u
from astropydantic import AstroPydanticQuantity
from pydantic import BaseModel
from structlog.types import FilteringBoundLogger

from sotrplib.sifter.core import (
    DefaultSifter,
    EmptySifter,
    SiftingProvider,
    SimpleCatalogSifter,
)


class SifterConfig(BaseModel, ABC):
    """Abstract base for sifter configuration objects.

    Sifters cross-match pipeline candidates against source catalogs and
    classify results.  Each subclass defines a ``sifter_type`` discriminator
    and implements ``to_sifter``.
    """

    sifter_type: str

    @abstractmethod
    def to_sifter(self, log: FilteringBoundLogger | None = None) -> SiftingProvider:
        """Construct the configured sifter.

        Parameters
        ----------
        log : FilteringBoundLogger, optional
            Structured logger.

        Returns
        -------
        SiftingProvider
            The constructed sifter instance.
        """
        return


class EmptySifterConfig(SifterConfig):
    """Configuration for a no-op sifter that passes all candidates unchanged."""

    sifter_type: Literal["empty"] = "empty"

    def to_sifter(self, log: FilteringBoundLogger | None = None) -> EmptySifter:
        return EmptySifter()


class SimpleCatalogSifterConfig(SifterConfig):
    """Configuration for the simple catalog cross-match sifter.

    Fields
    ------
    radius : Quantity[arcmin]
        Match radius (default 1 arcmin).
    method : {"closest", "all"}
        Whether to return only the closest match or all matches within
        ``radius`` (default ``"closest"``).
    """

    sifter_type: Literal["simple"] = "simple"
    radius: AstroPydanticQuantity[u.arcmin] = 1.0 * u.arcmin
    method: Literal["closest", "all"] = "closest"

    def to_sifter(self, log: FilteringBoundLogger | None = None) -> SimpleCatalogSifter:
        return SimpleCatalogSifter(
            radius=self.radius,
            method=self.method,
            log=log,
        )


class DefaultSifterConfig(SifterConfig):
    """Configuration for the default pipeline sifter.

    Fields
    ------
    min_match_radius : Quantity[arcmin]
        Minimum cross-match radius (default 1.5 arcmin).
    """

    sifter_type: Literal["default"] = "default"
    min_match_radius: AstroPydanticQuantity[u.arcmin] = 1.5 * u.arcmin

    def to_sifter(self, log: FilteringBoundLogger | None = None) -> SiftingProvider:
        return DefaultSifter(
            min_match_radius=self.min_match_radius,
            log=log,
        )


AllSifterConfigTypes = (
    EmptySifterConfig | DefaultSifterConfig | SimpleCatalogSifterConfig
)
