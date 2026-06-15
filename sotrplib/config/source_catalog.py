from abc import ABC, abstractmethod
from typing import Literal

from astropy import units as u
from astropydantic import AstroPydanticICRS, AstroPydanticQuantity
from pydantic import BaseModel
from structlog.types import FilteringBoundLogger

from sotrplib.source_catalog.core import RegisteredSourceCatalog, SourceCatalog
from sotrplib.source_catalog.socat import SOCat


class SourceCatalogConfig(BaseModel, ABC):
    """Abstract base for source catalog configuration objects.

    Each subclass defines a ``catalog_type`` discriminator and implements
    ``to_source_catalog`` to construct a ``SourceCatalog``.
    """

    catalog_type: str

    @abstractmethod
    def to_source_catalog(
        self, log: FilteringBoundLogger | None = None
    ) -> SourceCatalog:
        """Construct the configured source catalog.

        Parameters
        ----------
        log : FilteringBoundLogger, optional
            Structured logger.

        Returns
        -------
        SourceCatalog
            The constructed catalog instance.
        """
        return


class EmptySourceCatalogConfig(SourceCatalogConfig):
    """Configuration for an empty (no-op) source catalog."""

    catalog_type: Literal["empty"] = "empty"

    def to_source_catalog(
        self, log: FilteringBoundLogger | None = None
    ) -> RegisteredSourceCatalog:
        return RegisteredSourceCatalog(sources=[])


class SOCatConfig(SourceCatalogConfig):
    """Configuration for the Simons Observatory source catalog (SOCat).

    Fields
    ------
    flux_lower_limit : Quantity
        Minimum source flux to include (default 0.03 Jy).
    additional_positions : list of AstroPydanticICRS or None
        Extra sky positions to add to the catalog.
    additional_fluxes : list of Quantity or None
        Flux estimates for the extra positions.
    additional_source_ids : list of str or None
        String identifiers for the extra sources.
    """

    catalog_type: Literal["socat"] = "socat"
    flux_lower_limit: AstroPydanticQuantity = 0.03 * u.Jy
    additional_positions: list[AstroPydanticICRS] | None = None
    additional_fluxes: list[AstroPydanticQuantity] | None = None
    additional_source_ids: list[str] | None = None

    def to_source_catalog(self, log=None) -> SOCat:
        return SOCat(
            flux_lower_limit=self.flux_lower_limit,
            additional_positions=self.additional_positions,
            additional_fluxes=self.additional_fluxes,
            additional_source_ids=self.additional_source_ids,
            log=log,
        )


AllSourceCatalogConfigTypes = SOCatConfig | EmptySourceCatalogConfig
