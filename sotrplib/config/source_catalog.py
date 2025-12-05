from abc import ABC, abstractmethod
from typing import Literal

from astropy import units as u
from astropydantic import AstroPydanticQuantity
from pydantic import BaseModel
from structlog.types import FilteringBoundLogger

from sotrplib.source_catalog.core import RegisteredSourceCatalog, SourceCatalog
from sotrplib.source_catalog.socat import SOCat


class SourceCatalogConfig(BaseModel, ABC):
    catalog_type: str

    @abstractmethod
    def to_source_catalog(
        self, log: FilteringBoundLogger | None = None
    ) -> SourceCatalog:
        return


class EmptySourceCatalogConfig(SourceCatalogConfig):
    catalog_type: Literal["empty"] = "empty"

    def to_source_catalog(
        self, log: FilteringBoundLogger | None = None
    ) -> RegisteredSourceCatalog:
        return RegisteredSourceCatalog(sources=[])


class SOCatConfig(SourceCatalogConfig):
    catalog_type: Literal["socat"] = "socat"
    flux_lower_limit: AstroPydanticQuantity = 0.03 * u.Jy

    def to_source_catalog(self, log=None) -> SOCat:
        return SOCat(
            flux_lower_limit=self.flux_lower_limit,
            log=log,
        )


AllSourceCatalogConfigTypes = SOCatConfig | EmptySourceCatalogConfig
