from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

from astropy import units as u
from astropydantic import AstroPydanticQuantity
from pydantic import BaseModel
from structlog.types import FilteringBoundLogger

from sotrplib.source_catalog.core import SourceCatalog
from sotrplib.source_catalog.socat import SOCatFITSCatalog, SOCatWebskyFITSCatalog


class SourceCatalogConfig(BaseModel, ABC):
    catalog_type: str

    @abstractmethod
    def to_source_catalog(
        self, log: FilteringBoundLogger | None = None
    ) -> SourceCatalog:
        return


class EmptySourceCatalogConfig(SourceCatalogConfig):
    catalog_type: Literal["empty"] = "empty"
    path: Path | None = None

    def to_source_catalog(self, log=None) -> SOCatFITSCatalog:
        return SOCatFITSCatalog(
            log=log,
        )


class SOCatFITSCatalogConfig(SourceCatalogConfig):
    catalog_type: Literal["socat", "fits"] = "fits"
    path: Path | None = None
    hdu: int = 1
    flux_lower_limit: AstroPydanticQuantity = 0.03 * u.Jy

    def to_source_catalog(self, log=None) -> SOCatFITSCatalog:
        return SOCatFITSCatalog(
            path=self.path,
            hdu=self.hdu,
            flux_lower_limit=self.flux_lower_limit,
            log=log,
        )


class SOCatFITSWebskyCatalogConfig(SourceCatalogConfig):
    catalog_type: Literal["websky"] = "websky"
    path: Path | None = None
    hdu: int = 1
    flux_lower_limit: AstroPydanticQuantity = 0.03 * u.Jy

    def to_source_catalog(self, log=None) -> SOCatWebskyFITSCatalog:
        return SOCatWebskyFITSCatalog(
            path=self.path,
            hdu=self.hdu,
            flux_lower_limit=self.flux_lower_limit,
            log=log,
        )


AllSourceCatalogConfigTypes = (
    EmptySourceCatalogConfig | SOCatFITSCatalogConfig | SOCatFITSWebskyCatalogConfig
)
