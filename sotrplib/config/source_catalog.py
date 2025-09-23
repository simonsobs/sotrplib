from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

from astropy import units as u
from astropydantic import AstroPydanticQuantity
from pydantic import BaseModel
from structlog.types import FilteringBoundLogger

from sotrplib.source_catalog.database import (
    MockDatabase,
    MockSourceCatalog,
)
from sotrplib.source_catalog.socat import SOCatFITSCatalog


class SourceCatalogConfig(BaseModel, ABC):
    catalog_type: str

    @abstractmethod
    def to_source_catalog(
        self, log: FilteringBoundLogger | None = None
    ) -> MockDatabase:
        return


class EmptySourceCatalogConfig(SourceCatalogConfig):
    catalog_type: Literal["empty"] = "empty"

    def to_source_catalog(
        self, log: FilteringBoundLogger | None = None
    ) -> MockDatabase:
        return MockDatabase(log=log)


class MockSourceCatalogConfig(SourceCatalogConfig):
    catalog_type: Literal["mock_socat"] = "mock_socat"
    db_path: Path | None = None
    flux_lower_limit: AstroPydanticQuantity = 0.03 * u.Jy

    def to_source_catalog(
        self, log: FilteringBoundLogger | None = None
    ) -> MockSourceCatalog:
        return MockSourceCatalog(
            db_path=self.db_path,
            flux_lower_limit=self.flux_lower_limit,
            log=log,
        )


class SOCatFITSCatalogConfig(SourceCatalogConfig):
    catalog_type: Literal["fits"] = "fits"
    path: Path
    hdu: int = 1
    flux_lower_limit: AstroPydanticQuantity = 0.03 * u.Jy

    def to_source_catalog(self, log=None) -> SOCatFITSCatalog:
        return SOCatFITSCatalog(
            path=self.path,
            hdu=self.hdu,
            flux_lower_limit=self.flux_lower_limit,
            log=log,
        )


AllSourceCatalogConfigTypes = (
    EmptySourceCatalogConfig | MockSourceCatalogConfig | SOCatFITSCatalogConfig
)
