from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

from astropy import units as u
from astropydantic import AstroPydanticQuantity
from pydantic import BaseModel
from structlog.types import FilteringBoundLogger

from sotrplib.source_catalog.database import (
    EmptyMockSourceCatalog,
    MockDatabase,
    MockSourceCatalog,
)


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
    ) -> EmptyMockSourceCatalog:
        return EmptyMockSourceCatalog(log=log)


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


AllSourceCatalogConfigTypes = EmptySourceCatalogConfig | MockSourceCatalogConfig
