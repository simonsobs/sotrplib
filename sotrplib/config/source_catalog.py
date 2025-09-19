from abc import ABC, abstractmethod
from typing import Literal

from astropy import units as u
from pydantic import BaseModel
from structlog.types import FilteringBoundLogger

from sotrplib.source_catalog.database import (
    MockACTDatabase,
    MockDatabase,
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
    ) -> MockDatabase:
        return MockDatabase(log=log)


class MockSourceCatalogConfig(SourceCatalogConfig):
    catalog_type: Literal["mock_socat"] = "mock_socat"
    db_path: str | None = None
    flux_lower_limit: float = 0.03
    flux_units: str = "Jy"

    def to_source_catalog(
        self, log: FilteringBoundLogger | None = None
    ) -> MockACTDatabase:
        return MockACTDatabase(
            db_path=self.db_path,
            flux_lower_limit=u.Quantity(self.flux_lower_limit, self.flux_units),
            log=log,
        )


AllSourceCatalogConfigTypes = EmptySourceCatalogConfig | MockSourceCatalogConfig
