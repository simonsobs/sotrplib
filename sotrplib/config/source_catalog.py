from abc import ABC, abstractmethod
from pathlib import Path
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


class SOCatUpdateConfig(BaseModel):
    """
    How the runner updates socat after map matching. See
    SOCat.update.
    """

    add_confirmed_transients: bool = True
    "Add each confirmed map-match group as a new monitored fixed source."
    duplicate_radius: AstroPydanticQuantity[u.arcmin] = 1.0 * u.arcmin
    "Do not add a group if a socat source is within this distance."
    name_prefix: str = "SOT"
    'A new source is named "<name_prefix> J<hhmmss.s><+ddmmss>".'
    record_directory: Path | None = None
    "If set, write a JSON record of each update (socat_update_<start>.json)."


class SOCatConfig(SourceCatalogConfig):
    catalog_type: Literal["socat"] = "socat"
    update: SOCatUpdateConfig | None = None
    "If set, update socat with the confirmed map-match groups of each run."

    def to_source_catalog(self, log=None) -> SOCat:
        if self.update is None:
            return SOCat(log=log)
        return SOCat(
            log=log,
            add_confirmed_transients=self.update.add_confirmed_transients,
            duplicate_radius=self.update.duplicate_radius,
            name_prefix=self.update.name_prefix,
            record_directory=self.update.record_directory,
        )


AllSourceCatalogConfigTypes = SOCatConfig | EmptySourceCatalogConfig
